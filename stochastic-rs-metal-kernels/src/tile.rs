//! The FFT tile on bit-reversed input: stage 0 in registers, then SIMD shuffles, with one
//! threadgroup exchange moving position bits 6.. into the lanes.

use metal_oxide_device::{
  DynamicThreadgroupBuffer, ReadBuffer, ReadWriteBuffer, WriteBuffer, block_idx, kernel, simdgroup,
  thread_idx, threadgroup::barrier, uninit_array,
};

use crate::fft::{Complex, Readout, put, twiddle};

/// Threadgroup word for tile position `p`; one pad word per 32 avoids bank conflicts.
pub(crate) fn slot(p: u32) -> u32 {
  p + (p >> 5)
}

/// Tile position of value `k` of thread `i`. Before the exchange (`high == 0`) the lane bits
/// address stages 1-5; after it the low `high` thread bits address stages 6..6 + high.
pub(crate) fn position(i: u32, k: u32, high: u32) -> u32 {
  if high == 0 {
    ((i >> 5) << 6) | ((i & 31) << 1) | k
  } else {
    k | (((i >> high) & 31) << 1) | ((i & ((1 << high) - 1)) << 6)
  }
}

/// # Safety
/// Every lane of the SIMD group executes this with the same mask.
unsafe fn shuffle(value: Complex, mask: u32) -> Complex {
  // SAFETY: the caller keeps the SIMD group convergent.
  unsafe {
    Complex {
      re: simdgroup::shuffle_xor(value.re, mask),
      im: simdgroup::shuffle_xor(value.im, mask),
    }
  }
}

/// A DIT butterfly whose pair is split across two lanes; `upper` lanes hold the second value.
fn butterfly(value: Complex, other: Complex, w: Complex, upper: bool) -> Complex {
  if upper {
    other.sub(value.mul(w))
  } else {
    value.add(other.mul(w))
  }
}

/// Runs stages `from..to` of one layout. Safety: the block is convergent and the twiddle
/// table holds 2^(table_log - 1) entries, with table_log >= to.
unsafe fn lane_stages(
  mut values: [Complex; 2],
  twiddles: ReadBuffer<f32>,
  table_log: u32,
  from: u32,
  to: u32,
  high: u32,
) -> [Complex; 2] {
  let i = thread_idx().x;
  let lane_bit = if high == 0 { 1 } else { 6 };
  for s in from..to {
    let mask = 1 << (s - lane_bit);
    let upper = i & mask != 0;
    for k in 0..2 {
      let pos = position(i, k, high) & ((1 << s) - 1);
      // SAFETY: pos < 2^s indexes the table at stride 2^(table_log - s - 1).
      unsafe {
        let w = twiddle(twiddles, pos << (table_log - s - 1));
        let other = shuffle(values[k as usize], mask);
        values[k as usize] = butterfly(values[k as usize], other, w, upper);
      }
    }
  }
  values
}

/// Transforms the tile held at `position(i, 0..2, 0)`; afterwards `p` sits at `slot(p)`. Safety:
/// the whole block calls it, table_log >= tile_log and scratch covers slot(2^tile_log).
pub(crate) unsafe fn stages(
  mut values: [Complex; 2],
  twiddles: ReadBuffer<f32>,
  table_log: u32,
  tile_log: u32,
  shared_real: DynamicThreadgroupBuffer<f32>,
  shared_imag: DynamicThreadgroupBuffer<f32>,
) {
  let i = thread_idx().x;
  // SAFETY: positions stay inside the tile and both scratch words; every thread reaches each
  // barrier and shuffle with the same arguments.
  unsafe {
    let (a, b) = (values[0], values[1]);
    values[0] = a.add(b);
    values[1] = a.sub(b);
    let first = if tile_log < 6 { tile_log } else { 6 };
    values = lane_stages(values, twiddles, table_log, 1, first, 0);
    for k in 0..2 {
      let p = position(i, k, 0);
      shared_real.store_unchecked(slot(p), values[k as usize].re);
      shared_imag.store_unchecked(slot(p), values[k as usize].im);
    }
    if tile_log > 6 {
      let u = tile_log - 6;
      barrier();
      for k in 0..2 {
        let p = position(i, k, u);
        values[k as usize] = Complex {
          re: shared_real.load_unchecked(slot(p)),
          im: shared_imag.load_unchecked(slot(p)),
        };
      }
      values = lane_stages(values, twiddles, table_log, 6, tile_log, u);
      barrier();
      for k in 0..2 {
        let p = position(i, k, u);
        shared_real.store_unchecked(slot(p), values[k as usize].re);
        shared_imag.store_unchecked(slot(p), values[k as usize].im);
      }
    }
    barrier();
  }
}

/// Generates and transforms one tile, leaving position `p` at word `slot(p)` after a barrier.
/// Safety: every thread of the block calls this under the `gen_tile_fft` contract.
#[allow(clippy::too_many_arguments)]
unsafe fn transform(
  eigs: ReadBuffer<f32>,
  twiddles: ReadBuffer<f32>,
  log_n: u32,
  tile_log: u32,
  seed: u32,
  base_cell: u64,
  shared_real: DynamicThreadgroupBuffer<f32>,
  shared_imag: DynamicThreadgroupBuffer<f32>,
) -> (u32, u32) {
  let i = thread_idx().x;
  let tile_base = block_idx().x << tile_log;
  let batch = tile_base >> log_n;
  let within = tile_base & ((1 << log_n) - 1);
  let cell_base =
    base_cell.wrapping_add((batch as u64).wrapping_mul(2).wrapping_mul(1_u64 << log_n));
  // SAFETY: positions stay inside the transform; every thread reaches the tile's barriers.
  unsafe {
    let mut values: [Complex; 2] = uninit_array();
    for k in 0..2 {
      let p = within + position(i, k, 0);
      let natural = p.reverse_bits() >> (32 - log_n);
      let pair = crate::random::normal_pair(cell_base.wrapping_add(natural as u64), seed);
      let eig = eigs.load_unchecked(p);
      values[k as usize] = Complex {
        re: pair.0 * eig,
        im: pair.1 * eig,
      };
    }
    stages(
      values,
      twiddles,
      tile_log,
      tile_log,
      shared_real,
      shared_imag,
    );
  }
  (batch, tile_base)
}

/// # Safety
/// One block of 2^(tile_log - 1) threads (32-lane SIMD) per tile, tile_log < log_n; eigs covers a
/// transform, twiddles 2^(tile_log - 1), scratch slot(2^tile_log), real/imag the grid; disjoint.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gen_tile_fft(
  real: WriteBuffer<f32>,
  imag: WriteBuffer<f32>,
  eigs: ReadBuffer<f32>,
  twiddles: ReadBuffer<f32>,
  log_n: u32,
  tile_log: u32,
  seed: u32,
  base_cell: u64,
  shared_real: DynamicThreadgroupBuffer<f32>,
  shared_imag: DynamicThreadgroupBuffer<f32>,
) {
  // SAFETY: forwarded entry contract; each thread writes two distinct positions of its tile.
  unsafe {
    let (_, tile_base) = transform(
      eigs,
      twiddles,
      log_n,
      tile_log,
      seed,
      base_cell,
      shared_real,
      shared_imag,
    );
    let i = thread_idx().x;
    let half = 1 << (tile_log - 1);
    for k in 0..2 {
      let p = i + k * half;
      real.store_unchecked(tile_base + p, shared_real.load_unchecked(slot(p)));
      imag.store_unchecked(tile_base + p, shared_imag.load_unchecked(slot(p)));
    }
  }
}

/// # Safety
/// One block of 2^(log_n - 1) threads (32-lane SIMD) per transform; eigs covers it, twiddles
/// 2^(log_n - 1), scratch slot(2^log_n) and output rows * size values; all disjoint.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gen_fft_extract(
  eigs: ReadBuffer<f32>,
  twiddles: ReadBuffer<f32>,
  log_n: u32,
  seed: u32,
  base_cell: u64,
  output: WriteBuffer<f32>,
  size: u32,
  scale: f32,
  rows: u32,
  parity: u32,
  shared_real: DynamicThreadgroupBuffer<f32>,
  shared_imag: DynamicThreadgroupBuffer<f32>,
) {
  let config = Readout {
    size,
    scale,
    rows,
    parity,
  };
  // SAFETY: forwarded entry contract; each thread reads out two distinct frequencies.
  unsafe {
    let (batch, _) = transform(
      eigs,
      twiddles,
      log_n,
      log_n,
      seed,
      base_cell,
      shared_real,
      shared_imag,
    );
    let i = thread_idx().x;
    let half = 1 << (log_n - 1);
    for k in 0..2 {
      let p = i + k * half;
      let value = Complex {
        re: shared_real.load_unchecked(slot(p)),
        im: shared_imag.load_unchecked(slot(p)),
      };
      put(output, config, batch, p, value);
    }
  }
}

/// # Safety
/// One block of 2^(tile_log - 1) threads (32-lane SIMD) per tile of bit-reversed length-2^log_n
/// transforms, tile_log <= log_n; twiddles hold 2^(log_n - 1), scratch slot(2^tile_log).
#[kernel]
pub unsafe fn fft_tile(
  real: ReadWriteBuffer<f32>,
  imag: ReadWriteBuffer<f32>,
  twiddles: ReadBuffer<f32>,
  log_n: u32,
  tile_log: u32,
  shared_real: DynamicThreadgroupBuffer<f32>,
  shared_imag: DynamicThreadgroupBuffer<f32>,
) {
  let i = thread_idx().x;
  let tile_base = block_idx().x << tile_log;
  // SAFETY: each block owns one tile; every thread loads and stores two distinct positions.
  unsafe {
    let mut values: [Complex; 2] = uninit_array();
    for k in 0..2 {
      let p = tile_base + position(i, k, 0);
      values[k as usize] = Complex {
        re: real.load_unchecked(p),
        im: imag.load_unchecked(p),
      };
    }
    stages(values, twiddles, log_n, tile_log, shared_real, shared_imag);
    let half = 1 << (tile_log - 1);
    for k in 0..2 {
      let p = i + k * half;
      real.store_unchecked(tile_base + p, shared_real.load_unchecked(slot(p)));
      imag.store_unchecked(tile_base + p, shared_imag.load_unchecked(slot(p)));
    }
  }
}
