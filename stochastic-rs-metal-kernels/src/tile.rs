use metal_oxide_device::{
  DynamicThreadgroupBuffer, ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, math,
  thread_idx, threadgroup::barrier,
};

/// # Safety
/// Each block owns a tile of 2 * block_dim.x elements within one length-n transform.
/// The grid covers complete transforms; both scratch allocations cover a tile.
/// eigs and reverse cover n entries; stages fits the tile and buffers are disjoint.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gen_tile_fft(
  real: WriteBuffer<f32>,
  imag: WriteBuffer<f32>,
  eigs: ReadBuffer<f32>,
  reverse: ReadBuffer<u32>,
  n: u32,
  seed: u32,
  base_cell: u64,
  stages: u32,
  shared_real: DynamicThreadgroupBuffer<f32>,
  shared_imag: DynamicThreadgroupBuffer<f32>,
) {
  if n == 0 {
    return;
  }
  let lane = thread_idx().x;
  let half_tile = block_dim().x;
  let tile = half_tile * 2;
  let tile_base = block_idx().x * tile;
  let batch = tile_base / n;
  let within = tile_base % n;
  let cell_base = base_cell.wrapping_add((batch as u64).wrapping_mul(2).wrapping_mul(n as u64));
  // SAFETY: the caller supplies complete tiles and two initialized input tables.
  // Every lane reaches each barrier, including the read-before-write stage barrier.
  unsafe {
    let mut k = lane;
    while k < tile {
      let natural = reverse.load_unchecked(within + k);
      let pair = crate::random::normal_pair(cell_base.wrapping_add(natural as u64), seed);
      let eig = eigs.load_unchecked(natural);
      shared_real.store_unchecked(k, pair.0 * eig);
      shared_imag.store_unchecked(k, pair.1 * eig);
      k += half_tile;
    }
    barrier();
    let mut stage = 0;
    while stage < stages {
      let half = 1_u32 << stage;
      if half > 0 {
        let stride = half * 2;
        let group = lane / half;
        let pos = lane % half;
        let i = group * stride + pos;
        let j = i + half;
        let angle = -2.0 * core::f32::consts::PI * pos as f32 / stride as f32;
        let wr = math::cos(angle);
        let wi = math::sin(angle);
        let tr = shared_real.load_unchecked(j) * wr - shared_imag.load_unchecked(j) * wi;
        let ti = shared_real.load_unchecked(j) * wi + shared_imag.load_unchecked(j) * wr;
        let ar = shared_real.load_unchecked(i);
        let ai = shared_imag.load_unchecked(i);
        barrier();
        shared_real.store_unchecked(i, ar + tr);
        shared_imag.store_unchecked(i, ai + ti);
        shared_real.store_unchecked(j, ar - tr);
        shared_imag.store_unchecked(j, ai - ti);
        barrier();
      }
      stage += 1;
    }
    let mut k = lane;
    while k < tile {
      real.store_unchecked(tile_base + k, shared_real.load_unchecked(k));
      imag.store_unchecked(tile_base + k, shared_imag.load_unchecked(k));
      k += half_tile;
    }
  }
}
