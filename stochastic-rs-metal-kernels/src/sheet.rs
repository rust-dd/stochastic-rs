//! Two-dimensional intrinsic embedding and Stein's linear correction.
//! Kroese & Botev (2015), Spatial process simulation, §4.4, arXiv:1308.0399.

use metal_oxide_device::{
  DynamicThreadgroupBuffer, ReadBuffer, WriteBuffer, block_idx, kernel, thread_idx, uninit_array,
};

use crate::fft::Complex;
use crate::tile::{position, slot, stages};

/// # Safety
/// One block of 2^(tile_log - 1) threads (32-lane SIMD) per row tile, 1 <= tile_log <= log_cols;
/// lam covers a sheet of 2^log_cells, twiddles 2^(log_cols - 1), scratch slot(2^tile_log).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn sheet_rows(
  real: WriteBuffer<f32>,
  imag: WriteBuffer<f32>,
  lam: ReadBuffer<f32>,
  twiddles: ReadBuffer<f32>,
  log_cols: u32,
  log_cells: u32,
  tile_log: u32,
  seed: u32,
  first_cell: u64,
  shared_real: DynamicThreadgroupBuffer<f32>,
  shared_imag: DynamicThreadgroupBuffer<f32>,
) {
  let i = thread_idx().x;
  let tile_base = block_idx().x << tile_log;
  let row = tile_base >> log_cols << log_cols;
  let within = tile_base - row;
  // SAFETY: each block owns one tile of one row; every thread generates and stores two distinct
  // cells and reaches the tile's barriers and shuffles.
  unsafe {
    let mut values: [Complex; 2] = uninit_array();
    let mut k = 0;
    while k < 2 {
      let cell = row + ((within + position(i, k, 0)).reverse_bits() >> (32 - log_cols));
      let pair = crate::random::normal_pair(first_cell.wrapping_add(cell as u64), seed);
      let eig = lam.load_unchecked(cell & ((1 << log_cells) - 1));
      values[k as usize] = Complex {
        re: pair.0 * eig,
        im: pair.1 * eig,
      };
      k += 1;
    }
    stages(
      values,
      twiddles,
      log_cols,
      tile_log,
      shared_real,
      shared_imag,
    );
    let half = 1 << (tile_log - 1);
    let mut k = 0;
    while k < 2 {
      let p = i + k * half;
      real.store_unchecked(tile_base + p, shared_real.load_unchecked(slot(p)));
      imag.store_unchecked(tile_base + p, shared_imag.load_unchecked(slot(p)));
      k += 1;
    }
  }
}

/// # Safety
/// Sheets hold 2^log_rows rows of 2^log_cols cells with log_rows >= 1; total counts the cells
/// of real/imag and the transposed outputs, and all four buffers are disjoint.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn sheet_transpose(
  real: ReadBuffer<f32>,
  imag: ReadBuffer<f32>,
  real_t: WriteBuffer<f32>,
  imag_t: WriteBuffer<f32>,
  log_rows: u32,
  log_cols: u32,
  total: u32,
) {
  let tid = crate::index();
  if tid < total {
    let log_cells = log_rows + log_cols;
    let sheet = tid >> log_cells << log_cells;
    let row = (tid >> log_cols) & ((1 << log_rows) - 1);
    let col = tid & ((1 << log_cols) - 1);
    let column = sheet + (col << log_rows);
    // SAFETY: transposition and bit reversal map each source to a unique destination.
    unsafe {
      let dst = column + (row.reverse_bits() >> (32 - log_rows));
      real_t.store_unchecked(dst, real.load_unchecked(tid));
      imag_t.store_unchecked(dst, imag.load_unchecked(tid));
    }
  }
}

/// # Safety
/// real covers cells per sheet in transposed order; output covers total values.
/// m and n select a leading submatrix of each sheet; buffers are disjoint.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn sheet_extract(
  real: ReadBuffer<f32>,
  output: WriteBuffer<f32>,
  cells: u32,
  rows: u32,
  m: u32,
  n: u32,
  r: f32,
  correction: f32,
  seed: u32,
  correction_cell: u64,
  total: u32,
) {
  let tid = crate::index();
  let per = m * n;
  if tid < total && per > 0 && n > 0 {
    let sheet = tid / per;
    let local = tid % per;
    let i = local / n;
    let j = local % n;
    let base = sheet * cells;
    let pair = crate::random::normal_pair(correction_cell.wrapping_add(sheet as u64), seed);
    let ty = r * (i + 1) as f32 / m as f32;
    let tx = r * (j + 1) as f32 / n as f32;
    // SAFETY: each lane reads within its sheet and writes its own output element.
    unsafe {
      let value = real.load_unchecked(base + j * rows + i) - real.load_unchecked(base);
      output.store_unchecked(tid, value + correction * (ty * pair.0 + tx * pair.1));
    }
  }
}
