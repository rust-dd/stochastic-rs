//! Two-dimensional intrinsic embedding and Stein's linear correction.
//! Kroese & Botev (2015), Spatial process simulation, §4.4, arXiv:1308.0399.

use metal_oxide_device::{
  DynamicThreadgroupBuffer, ReadBuffer, ReadWriteBuffer, WriteBuffer, block_dim, block_idx, kernel,
  math, thread_idx, threadgroup::barrier,
};

/// # Safety
/// Work buffers cover total cells; lam covers cells and reverse covers cols.
/// Rows are complete power-of-two transforms; all buffers are disjoint.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn sheet_generate(
  real: WriteBuffer<f32>,
  imag: WriteBuffer<f32>,
  lam: ReadBuffer<f32>,
  reverse: ReadBuffer<u32>,
  cells: u32,
  cols: u32,
  seed: u32,
  first_cell: u64,
  total: u32,
) {
  let tid = crate::index();
  if tid < total && cells > 0 && cols > 0 {
    let sheet = tid / cells;
    let local = tid % cells;
    let row = local / cols;
    let col = local % cols;
    let pair = crate::random::normal_pair(first_cell.wrapping_add(tid as u64), seed);
    // SAFETY: every lane owns a unique bit-reversed cell within its row.
    unsafe {
      let eig = lam.load_unchecked(local);
      let dst = sheet * cells + row * cols + reverse.load_unchecked(col);
      real.store_unchecked(dst, pair.0 * eig);
      imag.store_unchecked(dst, pair.1 * eig);
    }
  }
}

/// # Safety
/// All work buffers cover total cells in complete rows * cols sheets.
/// reverse covers rows with its bit-reversal permutation; buffers are disjoint.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn sheet_transpose(
  real: ReadBuffer<f32>,
  imag: ReadBuffer<f32>,
  real_t: WriteBuffer<f32>,
  imag_t: WriteBuffer<f32>,
  reverse: ReadBuffer<u32>,
  cells: u32,
  rows: u32,
  cols: u32,
  total: u32,
) {
  let tid = crate::index();
  if tid < total && cells > 0 && cols > 0 {
    let sheet = tid / cells;
    let local = tid % cells;
    let row = local / cols;
    let col = local % cols;
    // SAFETY: transposition and bit reversal map each source to a unique destination.
    unsafe {
      let dst = sheet * cells + col * rows + reverse.load_unchecked(row);
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

/// # Safety
/// Both buffers cover 2 * butterflies values in complete length-n transforms.
/// half_stride selects a valid radix-2 stage; the buffers are disjoint.
#[kernel]
pub unsafe fn sheet_fft_butterfly(
  real: ReadWriteBuffer<f32>,
  imag: ReadWriteBuffer<f32>,
  n: u32,
  half_stride: u32,
  butterflies: u32,
) {
  let tid = crate::index();
  if tid < butterflies && n >= 2 && half_stride > 0 {
    let per = n / 2;
    let batch = tid / per;
    let local = tid % per;
    let stride = half_stride * 2;
    let group = local / half_stride;
    let pos = local % half_stride;
    let i = batch * n + group * stride + pos;
    let j = i + half_stride;
    let angle = -2.0 * core::f32::consts::PI * pos as f32 / stride as f32;
    let wr = math::cos(angle);
    let wi = math::sin(angle);
    // SAFETY: each lane exclusively owns one butterfly in this transform stage.
    unsafe {
      let tr = real.load_unchecked(j) * wr - imag.load_unchecked(j) * wi;
      let ti = real.load_unchecked(j) * wi + imag.load_unchecked(j) * wr;
      let ar = real.load_unchecked(i);
      let ai = imag.load_unchecked(i);
      real.store_unchecked(i, ar + tr);
      imag.store_unchecked(i, ai + ti);
      real.store_unchecked(j, ar - tr);
      imag.store_unchecked(j, ai - ti);
    }
  }
}

/// # Safety
/// Each block owns a complete tile of 2 * block_dim.x values in disjoint buffers.
/// stages fits that tile; each scratch allocation covers the tile.
#[kernel]
pub unsafe fn sheet_fft_tile(
  real: ReadWriteBuffer<f32>,
  imag: ReadWriteBuffer<f32>,
  stages: u32,
  shared_real: DynamicThreadgroupBuffer<f32>,
  shared_imag: DynamicThreadgroupBuffer<f32>,
) {
  let lane = thread_idx().x;
  let half_tile = block_dim().x;
  let tile = half_tile * 2;
  let base = block_idx().x * tile;
  // SAFETY: complete tiles are disjoint; every lane reaches every stage barrier.
  unsafe {
    let mut k = lane;
    while k < tile {
      shared_real.store_unchecked(k, real.load_unchecked(base + k));
      shared_imag.store_unchecked(k, imag.load_unchecked(base + k));
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
      real.store_unchecked(base + k, shared_real.load_unchecked(k));
      imag.store_unchecked(base + k, shared_imag.load_unchecked(k));
      k += half_tile;
    }
  }
}
