//! Device-side fGN kernels for the experimental cuda-oxide backend.
//!
//! These are the same kernels the host launches via `cuda_launch!` in
//! `stochastic-rs-stochastic`'s `noise::fgn::cuda_oxide`. They live in their
//! own device-only crate so the cuda-oxide rustc codegen backend can lower
//! them to PTX without trying to compile the host-only code in the umbrella
//! lib. See this crate's `Cargo.toml` for the two-mode build contract.

#![allow(clippy::missing_safety_doc)]

use cuda_device::{DisjointSlice, kernel, thread};

const TAU_F32: f32 = 6.283_185_5;
const TAU_F64: f64 = 6.283_185_307_179_586;

#[inline]
fn reverse_bits_n(mut x: usize, bits: usize) -> usize {
  let mut y = 0usize;
  let mut i = 0usize;
  while i < bits {
    y = (y << 1) | (x & 1);
    x >>= 1;
    i += 1;
  }
  y
}

#[inline]
fn philox2x32_10(tid: usize, seed: u64, seq: u64) -> (u32, u32) {
  let ctr = tid as u64 + seq;
  let mut lo = ctr as u32;
  let mut hi = (ctr >> 32) as u32;
  let mut k = seed as u32;
  let mut i = 0;
  while i < 10 {
    let p = 0xD251_1F53_u64 * lo as u64;
    lo = ((p >> 32) as u32) ^ hi ^ k;
    hi = p as u32;
    k = k.wrapping_add(0x9E37_79B9);
    i += 1;
  }
  (lo, hi)
}

#[kernel]
pub fn gen_scale_f32(
  mut data: DisjointSlice<f32>,
  sqrt_eigs: &[f32],
  traj_size: usize,
  total_complex: usize,
  seed: u64,
  seq: u64,
) {
  let tid = thread::index_1d().get();
  if tid < total_complex {
    let (lo, hi) = philox2x32_10(tid, seed, seq);
    let u1 = (lo as f32 + 0.5) * 2.328_306_4e-10;
    let u2 = (hi as f32 + 0.5) * 2.328_306_4e-10;
    let r = (-2.0 * u1.ln()).sqrt();
    let angle = TAU_F32 * u2;
    let eig = sqrt_eigs[tid % traj_size];
    let base = 2 * tid;
    unsafe {
      let ptr = data.as_mut_ptr();
      *ptr.add(base) = r * angle.cos() * eig;
      *ptr.add(base + 1) = r * angle.sin() * eig;
    }
  }
}

#[kernel]
pub fn bit_reverse_f32(mut data: DisjointSlice<f32>, traj_size: usize, log_n: usize) {
  let tid = thread::index_1d().get();
  let batch = tid / traj_size;
  let local = tid % traj_size;
  let rev = reverse_bits_n(local, log_n);
  if local < rev {
    let i = 2 * (batch * traj_size + local);
    let j = 2 * (batch * traj_size + rev);
    unsafe {
      let ptr = data.as_mut_ptr();
      let ar = *ptr.add(i);
      let ai = *ptr.add(i + 1);
      *ptr.add(i) = *ptr.add(j);
      *ptr.add(i + 1) = *ptr.add(j + 1);
      *ptr.add(j) = ar;
      *ptr.add(j + 1) = ai;
    }
  }
}

#[kernel]
pub fn fft_stage_f32(mut data: DisjointSlice<f32>, traj_size: usize, half_stride: usize) {
  let tid = thread::index_1d().get();
  let butterflies_per_batch = traj_size / 2;
  let batch = tid / butterflies_per_batch;
  let local = tid % butterflies_per_batch;
  let stride = half_stride * 2;
  let group = local / half_stride;
  let pos = local % half_stride;
  let i_complex = batch * traj_size + group * stride + pos;
  let j_complex = i_complex + half_stride;
  let angle = -TAU_F32 * (pos as f32) / (stride as f32);
  let wr = angle.cos();
  let wi = angle.sin();

  unsafe {
    let ptr = data.as_mut_ptr();
    let i = 2 * i_complex;
    let j = 2 * j_complex;
    let ar = *ptr.add(i);
    let ai = *ptr.add(i + 1);
    let br = *ptr.add(j);
    let bi = *ptr.add(j + 1);
    let tr = br * wr - bi * wi;
    let ti = br * wi + bi * wr;
    *ptr.add(i) = ar + tr;
    *ptr.add(i + 1) = ai + ti;
    *ptr.add(j) = ar - tr;
    *ptr.add(j + 1) = ai - ti;
  }
}

#[kernel]
pub fn extract_real_f32(
  data: &[f32],
  mut output: DisjointSlice<f32>,
  out_size: usize,
  traj_size: usize,
  scale: f32,
) {
  let tid = thread::index_1d();
  if let Some(out) = output.get_mut(tid) {
    let flat = tid.get();
    let traj_id = flat / out_size;
    let idx = flat % out_size;
    *out = data[2 * (traj_id * traj_size + idx + 1)] * scale;
  }
}

#[kernel]
pub fn gen_scale_f64(
  mut data: DisjointSlice<f64>,
  sqrt_eigs: &[f64],
  traj_size: usize,
  total_complex: usize,
  seed: u64,
  seq: u64,
) {
  let tid = thread::index_1d().get();
  if tid < total_complex {
    let (lo, hi) = philox2x32_10(tid, seed, seq);
    let u1 = (lo as f64 + 0.5) * 2.328_306_436_538_696_3e-10;
    let u2 = (hi as f64 + 0.5) * 2.328_306_436_538_696_3e-10;
    let r = (-2.0 * u1.ln()).sqrt();
    let angle = TAU_F64 * u2;
    let eig = sqrt_eigs[tid % traj_size];
    let base = 2 * tid;
    unsafe {
      let ptr = data.as_mut_ptr();
      *ptr.add(base) = r * angle.cos() * eig;
      *ptr.add(base + 1) = r * angle.sin() * eig;
    }
  }
}

#[kernel]
pub fn bit_reverse_f64(mut data: DisjointSlice<f64>, traj_size: usize, log_n: usize) {
  let tid = thread::index_1d().get();
  let batch = tid / traj_size;
  let local = tid % traj_size;
  let rev = reverse_bits_n(local, log_n);
  if local < rev {
    let i = 2 * (batch * traj_size + local);
    let j = 2 * (batch * traj_size + rev);
    unsafe {
      let ptr = data.as_mut_ptr();
      let ar = *ptr.add(i);
      let ai = *ptr.add(i + 1);
      *ptr.add(i) = *ptr.add(j);
      *ptr.add(i + 1) = *ptr.add(j + 1);
      *ptr.add(j) = ar;
      *ptr.add(j + 1) = ai;
    }
  }
}

#[kernel]
pub fn fft_stage_f64(mut data: DisjointSlice<f64>, traj_size: usize, half_stride: usize) {
  let tid = thread::index_1d().get();
  let butterflies_per_batch = traj_size / 2;
  let batch = tid / butterflies_per_batch;
  let local = tid % butterflies_per_batch;
  let stride = half_stride * 2;
  let group = local / half_stride;
  let pos = local % half_stride;
  let i_complex = batch * traj_size + group * stride + pos;
  let j_complex = i_complex + half_stride;
  let angle = -TAU_F64 * (pos as f64) / (stride as f64);
  let wr = angle.cos();
  let wi = angle.sin();

  unsafe {
    let ptr = data.as_mut_ptr();
    let i = 2 * i_complex;
    let j = 2 * j_complex;
    let ar = *ptr.add(i);
    let ai = *ptr.add(i + 1);
    let br = *ptr.add(j);
    let bi = *ptr.add(j + 1);
    let tr = br * wr - bi * wi;
    let ti = br * wi + bi * wr;
    *ptr.add(i) = ar + tr;
    *ptr.add(i + 1) = ai + ti;
    *ptr.add(j) = ar - tr;
    *ptr.add(j + 1) = ai - ti;
  }
}

#[kernel]
pub fn extract_real_f64(
  data: &[f64],
  mut output: DisjointSlice<f64>,
  out_size: usize,
  traj_size: usize,
  scale: f64,
) {
  let tid = thread::index_1d();
  if let Some(out) = output.get_mut(tid) {
    let flat = tid.get();
    let traj_id = flat / out_size;
    let idx = flat % out_size;
    *out = data[2 * (traj_id * traj_size + idx + 1)] * scale;
  }
}
