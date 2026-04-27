//! # Accelerate / vDSP
//!
//! macOS-optimized Fgn sampling using Apple's Accelerate framework (vDSP FFT).
//! Uses the AMX coprocessor and NEON SIMD on Apple Silicon.
//! Split-complex format, in-place FFT, zero external dependencies.
//!
use std::any::TypeId;
use std::ffi::c_void;

use anyhow::Result;
use either::Either;
use ndarray::Array1;
use ndarray::Array2;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use super::Fgn;
use stochastic_rs_core::simd_rng::SeedExt;
use crate::traits::FloatExt;

#[repr(C)]
struct DSPSplitComplex {
  realp: *mut f32,
  imagp: *mut f32,
}

const FFT_FORWARD: i32 = 1;
const FFT_RADIX2: i32 = 0;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
  fn vDSP_create_fftsetup(log2n: u64, radix: i32) -> *mut c_void;
  fn vDSP_destroy_fftsetup(setup: *mut c_void);
  fn vDSP_fft_zip(
    setup: *mut c_void,
    c: *mut DSPSplitComplex,
    stride: i64,
    log2n: u64,
    direction: i32,
  );
}

fn sample_f32<T: FloatExt>(
  sqrt_eigs: &[f32],
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
) -> Result<Either<Array1<T>, Array2<T>>> {
  let traj_size = 2 * n;
  let out_size = n - offset;
  let scale = (out_size.max(1) as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);
  let total = m * traj_size;
  let log2n = traj_size.trailing_zeros() as u64;

  // Generate normals (split format)
  let mut real = vec![0.0f32; total];
  let mut imag = vec![0.0f32; total];
  {
    let normal = stochastic_rs_distributions::normal::SimdNormal::<f32>::new(0.0, 1.0);
    normal.fill_slice_fast(&mut real);
    normal.fill_slice_fast(&mut imag);
  }

  // Scale by eigenvalues
  for i in 0..total {
    let e = sqrt_eigs[i % traj_size];
    real[i] *= e;
    imag[i] *= e;
  }

  // vDSP FFT setup (reusable across batches)
  let setup = unsafe { vDSP_create_fftsetup(log2n, FFT_RADIX2) };
  if setup.is_null() {
    anyhow::bail!("vDSP_create_fftsetup failed for log2n={log2n}");
  }

  // In-place FFT per batch — sequential loop.
  // vDSP itself uses SIMD/AMX internally; rayon parallelism happens at the
  // caller level via sample_par() if needed.
  for b in 0..m {
    let base = b * traj_size;
    let mut sc = DSPSplitComplex {
      realp: real[base..].as_mut_ptr(),
      imagp: imag[base..].as_mut_ptr(),
    };
    unsafe { vDSP_fft_zip(setup, &mut sc, 1, log2n, FFT_FORWARD) };
  }

  unsafe { vDSP_destroy_fftsetup(setup) };

  // Extract real parts [1..out_size+1] with scaling
  let mut output = vec![0.0f32; m * out_size];
  for b in 0..m {
    let base = b * traj_size;
    for j in 0..out_size {
      output[b * out_size + j] = real[base + j + 1] * scale;
    }
  }

  let fgn = arr2_f32::<T>(&output, m, out_size);
  if m == 1 {
    return Ok(Either::Left(fgn.row(0).to_owned()));
  }
  Ok(Either::Right(fgn))
}

fn arr2_f32<T: FloatExt>(data: &[f32], m: usize, cols: usize) -> Array2<T> {
  if TypeId::of::<T>() == TypeId::of::<f32>() {
    let out = Array2::<f32>::from_shape_vec((m, cols), data.to_vec()).expect("shape");
    unsafe { std::mem::transmute::<Array2<f32>, Array2<T>>(out) }
  } else {
    let mut out = Array2::<T>::zeros((m, cols));
    for i in 0..m {
      for j in 0..cols {
        out[[i, j]] = T::from_f64_fast(data[i * cols + j] as f64);
      }
    }
    out
  }
}

impl<T: FloatExt, S: SeedExt> Fgn<T, S> {
  pub(crate) fn sample_accelerate_impl(&self, m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    let n = self.n;
    let offset = self.offset;
    let hurst = self.hurst.to_f64().unwrap();
    let t = self.t.unwrap_or(T::one()).to_f64().unwrap();
    let eigs: Vec<f32> = self
      .sqrt_eigenvalues
      .iter()
      .map(|x| x.to_f32().unwrap())
      .collect();
    sample_f32::<T>(&eigs, n, m, offset, hurst, t)
  }
}
