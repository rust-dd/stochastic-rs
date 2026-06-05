//! # Accelerate / vDSP
//!
//! macOS-optimized Fgn sampling using Apple's Accelerate framework (vDSP FFT).
//! Split-complex format, in-place FFT, zero external dependencies.
//!
//! The vDSP FFT *setup* (twiddle tables) and the split-complex scratch buffers
//! are cached per thread and reused across calls. Creating a setup or
//! reallocating the buffers on every `sample()` (as a naive port would) costs
//! far more than the transform itself — the setup is an `O(N)` twiddle
//! precompute that Apple documents as create-once / reuse. The setup is
//! read-only during `vDSP_fft_zip`, so a per-thread cache keyed by `log2n` is
//! safe and needs no `Send`/`Sync` wrapper.
//!
//! This sampler is single-path optimal; batches are parallelised across cores
//! at the device level (see [`crate::device`]), one task per path, each reusing
//! its worker's cached setup and scratch.
use std::any::TypeId;
use std::cell::RefCell;
use std::ffi::c_void;

use anyhow::Result;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;

use super::Fgn;
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

/// A vDSP FFT setup owned by the thread that created it; freed on thread exit.
struct FftSetup {
  log2n: u64,
  ptr: *mut c_void,
}

impl Drop for FftSetup {
  fn drop(&mut self) {
    unsafe { vDSP_destroy_fftsetup(self.ptr) };
  }
}

/// Reusable split-complex scratch (`real`/`imag`), output staging, and an f32
/// eigenvalue buffer for the non-`f32` precision path.
struct Scratch {
  real: Vec<f32>,
  imag: Vec<f32>,
  out: Vec<f32>,
  eig: Vec<f32>,
}

thread_local! {
  static SETUPS: RefCell<Vec<FftSetup>> = RefCell::new(Vec::new());
  static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch {
    real: Vec::new(),
    imag: Vec::new(),
    out: Vec::new(),
    eig: Vec::new(),
  });
}

/// Returns a vDSP setup for `log2n`, creating it once per thread and reusing it
/// thereafter. The returned pointer stays valid for the thread's lifetime (the
/// owning [`FftSetup`] lives in the thread-local cache).
fn cached_setup(log2n: u64) -> Result<*mut c_void> {
  SETUPS.with(|cell| {
    let mut setups = cell.borrow_mut();
    if let Some(s) = setups.iter().find(|s| s.log2n == log2n) {
      return Ok(s.ptr);
    }
    let ptr = unsafe { vDSP_create_fftsetup(log2n, FFT_RADIX2) };
    if ptr.is_null() {
      anyhow::bail!("vDSP_create_fftsetup failed for log2n={log2n}");
    }
    setups.push(FftSetup { log2n, ptr });
    Ok(ptr)
  })
}

fn sample_f32<T: FloatExt, S: SeedExt>(
  eig_t: &[T],
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
  seed: &S,
) -> Result<Array2<T>> {
  let traj_size = 2 * n;
  let out_size = n - offset;
  let scale = (out_size.max(1) as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);
  let total = m * traj_size;
  let log2n = traj_size.trailing_zeros() as u64;
  let setup = cached_setup(log2n)?;
  let is_f32 = TypeId::of::<T>() == TypeId::of::<f32>();

  SCRATCH.with(|cell| {
    let Scratch {
      real,
      imag,
      out,
      eig,
    } = &mut *cell.borrow_mut();

    // f32 eigenvalues: zero-copy when `T == f32`, else convert once into the
    // cached `eig` buffer (no per-call allocation either way).
    let eig_f32: &[f32] = if is_f32 {
      // SAFETY: `T == f32`, checked above, and the eigenvalue slice is contiguous.
      unsafe { std::slice::from_raw_parts(eig_t.as_ptr() as *const f32, eig_t.len()) }
    } else {
      eig.clear();
      eig.extend(eig_t.iter().map(|x| x.to_f32().unwrap()));
      eig.as_slice()
    };

    real.resize(total, 0.0);
    imag.resize(total, 0.0);
    let normal = stochastic_rs_distributions::normal::SimdNormal::<f32>::new(0.0, 1.0, seed);
    normal.fill_slice_fast(real.as_mut_slice());
    normal.fill_slice_fast(imag.as_mut_slice());

    // Scale by the tiled eigenvalues (no per-element modulo).
    for b in 0..m {
      let base = b * traj_size;
      for j in 0..traj_size {
        let e = eig_f32[j];
        real[base + j] *= e;
        imag[base + j] *= e;
      }
    }

    // In-place forward FFT per trajectory, sharing the cached setup.
    for b in 0..m {
      let base = b * traj_size;
      let mut sc = DSPSplitComplex {
        realp: real[base..].as_mut_ptr(),
        imagp: imag[base..].as_mut_ptr(),
      };
      unsafe { vDSP_fft_zip(setup, &mut sc, 1, log2n, FFT_FORWARD) };
    }

    // Extract the real parts `[1..=out_size]` with scaling.
    out.resize(m * out_size, 0.0);
    for b in 0..m {
      let base = b * traj_size;
      let obase = b * out_size;
      for j in 0..out_size {
        out[obase + j] = real[base + j + 1] * scale;
      }
    }

    Ok(arr2_f32::<T>(out.as_slice(), m, out_size))
  })
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

impl<T: FloatExt, S: SeedExt, B> Fgn<T, S, B> {
  pub(crate) fn sample_accelerate_impl(&self, m: usize) -> Result<Array2<T>> {
    let n = self.n;
    let offset = self.offset;
    let hurst = self.hurst.to_f64().unwrap();
    let t = self.t.unwrap_or(T::one()).to_f64().unwrap();
    let eig_t = self
      .sqrt_eigenvalues
      .as_slice()
      .expect("eigenvalues are contiguous");
    sample_f32::<T, S>(eig_t, n, m, offset, hurst, t, &self.seed)
  }
}
