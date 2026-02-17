//! # CUDA
//!
//! $$
//! \varepsilon \sim \mathcal N(0,\Sigma)\ \text{with optional fractional covariance shaping}
//! $$
//!
use std::any::TypeId;
use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::Result;
use either::Either;
use libloading::Library;
use libloading::Symbol;
use ndarray::Array1;
use ndarray::Array2;
use rand::Rng;

use super::FGN;
use crate::traits::FloatExt;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
struct CuComplexF32 {
  x: f32,
  y: f32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
struct CuComplexF64 {
  x: f64,
  y: f64,
}

type Fgn32InitFn = unsafe extern "C" fn(*const CuComplexF32, i32, i32, i32, i32) -> i32;
type Fgn32SampleFn = unsafe extern "C" fn(*mut f32, f32, u64);
type Fgn32CleanupFn = unsafe extern "C" fn();

type Fgn64InitFn = unsafe extern "C" fn(*const CuComplexF64, i32, i32, i32, i32) -> i32;
type Fgn64SampleFn = unsafe extern "C" fn(*mut f64, f64, u64);
type Fgn64CleanupFn = unsafe extern "C" fn();

struct CudaContextF32 {
  _lib: Library,
  fgn_sample: Symbol<'static, Fgn32SampleFn>,
  fgn_cleanup: Symbol<'static, Fgn32CleanupFn>,
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
}

impl Drop for CudaContextF32 {
  fn drop(&mut self) {
    unsafe {
      (self.fgn_cleanup)();
    }
  }
}

struct CudaContextF64 {
  _lib: Library,
  fgn_sample: Symbol<'static, Fgn64SampleFn>,
  fgn_cleanup: Symbol<'static, Fgn64CleanupFn>,
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
}

impl Drop for CudaContextF64 {
  fn drop(&mut self) {
    unsafe {
      (self.fgn_cleanup)();
    }
  }
}

static CUDA_CONTEXT_F32: Mutex<Option<CudaContextF32>> = Mutex::new(None);
static CUDA_CONTEXT_F64: Mutex<Option<CudaContextF64>> = Mutex::new(None);

unsafe fn load_symbol_with_fallback<'lib, T>(
  lib: &'lib Library,
  names: &[&[u8]],
) -> Result<Symbol<'lib, T>> {
  let mut last_error: Option<libloading::Error> = None;
  for &name in names {
    match lib.get::<T>(name) {
      Ok(sym) => return Ok(sym),
      Err(err) => last_error = Some(err),
    }
  }

  let tried = names
    .iter()
    .map(|name| String::from_utf8_lossy(name).to_string())
    .collect::<Vec<_>>()
    .join(", ");
  if let Some(err) = last_error {
    anyhow::bail!("could not load symbols [{tried}]: {err}");
  }
  anyhow::bail!("could not load symbols [{tried}]");
}

fn cuda_library_candidates() -> Vec<PathBuf> {
  let mut candidates = Vec::new();
  if let Ok(path) = std::env::var("STOCHASTIC_RS_CUDA_FGN_LIB_PATH") {
    if !path.is_empty() {
      candidates.push(PathBuf::from(path));
    }
  }
  if let Some(path) = option_env!("STOCHASTIC_RS_CUDA_FGN_LIB") {
    if !path.is_empty() {
      candidates.push(PathBuf::from(path));
    }
  }
  if cfg!(target_os = "windows") {
    candidates.push(PathBuf::from("src/stochastic/cuda/fgn_windows/fgn.dll"));
  } else if cfg!(target_os = "linux") {
    candidates.push(PathBuf::from("src/stochastic/cuda/fgn_linux/libfgn.so"));
  }
  candidates
}

fn open_cuda_library() -> Result<Library> {
  let candidates = cuda_library_candidates();
  let mut last_error: Option<String> = None;
  for path in &candidates {
    match unsafe { Library::new(path) } {
      Ok(lib) => return Ok(lib),
      Err(err) => {
        last_error = Some(format!("{}: {err}", path.display()));
      }
    }
  }

  if let Some(err) = last_error {
    anyhow::bail!(
      "failed to load CUDA FGN library (tried: {}), last error: {err}",
      candidates
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
    );
  }

  anyhow::bail!("no CUDA FGN library candidates configured");
}

fn array2_from_flat<T: FloatExt, U: Copy + Into<f64>>(
  host_output: &[U],
  m: usize,
  out_size: usize,
) -> Array2<T> {
  let mut out = Array2::<T>::zeros((m, out_size));
  for i in 0..m {
    for j in 0..out_size {
      out[[i, j]] = T::from_f64_fast(host_output[i * out_size + j].into());
    }
  }
  out
}

impl<T: FloatExt> FGN<T> {
  fn sample_cuda_f32(&self, m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    let n = self.n;
    let offset = self.offset;
    let hurst = self.hurst.to_f64().unwrap();
    let t = self.t.unwrap_or(T::one()).to_f64().unwrap();
    let scale = (n as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);
    let out_size = n - offset;

    let need_init = {
      let guard = CUDA_CONTEXT_F32.lock().unwrap();
      match &*guard {
        Some(ctx) => {
          ctx.n != n || ctx.m != m || ctx.offset != offset || ctx.hurst != hurst || ctx.t != t
        }
        None => true,
      }
    };

    if need_init {
      {
        let mut guard = CUDA_CONTEXT_F32.lock().unwrap();
        *guard = None;
      }

      let lib = open_cuda_library()?;
      let fgn_init: Symbol<Fgn32InitFn> =
        unsafe { load_symbol_with_fallback(&lib, &[b"fgn32_init", b"fgn_init"])? };
      let fgn_sample: Symbol<Fgn32SampleFn> =
        unsafe { load_symbol_with_fallback(&lib, &[b"fgn32_sample", b"fgn_sample"])? };
      let fgn_cleanup: Symbol<Fgn32CleanupFn> =
        unsafe { load_symbol_with_fallback(&lib, &[b"fgn32_cleanup", b"fgn_cleanup"])? };

      let host_sqrt_eigs: Vec<CuComplexF32> = self
        .sqrt_eigenvalues
        .iter()
        .map(|x| CuComplexF32 {
          x: x.to_f32().unwrap(),
          y: 0.0,
        })
        .collect();

      let init_status = unsafe {
        fgn_init(
          host_sqrt_eigs.as_ptr(),
          host_sqrt_eigs.len() as i32,
          n as i32,
          m as i32,
          offset as i32,
        )
      };
      if init_status != 0 {
        anyhow::bail!("fgn32_init failed with status {init_status}");
      }

      let fgn_sample: Symbol<'static, Fgn32SampleFn> = unsafe { std::mem::transmute(fgn_sample) };
      let fgn_cleanup: Symbol<'static, Fgn32CleanupFn> =
        unsafe { std::mem::transmute(fgn_cleanup) };

      let ctx = CudaContextF32 {
        _lib: lib,
        fgn_sample,
        fgn_cleanup,
        n,
        m,
        offset,
        hurst,
        t,
      };

      let mut guard = CUDA_CONTEXT_F32.lock().unwrap();
      *guard = Some(ctx);
    }

    let mut host_output = vec![0.0f32; m * out_size];
    let seed: u64 = rand::rng().random();
    {
      let guard = CUDA_CONTEXT_F32.lock().unwrap();
      let ctx = guard.as_ref().unwrap();
      unsafe {
        (ctx.fgn_sample)(host_output.as_mut_ptr(), scale, seed);
      }
    }

    let fgn = array2_from_flat::<T, f32>(&host_output, m, out_size);
    if m == 1 {
      return Ok(Either::Left(fgn.row(0).to_owned()));
    }

    Ok(Either::Right(fgn))
  }

  fn sample_cuda_f64(&self, m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    let n = self.n;
    let offset = self.offset;
    let hurst = self.hurst.to_f64().unwrap();
    let t = self.t.unwrap_or(T::one()).to_f64().unwrap();
    let scale = (n as f64).powf(-hurst) * t.powf(hurst);
    let out_size = n - offset;

    let need_init = {
      let guard = CUDA_CONTEXT_F64.lock().unwrap();
      match &*guard {
        Some(ctx) => {
          ctx.n != n || ctx.m != m || ctx.offset != offset || ctx.hurst != hurst || ctx.t != t
        }
        None => true,
      }
    };

    if need_init {
      {
        let mut guard = CUDA_CONTEXT_F64.lock().unwrap();
        *guard = None;
      }

      let lib = open_cuda_library()?;
      let fgn_init: Symbol<Fgn64InitFn> =
        unsafe { load_symbol_with_fallback(&lib, &[b"fgn64_init", b"fgn_init_f64"])? };
      let fgn_sample: Symbol<Fgn64SampleFn> =
        unsafe { load_symbol_with_fallback(&lib, &[b"fgn64_sample", b"fgn_sample_f64"])? };
      let fgn_cleanup: Symbol<Fgn64CleanupFn> =
        unsafe { load_symbol_with_fallback(&lib, &[b"fgn64_cleanup", b"fgn_cleanup_f64"])? };

      let host_sqrt_eigs: Vec<CuComplexF64> = self
        .sqrt_eigenvalues
        .iter()
        .map(|x| CuComplexF64 {
          x: x.to_f64().unwrap(),
          y: 0.0,
        })
        .collect();

      let init_status = unsafe {
        fgn_init(
          host_sqrt_eigs.as_ptr(),
          host_sqrt_eigs.len() as i32,
          n as i32,
          m as i32,
          offset as i32,
        )
      };
      if init_status != 0 {
        anyhow::bail!("fgn64_init failed with status {init_status}");
      }

      let fgn_sample: Symbol<'static, Fgn64SampleFn> = unsafe { std::mem::transmute(fgn_sample) };
      let fgn_cleanup: Symbol<'static, Fgn64CleanupFn> =
        unsafe { std::mem::transmute(fgn_cleanup) };

      let ctx = CudaContextF64 {
        _lib: lib,
        fgn_sample,
        fgn_cleanup,
        n,
        m,
        offset,
        hurst,
        t,
      };

      let mut guard = CUDA_CONTEXT_F64.lock().unwrap();
      *guard = Some(ctx);
    }

    let mut host_output = vec![0.0f64; m * out_size];
    let seed: u64 = rand::rng().random();
    {
      let guard = CUDA_CONTEXT_F64.lock().unwrap();
      let ctx = guard.as_ref().unwrap();
      unsafe {
        (ctx.fgn_sample)(host_output.as_mut_ptr(), scale, seed);
      }
    }

    let fgn = array2_from_flat::<T, f64>(&host_output, m, out_size);
    if m == 1 {
      return Ok(Either::Left(fgn.row(0).to_owned()));
    }

    Ok(Either::Right(fgn))
  }

  pub(crate) fn sample_cuda_impl(&self, m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
      return self.sample_cuda_f32(m);
    }
    match self.sample_cuda_f64(m) {
      Ok(out) => Ok(out),
      Err(err) => {
        let msg = err.to_string();
        if msg.contains("fgn64_") || msg.contains("fgn_init_f64") || msg.contains("fgn_sample_f64")
        {
          self.sample_cuda_f32(m)
        } else {
          Err(err)
        }
      }
    }
  }
}
