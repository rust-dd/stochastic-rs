use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::sync::Mutex;

#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use either::Either;
#[cfg(feature = "cuda")]
use libloading::Library;
#[cfg(feature = "cuda")]
use libloading::Symbol;
use ndarray::concatenate;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndrustfft::ndfft_par;
use ndrustfft::FftHandler;
use num_complex::Complex;
use phastft::fft_64;
use phastft::planner::Direction;
#[cfg(feature = "cuda")]
use rand::Rng;
use rand_distr::Distribution;

use crate::distributions::complex::ComplexDistribution;
use crate::distributions::normal::SimdNormal;
use crate::stochastic::Float;
use crate::stochastic::ProcessExt;

// CUDA type for complex numbers
#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
struct CuComplex {
  x: f32,
  y: f32,
}

// Persistent CUDA context for fast repeated sampling
#[cfg(feature = "cuda")]
struct CudaContext {
  _lib: Library, // Keep library alive
  fgn_sample: Symbol<'static, unsafe extern "C" fn(*mut f32, f32, u64)>,
  fgn_cleanup: Symbol<'static, unsafe extern "C" fn()>,
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
}

#[cfg(feature = "cuda")]
impl Drop for CudaContext {
  fn drop(&mut self) {
    unsafe {
      (self.fgn_cleanup)();
    }
  }
}

#[cfg(feature = "cuda")]
static CUDA_CONTEXT: Mutex<Option<CudaContext>> = Mutex::new(None);

pub struct FGN<T: Float> {
  pub hurst: T,
  pub n: usize,
  pub t: Option<T>,
  pub offset: usize,
  pub sqrt_eigenvalues: Arc<Array1<Complex<T>>>,
  pub fft_handler: Arc<FftHandler<T>>,
}

impl<T: Float> FGN<T> {
  pub fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n)
  }
}

impl<T: Float> FGN<T> {
  #[must_use]
  pub fn new(hurst: T, n: usize, t: Option<T>) -> Self {
    if !(T::zero()..=T::one()).contains(&hurst) {
      panic!("Hurst parameter must be between 0 and 1");
    }

    let offset = n.next_power_of_two() - n;
    let n = n.next_power_of_two();
    let mut r = Array1::linspace(T::zero(), T::from_usize_(n), n + 1);
    let f2 = T::from_usize_(2);
    r.mapv_inplace(|x| {
      if x == T::zero() {
        T::one()
      } else {
        T::from_f64_fast(0.5)
          * ((x + T::one()).powf(f2 * hurst) - f2 * x.powf(f2 * hurst)
            + (x - T::one()).powf(f2 * hurst))
      }
    });
    let r = concatenate(
      Axis(0),
      #[allow(clippy::reversed_empty_ranges)]
      &[r.view(), r.slice(s![..;-1]).slice(s![1..-1]).view()],
    )
    .unwrap();
    let data = r.mapv(|v| Complex::new(v, T::zero()));
    let r_fft = FftHandler::new(r.len());
    let mut sqrt_eigenvalues = Array1::<Complex<T>>::zeros(r.len());
    ndfft_par(&data, &mut sqrt_eigenvalues, &r_fft, 0);
    sqrt_eigenvalues.mapv_inplace(|x| Complex::new((x.re / T::from_usize_(2 * n)).sqrt(), x.im));

    Self {
      hurst,
      n,
      offset,
      t,
      sqrt_eigenvalues: Arc::new(sqrt_eigenvalues),
      fft_handler: Arc::new(FftHandler::new(2 * n)),
    }
  }

  /// Sample FGN using PhastFT (requires nightly Rust and `phastft` feature)
  /// PhastFT is a high-performance FFT library that uses SIMD and is competitive with FFTW
  pub fn sample_phastft(&self) -> Array1<T> {
    // Generate random complex numbers

    let rnd = self.sample_rnd(SimdNormal::<T, 64>::new(T::zero(), T::one()));

    // Multiply by sqrt eigenvalues
    let fgn_complex = &*self.sqrt_eigenvalues * &rnd;

    // PhastFT uses separate real and imaginary arrays
    let mut reals = fgn_complex
      .iter()
      .map(|c| c.re.to_f64().unwrap())
      .collect::<Vec<_>>();
    let mut imags = fgn_complex
      .iter()
      .map(|c| c.im.to_f64().unwrap())
      .collect::<Vec<_>>();

    // Perform FFT using PhastFT
    fft_64(&mut reals, &mut imags, Direction::Forward);

    // Extract real parts and scale
    let scale =
      T::from_usize_(self.n).powf(-self.hurst) * self.t.unwrap_or(T::one()).powf(self.hurst);
    let result = reals[1..self.n - self.offset + 1]
      .iter()
      .map(|&x| T::from_f64_fast(x) * scale)
      .collect();

    Array1::from_vec(result)
  }

  fn sample_rnd<D: Distribution<T>>(&self, d: D) -> Array1<Complex<T>> {
    Array1::<Complex<T>>::random(2 * self.n, ComplexDistribution::new(&d, &d))
  }
}

impl<T: Float> ProcessExt<T> for FGN<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let rnd = self.sample_rnd(SimdNormal::<T, 64>::new(T::zero(), T::one()));

    let fgn = &*self.sqrt_eigenvalues * &rnd;
    let mut fgn_fft = Array1::<Complex<T>>::zeros(2 * self.n);
    ndfft_par(&fgn, &mut fgn_fft, &*self.fft_handler, 0);
    let scale =
      T::from_usize_(self.n).powf(-self.hurst) * self.t.unwrap_or(T::one()).powf(self.hurst);
    let fgn = fgn_fft
      .slice(s![1..self.n - self.offset + 1])
      .mapv(|x| x.re * scale);
    fgn
  }

  #[cfg(feature = "cuda")]
  fn sample_cuda(&self, m: usize) -> Result<Either<Array1<f64>, Array2<f64>>> {
    // CUDA kernel compilation - see src/stochastic/cuda/CUDA_BUILD.md for details
    // Quick build:
    //   Linux:   cd src/stochastic/cuda && ./build.sh
    //   Windows: cd src\stochastic\cuda && build.bat

    type FgnInitFn = unsafe extern "C" fn(
      /* h_sqrt_eigs: */ *const CuComplex,
      /* eig_len:     */ i32,
      /* n:           */ i32,
      /* m:           */ i32,
      /* offset:      */ i32,
    ) -> i32;

    type FgnSampleFn = unsafe extern "C" fn(
      /* h_output:    */ *mut f32,
      /* scale:       */ f32,
      /* seed:        */ u64,
    );

    type FgnCleanupFn = unsafe extern "C" fn();

    let n = self.n;
    let offset = self.offset;
    let hurst = self.hurst.to_f64().unwrap();
    let t = self.t.unwrap_or(1.0.into()).to_f64().unwrap();
    // Scale factor: n^(-H) * t^H, same as CPU
    let scale = (n as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);
    let out_size = n - offset;

    // Check if we need to reinitialize
    let need_init = {
      let guard = CUDA_CONTEXT.lock().unwrap();
      match &*guard {
        Some(ctx) => {
          ctx.n != n || ctx.m != m || ctx.offset != offset || ctx.hurst != hurst || ctx.t != t
        }
        None => true,
      }
    };

    // Initialize if needed
    if need_init {
      // Drop old context first (this calls cleanup)
      {
        let mut guard = CUDA_CONTEXT.lock().unwrap();
        *guard = None;
      }

      #[cfg(target_os = "windows")]
      let lib = unsafe { Library::new("src/stochastic/cuda/fgn_windows/fgn.dll") }?;
      #[cfg(target_os = "linux")]
      let lib = unsafe { Library::new("src/stochastic/cuda/fgn_linux/libfgn.so") }?;

      // Get function pointers
      let fgn_init: Symbol<FgnInitFn> = unsafe { lib.get(b"fgn_init") }?;
      let fgn_sample: Symbol<FgnSampleFn> = unsafe { lib.get(b"fgn_sample") }?;
      let fgn_cleanup: Symbol<FgnCleanupFn> = unsafe { lib.get(b"fgn_cleanup") }?;

      // Prepare eigenvalues
      let host_sqrt_eigs: Vec<CuComplex> = self
        .sqrt_eigenvalues
        .iter()
        .map(|z| CuComplex {
          x: z.re.to_f32().unwrap(),
          y: z.im.to_f32().unwrap(),
        })
        .collect();

      // Initialize CUDA context (uploads eigenvalues, allocates buffers, creates FFT plan)
      unsafe {
        fgn_init(
          host_sqrt_eigs.as_ptr(),
          host_sqrt_eigs.len() as i32,
          n as i32,
          m as i32,
          offset as i32,
        );
      }

      // Store context (transmute to extend lifetime - safe because lib stays alive)
      let fgn_sample: Symbol<'static, unsafe extern "C" fn(*mut f32, f32, u64)> =
        unsafe { std::mem::transmute(fgn_sample) };
      let fgn_cleanup: Symbol<'static, unsafe extern "C" fn()> =
        unsafe { std::mem::transmute(fgn_cleanup) };

      let ctx = CudaContext {
        _lib: lib,
        fgn_sample,
        fgn_cleanup,
        n,
        m,
        offset,
        hurst,
        t,
      };

      let mut guard = CUDA_CONTEXT.lock().unwrap();
      *guard = Some(ctx);
    }

    // Fast path: just call fgn_sample (GPU already initialized)
    let mut host_output = vec![0.0f32; m * out_size];
    let mut rng = rand::rng();
    let seed: u64 = rng.random();

    {
      let guard = CUDA_CONTEXT.lock().unwrap();
      let ctx = guard.as_ref().unwrap();
      unsafe {
        (ctx.fgn_sample)(host_output.as_mut_ptr(), scale, seed);
      }
    }

    // Convert to ndarray
    let mut fgn = Array2::<f64>::zeros((m, out_size));
    for i in 0..m {
      for j in 0..out_size {
        fgn[[i, j]] = host_output[i * out_size + j] as f64;
      }
    }

    if m == 1 {
      return Ok(Either::Left(fgn.row(0).to_owned()));
    }

    Ok(Either::Right(fgn))
  }
}
