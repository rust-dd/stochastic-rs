//! # Trait definitions
//!
//! Foundational `FloatExt` / `SimdFloatExt` traits, generic callables and the
//! `DistributionExt` characteristic-function interface.

use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;
use std::ops::SubAssign;

use ndarray::Array1;
use ndarray::ScalarOperand;
use num_complex::Complex;
use num_complex::Complex64;
use rand::Rng;

pub enum Fn1D<T: FloatExt> {
  Native(fn(T) -> T),
  #[cfg(feature = "python")]
  Py(pyo3::Py<pyo3::PyAny>),
}

impl<T: FloatExt> Fn1D<T> {
  pub fn call(&self, t: T) -> T {
    match self {
      Fn1D::Native(f) => f(t),
      #[cfg(feature = "python")]
      Fn1D::Py(callable) => pyo3::Python::attach(|py| {
        let result: f64 = callable
          .call1(py, (t.to_f64().unwrap(),))
          .unwrap()
          .extract(py)
          .unwrap();
        T::from_f64_fast(result)
      }),
    }
  }
}

impl<T: FloatExt> From<fn(T) -> T> for Fn1D<T> {
  fn from(f: fn(T) -> T) -> Self {
    Fn1D::Native(f)
  }
}

pub enum Fn2D<T: FloatExt> {
  Native(fn(T, T) -> T),
  #[cfg(feature = "python")]
  Py(pyo3::Py<pyo3::PyAny>),
}

impl<T: FloatExt> Fn2D<T> {
  pub fn call(&self, t: T, u: T) -> T {
    match self {
      Fn2D::Native(f) => f(t, u),
      #[cfg(feature = "python")]
      Fn2D::Py(callable) => pyo3::Python::attach(|py| {
        let result: f64 = callable
          .call1(py, (t.to_f64().unwrap(), u.to_f64().unwrap()))
          .unwrap()
          .extract(py)
          .unwrap();
        T::from_f64_fast(result)
      }),
    }
  }
}

impl<T: FloatExt> From<fn(T, T) -> T> for Fn2D<T> {
  fn from(f: fn(T, T) -> T) -> Self {
    Fn2D::Native(f)
  }
}

#[cfg(feature = "python")]
pub struct CallableDist<T: FloatExt> {
  callable: pyo3::Py<pyo3::PyAny>,
  _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "python")]
impl<T: FloatExt> CallableDist<T> {
  pub fn new(callable: pyo3::Py<pyo3::PyAny>) -> Self {
    Self {
      callable,
      _phantom: std::marker::PhantomData,
    }
  }
}

#[cfg(feature = "python")]
impl<T: FloatExt> rand_distr::Distribution<T> for CallableDist<T> {
  fn sample<R: rand::Rng + ?Sized>(&self, _rng: &mut R) -> T {
    pyo3::Python::attach(|py| {
      let result: f64 = self.callable.call0(py).unwrap().extract::<f64>(py).unwrap();
      T::from_f64_fast(result)
    })
  }
}

pub trait SimdFloatExt: num_traits::Float + Default + Send + Sync + 'static {
  type Simd: Copy
    + std::ops::Mul<Output = Self::Simd>
    + std::ops::Add<Output = Self::Simd>
    + std::ops::Sub<Output = Self::Simd>
    + std::ops::Div<Output = Self::Simd>
    + std::ops::Neg<Output = Self::Simd>;

  fn splat(val: Self) -> Self::Simd;
  fn simd_from_array(arr: [Self; 8]) -> Self::Simd;
  fn simd_to_array(v: Self::Simd) -> [Self; 8];
  fn simd_ln(v: Self::Simd) -> Self::Simd;
  fn simd_sqrt(v: Self::Simd) -> Self::Simd;
  fn simd_cos(v: Self::Simd) -> Self::Simd;
  fn simd_sin(v: Self::Simd) -> Self::Simd;
  fn simd_exp(v: Self::Simd) -> Self::Simd;
  fn simd_tan(v: Self::Simd) -> Self::Simd;
  fn simd_max(a: Self::Simd, b: Self::Simd) -> Self::Simd;
  fn simd_powf(v: Self::Simd, exp: Self) -> Self::Simd;
  fn simd_floor(v: Self::Simd) -> Self::Simd;
  fn fill_uniform<R: Rng + ?Sized>(rng: &mut R, out: &mut [Self]);
  fn fill_uniform_simd(rng: &mut crate::simd_rng::SimdRng, out: &mut [Self]);
  fn sample_uniform<R: Rng + ?Sized>(rng: &mut R) -> Self;
  #[inline(always)]
  fn sample_uniform_simd(rng: &mut crate::simd_rng::SimdRng) -> Self {
    let mut buf = [Self::zero(); 8];
    Self::fill_uniform_simd(rng, &mut buf);
    buf[0]
  }
  fn simd_from_i32x8(v: wide::i32x8) -> Self::Simd;
  const PREFERS_F32_WN: bool = false;
  fn from_f64_fast(v: f64) -> Self;
  #[inline(always)]
  fn from_f32_fast(v: f32) -> Self {
    Self::from_f64_fast(v as f64)
  }
  fn pi() -> Self;
  fn two_pi() -> Self;
  fn min_positive_val() -> Self;
}

pub trait FloatExt:
  num_traits::Float
  + num_traits::FromPrimitive
  + num_traits::Signed
  + num_traits::FloatConst
  + Sum
  + SimdFloatExt
  + num_traits::Zero
  + Default
  + Debug
  + Send
  + Sync
  + ScalarOperand
  + AddAssign
  + SubAssign
  + 'static
{
  fn from_usize_(n: usize) -> Self;
  fn fill_standard_normal_slice(out: &mut [Self]);
  #[inline]
  fn fill_standard_normal_scaled_slice(out: &mut [Self], scale: Self) {
    Self::fill_standard_normal_slice(out);
    for x in out.iter_mut() {
      *x = *x * scale;
    }
  }
  fn with_fgn_complex_scratch<R, F: FnOnce(&mut [Complex<Self>]) -> R>(len: usize, f: F) -> R;
  fn normal_array(n: usize, mean: Self, std_dev: Self) -> Array1<Self>;
}

pub trait DistributionExt {
  fn characteristic_function(&self, _t: f64) -> Complex64 {
    Complex64::new(0.0, 0.0)
  }

  fn pdf(&self, _x: f64) -> f64 {
    0.0
  }

  fn cdf(&self, _x: f64) -> f64 {
    0.0
  }

  fn inv_cdf(&self, _p: f64) -> f64 {
    0.0
  }

  fn mean(&self) -> f64 {
    0.0
  }

  fn median(&self) -> f64 {
    0.0
  }

  fn mode(&self) -> f64 {
    0.0
  }

  fn variance(&self) -> f64 {
    0.0
  }

  fn skewness(&self) -> f64 {
    0.0
  }

  fn kurtosis(&self) -> f64 {
    0.0
  }

  fn entropy(&self) -> f64 {
    0.0
  }

  fn moment_generating_function(&self, _t: f64) -> f64 {
    0.0
  }
}

/// Rust-side bulk sampling API for distribution structs.
///
/// Implementors provide `fill_slice`; `sample_n` and `sample_matrix` are
/// lock-free convenience methods that allocate and fill contiguous buffers.
pub trait DistributionSampler<T> {
  fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]);

  #[inline]
  fn sample_n(&self, n: usize) -> Array1<T> {
    let mut out = Array1::<T>::uninit(n);
    let flat_uninit = out
      .as_slice_mut()
      .expect("distribution sample_n output must be contiguous");
    let flat = unsafe {
      // SAFETY: `flat_uninit` points to the output storage and `fill_slice`
      // fully initializes every element before `assume_init` below.
      std::slice::from_raw_parts_mut(flat_uninit.as_mut_ptr().cast::<T>(), flat_uninit.len())
    };
    let mut rng = crate::simd_rng::SimdRng::new();
    self.fill_slice(&mut rng, flat);
    unsafe {
      // SAFETY: all elements were initialized by `fill_slice` above.
      out.assume_init()
    }
  }

  #[inline]
  fn sample_matrix(&self, m: usize, n: usize) -> ndarray::Array2<T>
  where
    Self: Clone + Send,
    T: Send,
  {
    let mut out = ndarray::Array2::<T>::uninit((m, n));
    if m == 0 || n == 0 {
      return unsafe {
        // SAFETY: zero-length arrays have no elements to initialize.
        out.assume_init()
      };
    }
    let flat_uninit = out
      .as_slice_mut()
      .expect("distribution sample_matrix output must be contiguous");
    let flat = unsafe {
      // SAFETY: `flat_uninit` points to the output storage and each element
      // is initialized exactly once by the serial or parallel fill below.
      std::slice::from_raw_parts_mut(flat_uninit.as_mut_ptr().cast::<T>(), flat_uninit.len())
    };
    const MIN_PAR_CHUNK: usize = 16 * 1024;
    let total = flat.len();
    let max_workers_for_size = total.div_ceil(MIN_PAR_CHUNK).max(1);
    let workers = rayon::current_num_threads()
      .max(1)
      .min(max_workers_for_size);
    if workers == 1 {
      let mut rng = crate::simd_rng::SimdRng::new();
      self.fill_slice(&mut rng, flat);
      return unsafe {
        // SAFETY: all elements were initialized by `fill_slice`.
        out.assume_init()
      };
    }
    let chunk_len = total.div_ceil(workers);
    let base = self.clone();

    rayon::scope(move |scope| {
      for chunk in flat.chunks_mut(chunk_len) {
        let sampler = base.clone();
        scope.spawn(move |_| {
          let mut rng = crate::simd_rng::SimdRng::new();
          sampler.fill_slice(&mut rng, chunk);
        });
      }
    });
    unsafe {
      // SAFETY: every chunk is fully initialized by its worker.
      out.assume_init()
    }
  }
}
