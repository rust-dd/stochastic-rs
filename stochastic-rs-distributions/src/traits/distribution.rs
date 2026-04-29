//! Characteristic-function / pdf / cdf / moments interface and bulk samplers.

use ndarray::Array1;
use num_complex::Complex64;
use rand::Rng;

/// Analytical descriptors of a distribution.
///
/// All methods are provided with default implementations that **panic** via
/// [`unimplemented!()`]. Implementors override the methods that have a known
/// closed form for that distribution. This is intentional: silently returning
/// zero (the previous default) masked missing implementations and produced
/// downstream numerical bugs in pricing / calibration code.
pub trait DistributionExt {
  fn characteristic_function(&self, _t: f64) -> Complex64 {
    unimplemented!(
      "DistributionExt::characteristic_function is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn pdf(&self, _x: f64) -> f64 {
    unimplemented!(
      "DistributionExt::pdf is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn cdf(&self, _x: f64) -> f64 {
    unimplemented!(
      "DistributionExt::cdf is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn inv_cdf(&self, _p: f64) -> f64 {
    unimplemented!(
      "DistributionExt::inv_cdf is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn mean(&self) -> f64 {
    unimplemented!(
      "DistributionExt::mean is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn median(&self) -> f64 {
    unimplemented!(
      "DistributionExt::median is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn mode(&self) -> f64 {
    unimplemented!(
      "DistributionExt::mode is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn variance(&self) -> f64 {
    unimplemented!(
      "DistributionExt::variance is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn skewness(&self) -> f64 {
    unimplemented!(
      "DistributionExt::skewness is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn kurtosis(&self) -> f64 {
    unimplemented!(
      "DistributionExt::kurtosis is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn entropy(&self) -> f64 {
    unimplemented!(
      "DistributionExt::entropy is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn moment_generating_function(&self, _t: f64) -> f64 {
    unimplemented!(
      "DistributionExt::moment_generating_function is not implemented for {}",
      std::any::type_name::<Self>()
    )
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
