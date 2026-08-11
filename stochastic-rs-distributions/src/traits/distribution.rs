//! Characteristic-function / pdf / cdf / moments interface and bulk samplers.

use ndarray::Array1;
use num_complex::Complex64;

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
///
/// There is no `fill_slice(rng, out)` overload: per the crate's RNG policy,
/// the internal seeded stream is the only stream a sampler draws from.
/// Construct with `Deterministic::new(seed)` for reproducible output; `Self`
/// stores everything sampling needs, so these methods take no `Rng` of
/// their own.
pub trait DistributionSampler<T> {
  /// Fills `out` by drawing from this sampler's own internal SIMD RNG
  /// stream, seeded at construction.
  fn fill_slice(&self, out: &mut [T]);

  /// Builds an independent worker stream for the `stream_idx`-th chunk of a
  /// parallel `sample_matrix` fan-out.
  ///
  /// Deterministic when `self` was constructed from a [`Deterministic`]
  /// seed — the same `stream_idx` always yields a child with the same
  /// output — so two identically-seeded samplers produce bit-identical
  /// `sample_matrix` results regardless of thread count. Independent-random
  /// when `self` was constructed from [`Unseeded`], matching the un-forked
  /// parallel behavior.
  ///
  /// [`Deterministic`]: crate::simd_rng::Deterministic
  /// [`Unseeded`]: crate::simd_rng::Unseeded
  #[doc(hidden)]
  fn fork(&self, stream_idx: u64) -> Self;

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
    self.fill_slice(flat);
    unsafe {
      // SAFETY: all elements were initialized by `fill_slice` above.
      out.assume_init()
    }
  }

  #[inline]
  fn sample_matrix(&self, m: usize, n: usize) -> ndarray::Array2<T>
  where
    Self: Sized + Send,
    T: Send,
  {
    // `Simd*` samplers own an `UnsafeCell` buffer and are deliberately
    // `!Sync` (see the crate-level RNG policy), so the parallel branch
    // below must never capture `&self` inside the `rayon::scope` closure —
    // that would need `Self: Sync`, which no `Simd*` type can satisfy.
    // Every worker is forked to an owned value on this thread first, and
    // only owned `Self` values (already `Send`) cross into the closure.
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
      self.fill_slice(flat);
      return unsafe {
        // SAFETY: all elements were initialized by `fill_slice`.
        out.assume_init()
      };
    }
    let chunk_len = total.div_ceil(workers);
    let forked_workers = (0..workers)
      .map(|stream_idx| self.fork(stream_idx as u64))
      .collect::<Vec<_>>();

    rayon::scope(move |scope| {
      for (worker, chunk) in forked_workers.into_iter().zip(flat.chunks_mut(chunk_len)) {
        scope.spawn(move |_| {
          worker.fill_slice(chunk);
        });
      }
    });
    unsafe {
      // SAFETY: every chunk is fully initialized by its worker.
      out.assume_init()
    }
  }
}
