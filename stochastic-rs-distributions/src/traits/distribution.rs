//! Characteristic-function / pdf / cdf / moments interface and the sealed sampling traits.

use std::fmt::Debug;

use ndarray::Array1;
use ndarray::Array2;
use num_complex::Complex64;
use num_traits::Zero;
use rand::Rng;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_core::simd_rng::SimdRngExt;

use crate::seeded::Seeded;
use crate::seeded::worker_count;

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

mod sealed {
  pub trait Sealed {}
}

pub(crate) use sealed::Sealed;

/// A law whose parameters are plain data and whose stream state lives in [`Seeded`]; sealed to this crate.
pub trait SimdDistribution: Sealed + Clone + Send + Sync + 'static {
  #[doc(hidden)]
  type State<R: SimdRngExt>: Send + Sync + Clone + Debug;

  /// Draws the sub-streams in the law's fixed order; returns the fork basis.
  #[doc(hidden)]
  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (Self::State<R>, u64);

  /// Binds the law to one seeded [`SimdRng`] stream.
  fn seeded<S: SeedExt>(self, seed: &S) -> Seeded<Self, SimdRng> {
    Seeded::new(self, seed)
  }
}

/// A law with a bulk SIMD kernel and a buffered single draw; sealed, the kernels are the crate's own.
pub trait SimdKernel: SimdDistribution {
  /// One draw: the scalar of a univariate law, `Complex<T>` for `ComplexDistribution`.
  type Item: Copy + Zero + Send + Sync + 'static;

  #[doc(hidden)]
  fn fill<R: SimdRngExt>(&self, state: &mut Self::State<R>, out: &mut [Self::Item]);

  #[doc(hidden)]
  fn next<R: SimdRngExt>(&self, state: &mut Self::State<R>) -> Self::Item;

  /// Bulk SIMD fill driven by any rng: one `u64` from `rng` seeds a private stream.
  fn fill_with<G: Rng + ?Sized>(&self, rng: &mut G, out: &mut [Self::Item]) {
    Seeded::<Self, SimdRng>::new(self.clone(), &Deterministic::new(rng.next_u64())).fill_slice(out);
  }
}

/// Bulk sampling API of a [`Seeded`] stream; sealed, only `Seeded` implements it.
pub trait DistributionSampler<T: Copy + Zero + Send>: Sealed {
  /// Fills `out` from the stream's engine in bulk, past the single-draw buffer that `sample` pops.
  fn fill_slice(&mut self, out: &mut [T]);

  /// An independent child stream for worker `stream_idx`; advances this stream's fork basis.
  fn fork(&mut self, stream_idx: u64) -> Self
  where
    Self: Sized;

  /// `n` draws in a new array: the values `fill_slice` gives an `n`-slice.
  fn sample_n(&mut self, n: usize) -> Array1<T> {
    let mut out = Array1::<T>::zeros(n);
    self.fill_slice(out.as_slice_mut().expect("sample_n output is contiguous"));
    out
  }

  /// Fills an `m × n` matrix, in parallel above 16Ki values over streams forked on this thread:
  /// the output depends on the seed, `(m, n)` and the call count alone, never on the pool size.
  fn sample_matrix(&mut self, m: usize, n: usize) -> Array2<T>
  where
    Self: Sized + Send,
  {
    if m == 0 || n == 0 {
      return Array2::<T>::zeros((m, n));
    }
    if worker_count(m.saturating_mul(n)) == 1 {
      let mut out = Array2::<T>::zeros((m, n));
      self.fill_slice(
        out
          .as_slice_mut()
          .expect("sample_matrix output is contiguous"),
      );
      return out;
    }
    // Each worker zeroes its own chunk: zeroing the whole matrix up front measured slower here.
    let mut out = Array2::<T>::uninit((m, n));
    let flat = out
      .as_slice_mut()
      .expect("sample_matrix output is contiguous");
    let workers = worker_count(flat.len());
    let chunk_len = flat.len().div_ceil(workers);
    let forked = (0..workers)
      .map(|idx| self.fork(idx as u64))
      .collect::<Vec<_>>();
    rayon::scope(move |scope| {
      for (mut worker, chunk) in forked.into_iter().zip(flat.chunks_mut(chunk_len)) {
        scope.spawn(move |_| {
          for x in chunk.iter_mut() {
            x.write(T::zero());
          }
          // SAFETY: every element of `chunk` was written just above.
          let chunk =
            unsafe { std::slice::from_raw_parts_mut(chunk.as_mut_ptr().cast::<T>(), chunk.len()) };
          worker.fill_slice(chunk);
        });
      }
    });
    // SAFETY: the chunks partition the storage and every worker initialised its own.
    unsafe { out.assume_init() }
  }
}
