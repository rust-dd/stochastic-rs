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

/// Closed forms of a law: `None` where it has none, never a silent zero; `Some(NaN)` only where the
/// quantity provably does not exist or the argument lies outside its domain; `kurtosis` is excess.
pub trait DistributionExt {
  fn characteristic_function(&self, _t: f64) -> Option<Complex64> {
    None
  }

  fn pdf(&self, _x: f64) -> Option<f64> {
    None
  }

  fn cdf(&self, _x: f64) -> Option<f64> {
    None
  }

  fn quantile(&self, _p: f64) -> Option<f64> {
    None
  }

  fn mean(&self) -> Option<f64> {
    None
  }

  fn median(&self) -> Option<f64> {
    None
  }

  fn mode(&self) -> Option<f64> {
    None
  }

  fn variance(&self) -> Option<f64> {
    None
  }

  fn skewness(&self) -> Option<f64> {
    None
  }

  fn kurtosis(&self) -> Option<f64> {
    None
  }

  fn entropy(&self) -> Option<f64> {
    None
  }

  /// `E[e⁰] = 1` for every law, so the default answers `Some(1.0)` at `t = 0` and `None` elsewhere.
  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    (t == 0.0).then_some(1.0)
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
    let (mut state, _) = self.init::<SimdRng, _>(&Deterministic::new(rng.next_u64()));
    self.fill(&mut state, out);
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
    let workers = worker_count(m.saturating_mul(n));
    if workers == 1 {
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
