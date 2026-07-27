//! Jump-size specifications.
//!
//! [`ProcessExt`](super::ProcessExt) requires `Send + Sync`, which propagates
//! into every process that stores a jump-size distribution. That bound is what
//! kept the workspace's own `Simd*` distributions out of the jump slot: they
//! own an `UnsafeCell` sample buffer and are `!Sync` by construction.
//!
//! The way out is that [`ProcessExt::sample_par`](super::ProcessExt::sample_par)
//! builds **one sampler per rayon worker** via `map_init`. Nothing forces the
//! sampler to share the distribution — only the process has to be `Sync`. So a
//! process stores a [`JumpSpec`] (parameters, hence `Sync`) and each sampler
//! turns it into its own [`JumpSource`], which is free to be `!Sync`.
//!
//! Rust coherence does not allow both a blanket impl over
//! `rand_distr::Distribution` and dedicated SIMD specs — the compiler cannot
//! prove that [`NormalJump`] does not also implement `Distribution`. Any
//! `rand_distr` distribution therefore goes through the [`RandDist`] newtype,
//! which keeps the previous borrowing behaviour exactly.
use std::marker::PhantomData;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;

/// A per-sampler source of jump sizes.
pub trait JumpSource<T: FloatExt> {
  /// Draw a single jump size.
  fn draw(&mut self) -> T;

  /// Fill `out` with jump sizes.
  ///
  /// The default is a scalar loop over [`Self::draw`]. SIMD-backed sources
  /// override it with a bulk fill, which is where they actually pay off.
  fn fill(&mut self, out: &mut [T]) {
    for x in out.iter_mut() {
      *x = self.draw();
    }
  }
}

/// A jump-size specification: parameters only, therefore `Send + Sync`, that
/// each sampler expands into its own [`JumpSource`].
pub trait JumpSpec<T: FloatExt>: Send + Sync {
  /// The draw source this spec builds. May borrow from the spec, which is how
  /// the `rand_distr` blanket impl preserves the old borrowing behaviour.
  type Source<'a>: JumpSource<T>
  where
    Self: 'a;

  /// Build a source. Callers pass a seed source so each sampler gets an
  /// independent stream.
  fn build<'a, S: SeedExt>(&'a self, seed: &S) -> Self::Source<'a>;
}

/// Source produced by the `rand_distr` blanket impl: borrows the distribution
/// and drives it with a seeded RNG, matching the pre-`JumpSpec` behaviour.
pub struct BorrowedDist<'a, T: FloatExt, D, R> {
  dist: &'a D,
  rng: R,
  _marker: PhantomData<T>,
}

impl<T, D, R> JumpSource<T> for BorrowedDist<'_, T, D, R>
where
  T: FloatExt,
  D: Distribution<T>,
  R: Rng,
{
  #[inline]
  fn draw(&mut self) -> T {
    self.dist.sample(&mut self.rng)
  }
}

/// Adapts any `rand_distr::Distribution` into a [`JumpSpec`].
///
/// Draws stay scalar and the distribution is borrowed by the source, which is
/// the behaviour every jump process had before `JumpSpec` existed. Reach for
/// [`NormalJump`] or [`ExpJump`] when the jump size is normal or exponential —
/// those fill in bulk through the workspace's SIMD samplers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandDist<D>(pub D);

impl<T, D> JumpSpec<T> for RandDist<D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Source<'a>
    = BorrowedDist<'a, T, D, SimdRng>
  where
    Self: 'a;

  fn build<'a, S: SeedExt>(&'a self, seed: &S) -> Self::Source<'a> {
    BorrowedDist {
      dist: &self.0,
      rng: seed.rng(),
      _marker: PhantomData,
    }
  }
}

/// Normal jump sizes drawn through [`SimdNormal`], with a bulk [`JumpSource::fill`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalJump<T> {
  /// Mean jump size.
  pub mean: T,
  /// Jump size standard deviation.
  pub std_dev: T,
}

impl<T: FloatExt> NormalJump<T> {
  /// # Panics
  /// Panics if `std_dev` is not strictly positive.
  pub fn new(mean: T, std_dev: T) -> Self {
    assert!(std_dev > T::zero(), "std_dev must be > 0");
    Self { mean, std_dev }
  }
}

impl<T: FloatExt> JumpSource<T> for SimdNormal<T> {
  #[inline]
  fn draw(&mut self) -> T {
    self.sample_fast()
  }

  #[inline]
  fn fill(&mut self, out: &mut [T]) {
    self.fill_slice_fast(out);
  }
}

impl<T: FloatExt> JumpSpec<T> for NormalJump<T> {
  type Source<'a>
    = SimdNormal<T>
  where
    Self: 'a;

  fn build<'a, S: SeedExt>(&'a self, seed: &S) -> Self::Source<'a> {
    SimdNormal::<T>::new(self.mean, self.std_dev, seed)
  }
}

/// Exponential jump sizes drawn through [`SimdExp`], with a bulk fill.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpJump<T> {
  /// Rate parameter.
  pub lambda: T,
}

impl<T: FloatExt> ExpJump<T> {
  /// # Panics
  /// Panics if `lambda` is not strictly positive.
  pub fn new(lambda: T) -> Self {
    assert!(lambda > T::zero(), "lambda must be > 0");
    Self { lambda }
  }
}

impl<T: FloatExt> JumpSource<T> for SimdExp<T> {
  #[inline]
  fn draw(&mut self) -> T {
    self.sample_fast()
  }

  #[inline]
  fn fill(&mut self, out: &mut [T]) {
    self.fill_slice_fast(out);
  }
}

impl<T: FloatExt> JumpSpec<T> for ExpJump<T> {
  type Source<'a>
    = SimdExp<T>
  where
    Self: 'a;

  fn build<'a, S: SeedExt>(&'a self, seed: &S) -> Self::Source<'a> {
    SimdExp::<T>::new(self.lambda, seed)
  }
}

#[cfg(test)]
mod tests {
  use rand_distr::Normal;
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::ExpJump;
  use super::JumpSource;
  use super::JumpSpec;
  use super::NormalJump;
  use super::RandDist;

  /// The whole point of the split: a spec is `Sync` (so a process holding one
  /// stays `Sync`), while the source it builds need not be.
  #[test]
  fn specs_are_sync_even_when_their_sources_are_not() {
    fn assert_sync<T: Send + Sync>() {}
    assert_sync::<NormalJump<f64>>();
    assert_sync::<ExpJump<f64>>();
    assert_sync::<RandDist<Normal<f64>>>();
  }

  #[test]
  fn simd_specs_fill_in_bulk_and_are_reproducible() {
    let spec = NormalJump::<f64>::new(0.0, 1.0);
    let fill = |seed: u64| {
      let mut src = spec.build(&Deterministic::new(seed));
      let mut out = vec![0.0; 4096];
      src.fill(&mut out);
      out
    };
    assert_eq!(fill(42), fill(42), "same seed must reproduce");
    assert_ne!(fill(42), fill(43), "different seeds must differ");

    let xs = fill(42);
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    assert!(mean.abs() < 6.0 / (xs.len() as f64).sqrt(), "mean = {mean}");
  }

  #[test]
  fn exp_spec_draws_are_non_negative() {
    let spec = ExpJump::<f64>::new(1.5);
    let mut src = spec.build(&Deterministic::new(7));
    let mut out = vec![0.0; 2048];
    src.fill(&mut out);
    assert!(out.iter().all(|x| *x >= 0.0));
  }

  /// The `rand_distr` path keeps working, just without the bulk fill.
  #[test]
  fn rand_dist_adapter_draws() {
    let spec = RandDist(Normal::<f64>::new(0.0, 1.0).unwrap());
    let mut src = spec.build(&Deterministic::new(11));
    let x = src.draw();
    assert!(x.is_finite());
  }
}
