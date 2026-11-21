//! The Euler engine's view of [`HestonHullWhite2F`]: a launch carries the four state rows, and
//! the integrated short rate is rebuilt on the host from the rate row.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;

use super::HestonHullWhite2F;
use super::HestonHullWhite2FPaths;
use super::HestonHullWhite2FSampler;
use super::HybridScheme;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// The model in the engine's four slots `(S, v, r, u)`; it borrows, so it advances the process's
/// own seed.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct HestonHullWhite2FLaunch<'a, T: FloatExt, Sch: HybridScheme, S: SeedExt, B>(
  pub(super) &'a HestonHullWhite2F<T, Sch, S, B>,
);

impl<T: FloatExt, Sch: HybridScheme, S: SeedExt, B: crate::euler::EulerBackend<T>>
  crate::traits::Sealed for HestonHullWhite2FLaunch<'_, T, Sch, S, B>
{
}

impl<T: FloatExt, Sch: HybridScheme, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for HestonHullWhite2FLaunch<'_, T, Sch, S, B>
{
  type Output = [Array1<T>; 4];
  type Sampler<'s>
    = HestonHullWhite2FLaunchSampler<'s, T, Sch, S, B>
  where
    Self: 's;

  fn sampler(&self) -> HestonHullWhite2FLaunchSampler<'_, T, Sch, S, B> {
    HestonHullWhite2FLaunchSampler {
      inner: self.0.host_sampler(),
    }
  }
}

/// [`HestonHullWhite2FLaunch`]'s sampler: the model's own, its integrated-rate row dropped.
#[doc(hidden)]
#[derive(Debug)]
pub struct HestonHullWhite2FLaunchSampler<'a, T: FloatExt, Sch: HybridScheme, S: SeedExt, B> {
  inner: HestonHullWhite2FSampler<'a, T, Sch, S, B>,
}

impl<T: FloatExt, Sch: HybridScheme, S: SeedExt, B: Send + Sync> crate::traits::Sealed
  for HestonHullWhite2FLaunchSampler<'_, T, Sch, S, B>
{
}

impl<T: FloatExt, Sch: HybridScheme, S: SeedExt, B: Send + Sync> PathSampler<T>
  for HestonHullWhite2FLaunchSampler<'_, T, Sch, S, B>
{
  type Output = [Array1<T>; 4];

  fn sample_into(&mut self, out: &mut [Array1<T>; 4]) {
    *out = self.sample();
  }

  fn sample(&mut self) -> [Array1<T>; 4] {
    let HestonHullWhite2FPaths { s, v, r, u, .. } = self.inner.sample();
    [s, v, r, u]
  }
}

impl<T: FloatExt, Sch: HybridScheme, S: SeedExt, B: crate::euler::EulerBackend<T>>
  crate::euler::EulerSystem<T, 4> for HestonHullWhite2FLaunch<'_, T, Sch, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    Sch::euler_spec(self.0)
  }

  /// The families step `ln(S/S₀)` from zero and report `S₀ exp(·)`: the first spot is `S₀` itself.
  fn initial_state(&self) -> [T; 4] {
    let p = self.0;
    [T::zero(), p.v0, p.r0.unwrap_or(T::zero()), T::zero()]
  }

  fn grid_points(&self) -> usize {
    self.0.n
  }

  fn horizon(&self) -> T {
    self.0.t.unwrap_or(T::one())
  }

  fn curve(&self) -> Option<Vec<T>> {
    Some(self.0.rate_curve())
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.0.seed)
  }

  fn host_sample(&self) -> [Array1<T>; 4] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

/// `∫₀ᵗ r` by the trapezoid the host schemes accumulate, in their order of operations: on a
/// host backend the row equals the sampler's own bit for bit.
pub(super) fn integrated_rate<T: FloatExt>(r: &Array1<T>, dt: T) -> Array1<T> {
  let half = T::from_f64_fast(0.5);
  let mut out = Array1::<T>::zeros(r.len());
  let mut acc = T::zero();
  for i in 1..r.len() {
    acc += half * (r[i - 1] + r[i]) * dt;
    out[i] = acc;
  }
  out
}
