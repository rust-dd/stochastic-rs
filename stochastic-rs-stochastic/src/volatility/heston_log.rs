//! # Heston Log
//!
//! $$
//! \begin{aligned}
//! d\ln S_t &= (\mu - \tfrac12 v_t)\,dt + \sqrt{v_t}\,dW_t^S \\
//! dv_t &= \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v
//! \end{aligned}
//! $$
//!
//! where $\langle dW^S, dW^v\rangle = \rho\,dt$.
//! Log-spot simulation guarantees $S_t > 0$.
//!
//! Reference: Heston S. L. (1993) — *A Closed-Form Solution for Options
//! with Stochastic Volatility with Applications to Bond and Currency
//! Options*, Review of Financial Studies 6(2), 327–343,
//! DOI: 10.1093/rfs/6.2.327.
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Cpu;
use crate::params::Drift;
use crate::params::VarianceBoundary;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Every field has a matching `with_*` builder setter, e.g.
/// `HestonLog::new(..).with_kappa(2.0).with_rho(-0.4)`.
#[derive(Debug, Clone)]
pub struct HestonLog<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// The asset's drift: `μ`, a cost of carry, or the rate pair.
  pub drift: Drift<T>,
  /// Variance mean-reversion speed
  pub kappa: T,
  /// Long-run variance level
  pub theta: T,
  /// Volatility of variance (vol-of-vol)
  pub xi: T,
  /// Correlation between asset and variance Brownian motions
  pub rho: T,
  /// Grid points sampled along the path; `dt = t / (n − 1)`.
  pub n: usize,
  /// Initial spot.
  pub s0: T,
  /// Initial variance.
  pub v0: T,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Treatment of a negative square-root state after a step; the Feller condition is not required.
  pub boundary: VarianceBoundary,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> HestonLog<T, S> {
  pub fn new(
    drift: Drift<T>,
    kappa: T,
    theta: T,
    xi: T,
    rho: T,
    n: usize,
    s0: T,
    v0: T,
    t: Option<T>,
    boundary: VarianceBoundary,
    seed: S,
  ) -> Self {
    drift.assert_finite();
    assert!(
      kappa.is_finite(),
      "kappa must satisfy `kappa.is_finite()`, got kappa = {kappa:?}"
    );
    assert!(
      theta.is_finite(),
      "theta must satisfy `theta.is_finite()`, got theta = {theta:?}"
    );
    assert!(
      xi.is_finite(),
      "xi must satisfy `xi.is_finite()`, got xi = {xi:?}"
    );
    assert!(
      rho.is_finite(),
      "rho must satisfy `rho.is_finite()`, got rho = {rho:?}"
    );
    assert!(
      s0.is_finite(),
      "s0 must satisfy `s0.is_finite()`, got s0 = {s0:?}"
    );
    assert!(
      v0.is_finite(),
      "v0 must satisfy `v0.is_finite()`, got v0 = {v0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      kappa >= T::zero(),
      "kappa must satisfy `kappa >= T::zero()`, got kappa = {kappa:?}"
    );
    assert!(
      theta >= T::zero(),
      "theta must satisfy `theta >= T::zero()`, got theta = {theta:?}"
    );
    assert!(
      xi >= T::zero(),
      "xi must satisfy `xi >= T::zero()`, got xi = {xi:?}"
    );
    assert!(
      rho >= -T::one() && rho <= T::one(),
      "rho must satisfy `rho >= -T::one() && rho <= T::one()`, got rho = {rho:?}"
    );
    assert!(
      v0 >= T::zero(),
      "v0 must satisfy `v0 >= T::zero()`, got v0 = {v0:?}"
    );
    assert!(
      s0 > T::zero(),
      "s0 must satisfy `s0 > T::zero()`, got s0 = {s0:?}"
    );

    Self {
      backend: Cpu,
      drift,
      kappa,
      theta,
      xi,
      rho,
      n,
      s0,
      v0,
      t,
      boundary,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> HestonLog<T, S, B> {
  /// Replace `drift`, all else unchanged; panics on a non-finite rate.
  pub fn with_drift(mut self, drift: Drift<T>) -> Self {
    drift.assert_finite();
    self.drift = drift;
    self
  }

  /// Replace `kappa`, all else unchanged.
  pub fn with_kappa(mut self, kappa: T) -> Self {
    assert!(
      kappa.is_finite(),
      "kappa must satisfy `kappa.is_finite()`, got kappa = {kappa:?}"
    );
    assert!(
      kappa >= T::zero(),
      "kappa must satisfy `kappa >= T::zero()`, got kappa = {kappa:?}"
    );
    self.kappa = kappa;
    self
  }

  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: T) -> Self {
    assert!(
      theta.is_finite(),
      "theta must satisfy `theta.is_finite()`, got theta = {theta:?}"
    );
    assert!(
      theta >= T::zero(),
      "theta must satisfy `theta >= T::zero()`, got theta = {theta:?}"
    );
    self.theta = theta;
    self
  }

  /// Replace `xi`, all else unchanged.
  pub fn with_xi(mut self, xi: T) -> Self {
    assert!(
      xi.is_finite(),
      "xi must satisfy `xi.is_finite()`, got xi = {xi:?}"
    );
    assert!(
      xi >= T::zero(),
      "xi must satisfy `xi >= T::zero()`, got xi = {xi:?}"
    );
    self.xi = xi;
    self
  }

  /// Replace `rho`; a field write, since `sampler()` builds its correlated noise from `rho`.
  pub fn with_rho(mut self, rho: T) -> Self {
    assert!(
      rho.is_finite(),
      "rho must satisfy `rho.is_finite()`, got rho = {rho:?}"
    );
    assert!(
      rho >= -T::one() && rho <= T::one(),
      "rho must satisfy `rho >= -T::one() && rho <= T::one()`, got rho = {rho:?}"
    );
    self.rho = rho;
    self
  }

  /// Replace `s0`, all else unchanged.
  pub fn with_s0(mut self, s0: T) -> Self {
    assert!(
      s0.is_finite(),
      "s0 must satisfy `s0.is_finite()`, got s0 = {s0:?}"
    );
    assert!(
      s0 > T::zero(),
      "s0 must satisfy `s0 > T::zero()`, got s0 = {s0:?}"
    );
    self.s0 = s0;
    self
  }

  /// Replace `v0`, all else unchanged.
  pub fn with_v0(mut self, v0: T) -> Self {
    assert!(
      v0.is_finite(),
      "v0 must satisfy `v0.is_finite()`, got v0 = {v0:?}"
    );
    assert!(
      v0 >= T::zero(),
      "v0 must satisfy `v0 >= T::zero()`, got v0 = {v0:?}"
    );
    self.v0 = v0;
    self
  }

  /// Replace `boundary`, all else unchanged.
  pub fn with_boundary(mut self, boundary: VarianceBoundary) -> Self {
    self.boundary = boundary;
    self
  }

  /// Replace `n`, the grid points along the path, all else unchanged.
  pub fn with_points(mut self, n: usize) -> Self {
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`, all else unchanged.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    self.t = t;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// The Euler engine's view of the log-price Heston model. The drift is
/// resolved once here from whichever of the rate, carry or drift arguments
/// the process was built with.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for HestonLog<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    match self.boundary {
      VarianceBoundary::Floor => crate::euler::EulerSpec::LogHeston {
        drift: self.drift.rate(),
        kappa: self.kappa,
        theta: self.theta,
        xi: self.xi,
        rho: self.rho,
      },
      VarianceBoundary::Reflect => crate::euler::EulerSpec::LogHestonReflected {
        drift: self.drift.rate(),
        kappa: self.kappa,
        theta: self.theta,
        xi: self.xi,
        rho: self.rho,
      },
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [self.s0, self.v0.max(T::zero()), T::zero(), T::zero()]
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] HestonLog<T, S> { drift, kappa, theta, xi, rho, n, s0, v0, t, boundary, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for HestonLog<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for HestonLog<T, S, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = HestonLogSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> HestonLogSampler<T> {
    // `saturating_sub(1).max(1)` keeps the noise std finite for n ≤ 1, where
    // the streams are never used; for n ≥ 2 it equals `n - 1`, so the std and
    // hence the derived stream match the legacy `sample` exactly.
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    HestonLogSampler {
      n: self.n,
      s0: self.s0,
      v0: self.v0.max(T::zero()),
      drift: self.drift.rate(),
      kappa: self.kappa,
      theta: self.theta,
      xi: self.xi,
      rho: self.rho,
      dt,
      boundary: self.boundary,
      n1: SimdNormal::<T>::new(T::zero(), sqrt_dt).seeded(&self.seed),
      n2: SimdNormal::<T>::new(T::zero(), sqrt_dt).seeded(&self.seed),
    }
  }

  /// Through the Euler engine: on a device every component steps in the
  /// kernel, on the host devices it is this process's own sampler, chunked
  /// exactly as `ProcessExt` chunks.
  fn sample(&self) -> [Array1<T>; 2] {
    self.backend.system_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&[Array1<T>; 2]) -> R + Sync) -> Vec<R> {
    self.backend.system_paths_map(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<[Array1<T>; 2]> {
    self.backend.system_paths(self, m)
  }

  fn try_sample(&self) -> Result<[Array1<T>; 2], crate::device::DeviceError> {
    self.backend.try_system_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 2]>, crate::device::DeviceError> {
    self.backend.try_system_paths(self, m)
  }
}

/// Reusable [`HestonLog`] sampling state: owns the two Gaussian streams (one
/// driving the asset, one combined into the variance shock) and the
/// precomputed drift / step size so a Monte-Carlo loop reuses both buffers.
#[doc(hidden)]
#[derive(Debug)]
pub struct HestonLogSampler<T: FloatExt> {
  n: usize,
  s0: T,
  v0: T,
  drift: T,
  kappa: T,
  theta: T,
  xi: T,
  rho: T,
  dt: T,
  boundary: VarianceBoundary,
  n1: Seeded<SimdNormal<T>>,
  n2: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt> HestonLogSampler<T> {
  fn fill_paths(&mut self, s: &mut [T], v: &mut [T]) {
    if self.n == 0 {
      return;
    }
    assert!(
      self.s0 > T::zero(),
      "s0 must satisfy `s0 > T::zero()`, got s0 = {:?}",
      self.s0
    );
    s[0] = self.s0;
    v[0] = self.v0;
    if self.n == 1 {
      return;
    }

    let n_increments = self.n - 1;
    let dt = self.dt;
    let mut dws = vec![T::zero(); n_increments];
    let mut z = vec![T::zero(); n_increments];
    let mut dwv = vec![T::zero(); n_increments];
    self.n1.fill_slice(&mut dws);
    self.n2.fill_slice(&mut z);
    let corr_scale = (T::one() - self.rho * self.rho).sqrt();
    for i in 0..n_increments {
      dwv[i] = self.rho * dws[i] + corr_scale * z[i];
    }

    let drift = self.drift;
    let half = T::from_f64_fast(0.5);

    for i in 1..self.n {
      let v_prev = match self.boundary {
        VarianceBoundary::Floor => v[i - 1].max(T::zero()),
        VarianceBoundary::Reflect => v[i - 1].abs(),
      };

      let sqrt_v = v_prev.sqrt();

      let log_inc = (drift - half * v_prev) * dt + sqrt_v * dws[i - 1];
      s[i] = s[i - 1] * log_inc.exp();

      let dv = self.kappa * (self.theta - v_prev) * dt + self.xi * sqrt_v * dwv[i - 1];
      v[i] = match self.boundary {
        VarianceBoundary::Floor => (v_prev + dv).max(T::zero()),
        VarianceBoundary::Reflect => (v_prev + dv).abs(),
      };
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for HestonLogSampler<T> {}

impl<T: FloatExt> PathSampler<T> for HestonLogSampler<T> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, v] = out;
    self.fill_paths(
      s.as_slice_mut()
        .expect("HestonLog output must be contiguous"),
      v.as_slice_mut()
        .expect("HestonLog output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut s = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);
    self.fill_paths(
      s.as_slice_mut().expect("contiguous"),
      v.as_slice_mut().expect("contiguous"),
    );
    [s, v]
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn price_stays_positive() {
    let p = HestonLog::new(
      Drift::Mu(0.05_f64),
      1.5,
      0.04,
      0.3,
      -0.7,
      256,
      100.0,
      0.04,
      Some(1.0),
      VarianceBoundary::Floor,
      Unseeded,
    );
    let [s, _v] = p.sample();
    assert!(s.iter().all(|x| *x > 0.0));
  }

  #[test]
  fn variance_stays_non_negative() {
    let p = HestonLog::new(
      Drift::Mu(0.05_f64),
      1.5,
      0.04,
      0.5,
      -0.7,
      256,
      100.0,
      0.04,
      Some(1.0),
      VarianceBoundary::Floor,
      Unseeded,
    );
    let [_s, v] = p.sample();
    assert!(v.iter().all(|x| *x >= 0.0));
  }
}

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyHestonLog;
