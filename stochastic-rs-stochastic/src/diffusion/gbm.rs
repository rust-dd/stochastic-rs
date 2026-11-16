//! # Gbm
//!
//! $$
//! dS_t=\mu S_t\,dt+\sigma S_t\,dW_t,\quad S_0=s_0
//! $$
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_distributions::special::norm_pdf;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::device::DeviceError;
use crate::euler::EulerBackend;
use crate::malliavin::Malliavin;
use crate::traits::DistributionExt;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Debug, Clone)]
pub struct Gbm<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Constant proportional drift rate μ. GBM has no mean reversion — `mu`
  /// is not a long-run level; it is the constant rate at which `S_t`
  /// grows (or shrinks) in expectation.
  pub mu: T,
  /// Diffusion scale σ multiplying `S_t dW_t`.
  pub sigma: T,
  /// Number of points sampled along the GBM path.
  pub n: usize,
  /// Initial value S₀; positive, since a multiplicative process at zero stays there.
  pub x0: T,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Cached parameters of the terminal log-normal: ln S_T ∼ N(`ln_mu`, `ln_sigma`).
  ln_mu: f64,
  ln_sigma: f64,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Recomputes the cached terminal-log-normal parameters `(ln_mu, ln_sigma)`
/// from the four public fields they derive from — shared by `new()` and by
/// every `with_*` setter that touches `mu`/`sigma`/`x0`/`t`, so the two can
/// never drift apart.
fn terminal_lognormal_cache<T: FloatExt>(mu: T, sigma: T, x0: T, t: Option<T>) -> (f64, f64) {
  let x0_f64 = x0.to_f64().unwrap();
  let mu_f64 = mu.to_f64().unwrap();
  let sigma_f64 = sigma.to_f64().unwrap();
  let t_f64 = t.unwrap_or(T::one()).to_f64().unwrap();

  let mu_ln = x0_f64.ln() + (mu_f64 - 0.5 * sigma_f64 * sigma_f64) * t_f64;
  let sigma_ln = sigma_f64 * t_f64.sqrt();
  (mu_ln, sigma_ln)
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Gbm::default().with_mu(0.08).with_sigma(0.35)`.
impl<T: FloatExt, S: SeedExt> Gbm<T, S> {
  pub fn new(mu: T, sigma: T, n: usize, x0: T, t: Option<T>, seed: S) -> Self {
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      x0.is_finite(),
      "x0 must satisfy `x0.is_finite()`, got x0 = {x0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(
      sigma >= T::zero(),
      "sigma must satisfy `sigma >= T::zero()`, got sigma = {sigma:?}"
    );
    assert!(
      x0 > T::zero(),
      "x0 must satisfy `x0 > T::zero()`, got x0 = {x0:?}"
    );
    let (ln_mu, ln_sigma) = terminal_lognormal_cache(mu, sigma, x0, t);

    Self {
      mu,
      sigma,
      n,
      x0,
      t,
      ln_mu,
      ln_sigma,
      seed,
      backend: Cpu,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Gbm<T, S, B> {
  /// Replace `mu`; rebuilds the cached terminal-log-normal parameters
  /// (`ln_mu`/`ln_sigma`, read by [`DistributionExt`]'s moment methods) so
  /// they stay consistent with the new `mu`.
  pub fn with_mu(mut self, mu: T) -> Self {
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    self.mu = mu;
    (self.ln_mu, self.ln_sigma) = terminal_lognormal_cache(mu, self.sigma, self.x0, self.t);
    self
  }

  /// Replace `sigma`; rebuilds the cached terminal-log-normal parameters.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    assert!(
      sigma >= T::zero(),
      "sigma must satisfy `sigma >= T::zero()`, got sigma = {sigma:?}"
    );
    self.sigma = sigma;
    (self.ln_mu, self.ln_sigma) = terminal_lognormal_cache(self.mu, sigma, self.x0, self.t);
    self
  }

  /// Replace `x0`; rebuilds the cached terminal-log-normal parameters.
  pub fn with_x0(mut self, x0: T) -> Self {
    assert!(
      x0.is_finite(),
      "x0 must satisfy `x0.is_finite()`, got x0 = {x0:?}"
    );
    assert!(
      x0 > T::zero(),
      "x0 must satisfy `x0 > T::zero()`, got x0 = {x0:?}"
    );
    self.x0 = x0;
    (self.ln_mu, self.ln_sigma) = terminal_lognormal_cache(self.mu, self.sigma, x0, self.t);
    self
  }

  /// Replace `n`, the grid points along the path; the terminal log-normal cache does not read it.
  pub fn with_points(mut self, n: usize) -> Self {
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`; rebuilds the cached
  /// terminal-log-normal parameters.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    self.t = t;
    (self.ln_mu, self.ln_sigma) = terminal_lognormal_cache(self.mu, self.sigma, self.x0, t);
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// μ=0.05, σ=0.2, x₀=100, t=1, a textbook equity (this file's tests); 252 points, a trading year of
/// daily closes.
impl<T: FloatExt> Default for Gbm<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(0.05),
      T::from_f64_fast(0.2),
      252,
      T::from_f64_fast(100.0),
      Some(T::one()),
      Unseeded,
    )
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Gbm<T, S> { mu, sigma, n, x0, t, ln_mu, ln_sigma, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: EulerBackend<T>> crate::traits::Sealed for Gbm<T, S, B> {}

impl<T: FloatExt, S: SeedExt, B: EulerBackend<T>> ProcessExt<T> for Gbm<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = GbmSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> GbmSampler<T> {
    // `saturating_sub(1).max(1)` keeps dt finite for the degenerate n ≤ 1
    // cases; for n ≥ 2 it equals `n - 1`, so the noise std and hence the
    // derived stream are identical to the pre-sampler path.
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    GbmSampler {
      n: self.n,
      x0: self.x0,
      drift_scale: self.mu * dt,
      diff_scale: self.sigma,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt()).seeded(&self.seed),
    }
  }

  fn sample(&self) -> Array1<T> {
    self.backend.euler_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    self.backend.euler_paths_map(self, m, f)
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    self.backend.euler_paths_map_view(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    self.backend.euler_paths(self, m)
  }

  fn try_sample(&self) -> Result<Array1<T>, DeviceError> {
    self.backend.try_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, DeviceError> {
    self.backend.try_euler_paths(self, m)
  }
}

/// Reusable [`Gbm`] sampling state: precomputed Euler scales and the owned
/// Gaussian source, so a Monte-Carlo loop pays the `SimdNormal` setup once.
#[doc(hidden)]
#[derive(Debug)]
pub struct GbmSampler<T: FloatExt> {
  n: usize,
  x0: T,
  drift_scale: T,
  diff_scale: T,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt> GbmSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.normal.fill_slice(tail);
    let mut prev = self.x0;
    for z in tail.iter_mut() {
      let next = prev + self.drift_scale * prev + self.diff_scale * prev * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for GbmSampler<T> {}

impl<T: FloatExt> PathSampler<T> for GbmSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Gbm output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

impl<T: FloatExt, S: SeedExt> Gbm<T, S> {
  fn terminal_lognormal_params(&self) -> Option<(f64, f64)> {
    if self.ln_mu.is_nan() || self.ln_sigma.is_nan() || self.ln_sigma <= 0.0 {
      None
    } else {
      Some((self.ln_mu, self.ln_sigma))
    }
  }

  /// The path and `D_{t_i} S_T = σ S_T`, the same at every grid point.
  pub fn malliavin(&self) -> Malliavin<T, Array1<T>> {
    let paths = self.sample();
    let derivative = Array1::from_elem(self.n, self.sigma * paths[paths.len() - 1]);
    Malliavin { paths, derivative }
  }
}

// Terminal distribution of the Gbm: S_T ∼ LogNormal(ln_mu, ln_sigma) where
// ln_mu = ln(S_0) + (μ − ½σ²)·T,   ln_sigma = σ·√T.
impl<T: FloatExt, S: SeedExt> DistributionExt for Gbm<T, S> {
  fn pdf(&self, x: f64) -> Option<f64> {
    let (ln_mu, ln_sigma) = self.terminal_lognormal_params()?;
    if x <= 0.0 {
      return Some(0.0);
    }
    let z = (x.ln() - ln_mu) / ln_sigma;
    Some(norm_pdf(z) / (ln_sigma * x))
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    let (ln_mu, ln_sigma) = self.terminal_lognormal_params()?;
    if x <= 0.0 {
      return Some(0.0);
    }
    Some(norm_cdf((x.ln() - ln_mu) / ln_sigma))
  }

  fn quantile(&self, p: f64) -> Option<f64> {
    let (ln_mu, ln_sigma) = self.terminal_lognormal_params()?;
    Some((ln_mu + ln_sigma * ndtri(p)).exp())
  }

  fn mean(&self) -> Option<f64> {
    let (ln_mu, ln_sigma) = self.terminal_lognormal_params()?;
    Some((ln_mu + 0.5 * ln_sigma * ln_sigma).exp())
  }

  fn mode(&self) -> Option<f64> {
    let (ln_mu, ln_sigma) = self.terminal_lognormal_params()?;
    Some((ln_mu - ln_sigma * ln_sigma).exp())
  }

  fn median(&self) -> Option<f64> {
    let (ln_mu, _) = self.terminal_lognormal_params()?;
    Some(ln_mu.exp())
  }

  fn variance(&self) -> Option<f64> {
    let (ln_mu, ln_sigma) = self.terminal_lognormal_params()?;
    let s2 = ln_sigma * ln_sigma;
    Some((s2.exp() - 1.0) * (2.0 * ln_mu + s2).exp())
  }

  fn skewness(&self) -> Option<f64> {
    let (_, ln_sigma) = self.terminal_lognormal_params()?;
    let s2 = ln_sigma * ln_sigma;
    Some((s2.exp() + 2.0) * (s2.exp() - 1.0).sqrt())
  }

  fn kurtosis(&self) -> Option<f64> {
    let (_, ln_sigma) = self.terminal_lognormal_params()?;
    let s2 = ln_sigma * ln_sigma;
    Some((4.0 * s2).exp() + 2.0 * (3.0 * s2).exp() + 3.0 * (2.0 * s2).exp() - 6.0)
  }

  fn entropy(&self) -> Option<f64> {
    let (ln_mu, ln_sigma) = self.terminal_lognormal_params()?;
    Some(0.5 + 0.5 * (2.0 * std::f64::consts::PI * ln_sigma * ln_sigma).ln() + ln_mu)
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  #[test]
  fn the_malliavin_derivative_is_sigma_times_the_terminal_value() {
    let gbm = || Gbm::<f64, _>::new(0.05, 0.2, 17, 100.0, Some(1.0), Deterministic::new(3));
    let Malliavin {
      paths, derivative, ..
    } = gbm().malliavin();
    assert_eq!(paths, gbm().sample());
    assert!(derivative.iter().all(|d| *d == 0.2 * paths[16]));
  }

  /// The lognormal terminal law: ten closed forms, and `None` for its cf and mgf, which have none.
  #[test]
  fn the_terminal_law_answers_exactly_its_closed_forms() {
    let gbm = Gbm::new(0.05_f64, 0.2, 10, 100.0, Some(1.0), Unseeded);
    assert!(gbm.characteristic_function(0.4).is_none());
    assert!(gbm.moment_generating_function(0.4).is_none());
    assert!(gbm.moment_generating_function(0.0).is_none());
    assert!(gbm.pdf(100.0).is_some() && gbm.cdf(100.0).is_some() && gbm.quantile(0.3).is_some());
    assert!(gbm.mean().is_some() && gbm.median().is_some() && gbm.mode().is_some());
    assert!(gbm.variance().is_some() && gbm.skewness().is_some() && gbm.entropy().is_some());
    // scipy.stats.lognorm(0.2).stats(moments="k"): the terminal law's sigma * sqrt(t) is 0.2.
    assert!((gbm.kurtosis().unwrap() - 0.678_365_777_175_437_2).abs() < 1e-12);
  }

  #[test]
  fn a_degenerate_terminal_law_has_no_closed_forms() {
    let gbm = Gbm::new(0.05_f64, 0.0, 10, 100.0, Some(1.0), Unseeded);
    assert!(gbm.pdf(100.0).is_none());
    assert!(gbm.cdf(100.0).is_none());
    assert!(gbm.quantile(0.5).is_none());
    assert!(gbm.mean().is_none());
    assert!(gbm.mode().is_none());
    assert!(gbm.median().is_none());
    assert!(gbm.variance().is_none());
    assert!(gbm.skewness().is_none());
    assert!(gbm.kurtosis().is_none());
    assert!(gbm.entropy().is_none());
    assert!(gbm.moment_generating_function(0.0).is_none());
    assert!(gbm.characteristic_function(0.0).is_none());
  }

  #[test]
  fn the_path_starts_at_x0() {
    assert_eq!(
      Gbm::<f64>::new(0.05, 0.2, 64, 100.0, None, Unseeded).sample()[0],
      100.0
    );
  }

  #[test]
  fn valid_terminal_distribution_is_positive_and_finite() {
    let gbm = Gbm::new(0.05_f64, 0.2, 10, 100.0, Some(1.0), Unseeded);

    assert!(gbm.pdf(100.0).unwrap().is_finite());
    assert!(gbm.pdf(100.0).unwrap() > 0.0);
    assert!(gbm.cdf(100.0).unwrap() > 0.0);
    assert!(gbm.cdf(100.0).unwrap() < 1.0);
    assert!(gbm.mean().unwrap() > 0.0);
  }
}

py_process_1d!(PyGbm, Gbm,
  sig: (mu, sigma, n, x0, t=None, seed=None, dtype=None),
  params: (mu: f64, sigma: f64, n: usize, x0: f64, t: Option<f64>),
  device
);
