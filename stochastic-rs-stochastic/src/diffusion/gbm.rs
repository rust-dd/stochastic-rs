//! # Gbm
//!
//! $$
//! dS_t=\mu S_t\,dt+\sigma S_t\,dW_t,\quad S_0=s_0
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_distributions::special::norm_pdf;

use crate::buffer::array1_from_fill;
use crate::traits::DistributionExt;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct Gbm<T: FloatExt, S: SeedExt = Unseeded> {
  /// Drift / long-run mean-level parameter.
  pub mu: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Cached parameters of the terminal log-normal: ln S_T ∼ N(`ln_mu`, `ln_sigma`).
  ln_mu: f64,
  ln_sigma: f64,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Gbm<T, S> {
  pub fn new(mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    let x0_f64 = x0.unwrap_or(T::one()).to_f64().unwrap();
    let mu_f64 = mu.to_f64().unwrap();
    let sigma_f64 = sigma.to_f64().unwrap();
    let t_f64 = t.unwrap_or(T::one()).to_f64().unwrap();

    let mu_ln = x0_f64.ln() + (mu_f64 - 0.5 * sigma_f64 * sigma_f64) * t_f64;
    let sigma_ln = sigma_f64 * t_f64.sqrt();

    Self {
      mu,
      sigma,
      n,
      x0,
      t,
      ln_mu: mu_ln,
      ln_sigma: sigma_ln,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Gbm<T, S> {
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
      x0: self.x0.unwrap_or(T::one()),
      drift_scale: self.mu * dt,
      diff_scale: self.sigma,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`Gbm`] sampling state: precomputed Euler scales and the owned
/// Gaussian source, so a Monte-Carlo loop pays the `SimdNormal` setup once.
#[doc(hidden)]
pub struct GbmSampler<T: FloatExt> {
  n: usize,
  x0: T,
  drift_scale: T,
  diff_scale: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> GbmSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    // Default x0 = 1: a GBM started at 0 is absorbed at 0, so the marginal
    // convention of the type uses 1 (matches the constructor's `ln_mu`).
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.normal.fill_slice_fast(tail);
    let mut prev = self.x0;
    for z in tail.iter_mut() {
      let next = prev + self.drift_scale * prev + self.diff_scale * prev * *z;
      *z = next;
      prev = next;
    }
  }
}

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

  /// Malliavin derivative of the Gbm process
  ///
  /// The Malliavin derivative of the Gbm process is given by
  /// `D_r S_t = \sigma S_t \mathbf{1}_{0 \le r \le t}`.
  ///
  /// The Malliavin derivate of the Gbm shows the sensitivity of the stock price with respect to the Wiener process.
  pub fn malliavin(&self) -> [Array1<T>; 2] {
    let gbm = self.sample();
    let mut m = Array1::zeros(self.n);

    // reverse due the option pricing
    let s_t = *gbm.last().unwrap();
    for i in 0..self.n {
      m[i] = self.sigma * s_t;
    }

    [gbm, m]
  }
}

// Terminal distribution of the Gbm: S_T ∼ LogNormal(ln_mu, ln_sigma) where
// ln_mu = ln(S_0) + (μ − ½σ²)·T,   ln_sigma = σ·√T.
impl<T: FloatExt, S: SeedExt> DistributionExt for Gbm<T, S> {
  fn pdf(&self, x: f64) -> f64 {
    let Some((ln_mu, ln_sigma)) = self.terminal_lognormal_params() else {
      return 0.0;
    };
    if x <= 0.0 {
      return 0.0;
    }
    let z = (x.ln() - ln_mu) / ln_sigma;
    norm_pdf(z) / (ln_sigma * x)
  }

  fn cdf(&self, x: f64) -> f64 {
    let Some((ln_mu, ln_sigma)) = self.terminal_lognormal_params() else {
      return 0.0;
    };
    if x <= 0.0 {
      return 0.0;
    }
    norm_cdf((x.ln() - ln_mu) / ln_sigma)
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    let Some((ln_mu, ln_sigma)) = self.terminal_lognormal_params() else {
      return 0.0;
    };
    (ln_mu + ln_sigma * ndtri(p)).exp()
  }

  fn mean(&self) -> f64 {
    let Some((ln_mu, ln_sigma)) = self.terminal_lognormal_params() else {
      return 0.0;
    };
    (ln_mu + 0.5 * ln_sigma * ln_sigma).exp()
  }

  fn mode(&self) -> f64 {
    let Some((ln_mu, ln_sigma)) = self.terminal_lognormal_params() else {
      return 0.0;
    };
    (ln_mu - ln_sigma * ln_sigma).exp()
  }

  fn median(&self) -> f64 {
    let Some((ln_mu, _)) = self.terminal_lognormal_params() else {
      return 0.0;
    };
    ln_mu.exp()
  }

  fn variance(&self) -> f64 {
    let Some((ln_mu, ln_sigma)) = self.terminal_lognormal_params() else {
      return 0.0;
    };
    let s2 = ln_sigma * ln_sigma;
    (s2.exp() - 1.0) * (2.0 * ln_mu + s2).exp()
  }

  fn skewness(&self) -> f64 {
    let Some((_, ln_sigma)) = self.terminal_lognormal_params() else {
      return 0.0;
    };
    let s2 = ln_sigma * ln_sigma;
    (s2.exp() + 2.0) * (s2.exp() - 1.0).sqrt()
  }

  fn entropy(&self) -> f64 {
    let Some((ln_mu, ln_sigma)) = self.terminal_lognormal_params() else {
      return 0.0;
    };
    0.5 + 0.5 * (2.0 * std::f64::consts::PI * ln_sigma * ln_sigma).ln() + ln_mu
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn invalid_terminal_distribution_returns_zero_fallbacks() {
    let gbm = Gbm::new(0.05_f64, 0.0, 10, Some(100.0), Some(1.0), Unseeded);

    assert_eq!(gbm.pdf(100.0), 0.0);
    assert_eq!(gbm.cdf(100.0), 0.0);
    assert_eq!(gbm.inv_cdf(0.5), 0.0);
    assert_eq!(gbm.mean(), 0.0);
    assert_eq!(gbm.mode(), 0.0);
    assert_eq!(gbm.median(), 0.0);
    assert_eq!(gbm.variance(), 0.0);
    assert_eq!(gbm.skewness(), 0.0);
    assert_eq!(gbm.entropy(), 0.0);
  }

  #[test]
  fn valid_terminal_distribution_is_positive_and_finite() {
    let gbm = Gbm::new(0.05_f64, 0.2, 10, Some(100.0), Some(1.0), Unseeded);

    assert!(gbm.pdf(100.0).is_finite());
    assert!(gbm.pdf(100.0) > 0.0);
    assert!(gbm.cdf(100.0) > 0.0);
    assert!(gbm.cdf(100.0) < 1.0);
    assert!(gbm.mean() > 0.0);
  }
}

py_process_1d!(PyGbm, Gbm,
  sig: (mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
