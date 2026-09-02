//! Variance-gamma maximum likelihood in $(\log\sigma, \log\nu, \theta, \mu)$
//! through the Bessel-form density, with a profile pass over the smooth
//! parameters once the cusp in $\mu$ has been located.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::variance_gamma::SimdVarianceGamma;

use super::minimise;
use super::negative_log_likelihood;
use super::sample;
use crate::evt::gpd_fit::information_inverse;
use crate::traits::FloatExt;

/// Variance-gamma maximum-likelihood fit.
#[derive(Debug, Clone)]
pub struct VarianceGammaFit {
  /// Volatility $\hat\sigma$.
  pub sigma: f64,
  /// Gamma-clock variance rate $\hat\nu$.
  pub nu: f64,
  /// Drift $\hat\theta$.
  pub theta: f64,
  /// Location $\hat\mu$.
  pub mu: f64,
  /// Standard errors of `[sigma, nu, theta, mu]`.
  pub std_errors: Array1<f64>,
  /// Inverse observed information for `[sigma, nu, theta, mu]`.
  pub covariance: Array2<f64>,
  /// Maximised log-likelihood.
  pub log_likelihood: f64,
  /// $8 - 2\ell$.
  pub aic: f64,
  /// $4\log n - 2\ell$.
  pub bic: f64,
  /// Number of observations.
  pub nobs: usize,
  /// Simplex iterations.
  pub iterations: usize,
  /// Whether the simplex met its tolerance.
  pub converged: bool,
}

fn log_density(x: &[f64], sigma: f64, nu: f64, theta: f64, mu: f64) -> f64 {
  if !(sigma > 0.0 && sigma.is_finite() && nu > 0.0 && nu.is_finite() && theta.is_finite()) {
    return 1e300;
  }
  let d = SimdVarianceGamma::<f64>::new(sigma, nu, theta, mu, &Deterministic::new(0));
  negative_log_likelihood(x, |v| d.pdf(v).ln())
}

/// Variance-gamma maximum-likelihood fit of `data`.
///
/// # Panics
///
/// If there are fewer than 10 observations, any is non-finite, or they are
/// constant.
pub fn variance_gamma_fit<T: FloatExt>(data: ArrayView1<T>) -> VarianceGammaFit {
  let (x, mean, sd) = sample(data, 10);
  let objective = |p: &[f64]| log_density(&x, p[0].exp(), p[1].exp(), p[2], p[3]);
  let mut starts = Vec::new();
  for nu0 in [0.3_f64, 1.0] {
    for theta0 in [-0.2_f64, 0.0, 0.2] {
      starts.push(vec![sd.ln(), nu0.ln(), theta0 * sd, mean]);
    }
  }
  let (theta, iterations, converged) = minimise(&starts, objective);
  // The VG log-density has a cusp at x = μ, so the likelihood's maximum in
  // μ sits on a data point where the joint simplex stalls; with μ pinned
  // there the remaining parameters are smooth and a second simplex pass
  // recovers the last digits.
  let mu = theta[3];
  let profile = |p: &[f64]| log_density(&x, p[0].exp(), p[1].exp(), p[2], mu);
  let (smooth, iters_profile, converged_profile) = minimise(&[theta[..3].to_vec()], profile);
  let iterations = iterations + iters_profile;
  let converged = converged && converged_profile;
  let natural = [smooth[0].exp(), smooth[1].exp(), smooth[2], mu];
  let log_likelihood = -log_density(&x, natural[0], natural[1], natural[2], natural[3]);
  let nll = |p: &[f64]| log_density(&x, p[0], p[1], p[2], p[3]);
  let covariance = information_inverse(&nll, &natural, &[natural[0], natural[1], sd, sd]);
  let std_errors = Array1::from_iter((0..4).map(|i| covariance[[i, i]].sqrt()));
  let n = x.len() as f64;
  VarianceGammaFit {
    sigma: natural[0],
    nu: natural[1],
    theta: natural[2],
    mu: natural[3],
    std_errors,
    covariance,
    log_likelihood,
    aic: 8.0 - 2.0 * log_likelihood,
    bic: 4.0 * n.ln() - 2.0 * log_likelihood,
    nobs: x.len(),
    iterations,
    converged,
  }
}
