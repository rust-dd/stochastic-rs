//! Johnson SU maximum likelihood in $(\gamma, \log\delta, \xi, \log\lambda)$.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::johnson_su::SimdJohnsonSu;
use stochastic_rs_distributions::traits::DistributionExt;

use super::minimise;
use super::negative_log_likelihood;
use super::sample;
use crate::evt::gpd_fit::information_inverse;
use crate::traits::FloatExt;

/// Johnson SU maximum-likelihood fit.
#[derive(Debug, Clone)]
pub struct JohnsonSuFit {
  /// Skew shape $\hat\gamma$.
  pub gamma: f64,
  /// Tail shape $\hat\delta$.
  pub delta: f64,
  /// Location $\hat\xi$.
  pub xi: f64,
  /// Scale $\hat\lambda$.
  pub lambda: f64,
  /// Standard errors of `[gamma, delta, xi, lambda]`.
  pub std_errors: Array1<f64>,
  /// Inverse observed information for `[gamma, delta, xi, lambda]`.
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

fn log_density(x: &[f64], gamma: f64, delta: f64, xi: f64, lambda: f64) -> f64 {
  if !(delta > 0.0 && lambda > 0.0 && delta.is_finite() && lambda.is_finite()) {
    return 1e300;
  }
  let d = SimdJohnsonSu::<f64>::new(gamma, delta, xi, lambda, &Deterministic::new(0));
  negative_log_likelihood(x, |v| d.pdf(v).ln())
}

/// Johnson SU maximum-likelihood fit of `data`.
///
/// # Panics
///
/// If there are fewer than 10 observations, any is non-finite, or they are
/// constant.
pub fn johnson_su_fit<T: FloatExt>(data: ArrayView1<T>) -> JohnsonSuFit {
  let (x, mean, sd) = sample(data, 10);
  let objective = |p: &[f64]| log_density(&x, p[0], p[1].exp(), p[2], p[3].exp());
  let mut starts = Vec::new();
  for gamma0 in [-1.0, 0.0, 1.0] {
    for delta0 in [0.5_f64, 1.0, 2.0] {
      starts.push(vec![gamma0, delta0.ln(), mean, sd.ln()]);
    }
  }
  let (theta, iterations, converged) = minimise(&starts, objective);
  let natural = [theta[0], theta[1].exp(), theta[2], theta[3].exp()];
  let log_likelihood = -log_density(&x, natural[0], natural[1], natural[2], natural[3]);
  let nll = |p: &[f64]| log_density(&x, p[0], p[1], p[2], p[3]);
  let covariance = information_inverse(&nll, &natural, &[1.0, natural[1], sd, natural[3]]);
  let std_errors = Array1::from_iter((0..4).map(|i| covariance[[i, i]].sqrt()));
  let n = x.len() as f64;
  JohnsonSuFit {
    gamma: natural[0],
    delta: natural[1],
    xi: natural[2],
    lambda: natural[3],
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
