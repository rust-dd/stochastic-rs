//! Hansen skew-t maximum likelihood of `x = μ + σ z`, in `(μ, ln σ, ln(η − 2), atanh λ)`.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use stochastic_rs_distributions::skew_t::SimdSkewT;
use stochastic_rs_distributions::traits::DistributionExt;

use super::Sample;
use super::minimise;
use super::negative_log_likelihood;
use super::sample;
use crate::StatsError;
use crate::evt::gpd::fit_in_data_units;
use crate::evt::gpd::information_inverse;
use crate::traits::FloatExt;

/// Hansen skew-t maximum-likelihood fit with location and scale.
#[derive(Debug, Clone)]
pub struct SkewTFit {
  /// Location $\hat\mu$.
  pub mu: f64,
  /// Scale $\hat\sigma$ (the standardised density's unit).
  pub sigma: f64,
  /// Degrees of freedom $\hat\eta > 2$.
  pub eta: f64,
  /// Skew $\hat\lambda \in (-1, 1)$.
  pub lambda: f64,
  /// Standard errors of `[mu, sigma, eta, lambda]`.
  pub std_errors: Array1<f64>,
  /// Inverse observed information for `[mu, sigma, eta, lambda]`, `±∞` past the double range.
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

fn log_density(x: &[f64], mu: f64, sigma: f64, eta: f64, lambda: f64) -> f64 {
  if !(sigma > 0.0 && sigma.is_finite() && eta > 2.0 && eta.is_finite() && lambda.abs() < 1.0) {
    return 1e300;
  }
  let d = SimdSkewT::<f64>::new(eta, lambda);
  let log_sigma = sigma.ln();
  negative_log_likelihood(x, |v| {
    d.pdf((v - mu) / sigma)
      .expect("SimdSkewT has a closed-form pdf")
      .ln()
      - log_sigma
  })
}

/// Hansen skew-t maximum likelihood of `data` at unit scale; `Err` below 10 observations, on a
/// non-finite one, a constant sample or an estimate past the double range.
pub fn fit_skew_t<T: FloatExt>(data: ArrayView1<T>) -> Result<SkewTFit, StatsError> {
  let Sample { x, mean, sd, unit } = sample(data, 10)?;
  let objective = |p: &[f64]| log_density(&x, p[0], p[1].exp(), 2.0 + p[2].exp(), p[3].tanh());
  let mut starts = Vec::new();
  for eta0 in [4.0_f64, 8.0, 20.0] {
    for lambda0 in [-0.3_f64, 0.0, 0.3] {
      starts.push(vec![mean, sd.ln(), (eta0 - 2.0).ln(), lambda0.atanh()]);
    }
  }
  let (theta, iterations, converged) = minimise(&starts, objective);
  let natural = [
    theta[0],
    theta[1].exp(),
    2.0 + theta[2].exp(),
    theta[3].tanh(),
  ];
  let log_likelihood = -log_density(&x, natural[0], natural[1], natural[2], natural[3]);
  let nll = |p: &[f64]| log_density(&x, p[0], p[1], p[2], p[3]);
  let covariance = information_inverse(&nll, &natural, &[sd, natural[1], natural[2], 1.0]);
  let fit = fit_in_data_units(
    natural,
    &covariance,
    log_likelihood,
    x.len(),
    unit,
    [1, 1, 0, 0],
  )?;
  let [mu, sigma, eta, lambda] = fit.params;
  let n = x.len() as f64;
  Ok(SkewTFit {
    mu,
    sigma,
    eta,
    lambda,
    std_errors: fit.std_errors,
    covariance: fit.covariance,
    log_likelihood: fit.log_likelihood,
    aic: 8.0 - 2.0 * fit.log_likelihood,
    bic: 4.0 * n.ln() - 2.0 * fit.log_likelihood,
    nobs: x.len(),
    iterations,
    converged,
  })
}
