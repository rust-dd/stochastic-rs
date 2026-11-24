//! Johnson SU maximum likelihood in `(γ, ln δ, ξ, ln λ)`.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use stochastic_rs_distributions::johnson_su::SimdJohnsonSu;
use stochastic_rs_distributions::traits::DistributionExt;

use super::Sample;
use super::minimise;
use super::negative_log_likelihood;
use super::sample;
use super::words;
use crate::StatsError;
use crate::evt::gpd::fit_in_data_units;
use crate::evt::gpd::information_inverse;
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
  /// Inverse observed information for `[gamma, delta, xi, lambda]`, `±∞` past the double range.
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
  /// The simplex or its restart met a `1e-10` spread of its values; the L-BFGS polish is untested.
  pub converged: bool,
  /// `converged` or `iteration cap reached`.
  pub message: String,
}

fn log_density(x: &[f64], gamma: f64, delta: f64, xi: f64, lambda: f64) -> f64 {
  if !(delta > 0.0 && lambda > 0.0 && delta.is_finite() && lambda.is_finite()) {
    return 1e300;
  }
  let d = SimdJohnsonSu::<f64>::new(gamma, delta, xi, lambda);
  negative_log_likelihood(x, |v| {
    d.pdf(v).expect("SimdJohnsonSu has a closed-form pdf").ln()
  })
}

/// Johnson SU maximum likelihood of `data` at unit scale; `Err` below 10 observations, on a
/// non-finite one, a constant sample or an estimate past the double range.
pub fn fit_johnson_su<T: FloatExt>(data: ArrayView1<T>) -> Result<JohnsonSuFit, StatsError> {
  let Sample { x, mean, sd, unit } = sample(data, 10)?;
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
  let fit = fit_in_data_units(
    natural,
    &covariance,
    log_likelihood,
    x.len(),
    unit,
    [0, 0, 1, 1],
  )?;
  let [gamma, delta, xi, lambda] = fit.params;
  let n = x.len() as f64;
  Ok(JohnsonSuFit {
    gamma,
    delta,
    xi,
    lambda,
    std_errors: fit.std_errors,
    covariance: fit.covariance,
    log_likelihood: fit.log_likelihood,
    aic: 8.0 - 2.0 * fit.log_likelihood,
    bic: 4.0 * n.ln() - 2.0 * fit.log_likelihood,
    nobs: x.len(),
    iterations,
    converged,
    message: words(converged),
  })
}
