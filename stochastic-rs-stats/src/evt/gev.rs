//! GEV maximum likelihood for block maxima, and the return level exceeded once in `m` blocks.
//!
//! $$
//! \ell(\mu, \sigma, \xi) = -n\log\sigma - \Bigl(1 + \frac1\xi\Bigr)\sum_{i=1}^{n}\log t_i -
//! \sum_{i=1}^{n} t_i^{-1/\xi},\qquad
//! t_i = 1 + \xi\,\frac{z_i - \mu}{\sigma} > 0
//! $$
//!
//! $$
//! \ell(\mu, \sigma, 0) = -n\log\sigma - \sum_i w_i - \sum_i e^{-w_i},\qquad
//! w_i = \frac{z_i - \mu}{\sigma}
//! $$
//!
//! $$
//! z_m = \hat\mu - \frac{\hat\sigma}{\hat\xi}\Bigl[1 - \bigl(-\log(1 - 1/m)\bigr)^{-\hat\xi}\Bigr].
//! $$
//!
//! Standard errors are the inverse observed information, a central-difference Hessian.
//!
//! Reference: Coles (2001), eq. 3.7, eq. 3.9 at `ξ = 0`, eq. 3.4, DOI: 10.1007/978-1-4471-3675-0

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use super::gpd::fit_in_data_units;
use super::gpd::information_inverse;
use crate::StatsError;
use crate::input::reject_constant;
use crate::input::validate;
use crate::optim::search::Coordinate;
use crate::optim::search::NO_VALUE;
use crate::optim::search::by_differences;
use crate::optim::search::minimise;
use crate::stationarity::common::unit_scaled;
use crate::traits::FloatExt;
use crate::traits::RealExt;

/// Euler–Mascheroni constant, the Gumbel mean offset used for the
/// starting values.
const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

/// GEV maximum-likelihood fit to block maxima.
#[derive(Debug, Clone)]
pub struct GevFit {
  /// Location $\hat\mu$.
  pub mu: f64,
  /// Scale $\hat\sigma$.
  pub sigma: f64,
  /// Shape $\hat\xi$.
  pub xi: f64,
  /// Standard errors of `[mu, sigma, xi]`.
  pub std_errors: Array1<f64>,
  /// Inverse observed information for `[mu, sigma, xi]`, `±∞` past the double range.
  pub covariance: Array2<f64>,
  /// Maximised log-likelihood.
  pub log_likelihood: f64,
  /// $6 - 2\ell$.
  pub aic: f64,
  /// $3\log n - 2\ell$.
  pub bic: f64,
  /// Number of block maxima.
  pub nobs: usize,
  /// Simplex iterations.
  pub iterations: usize,
  /// No step along one parameter of the search gains `1e-7` in log-likelihood, and no limit of
  /// the family fits as well: stationary along each, not global.
  pub converged: bool,
  /// `converged`, or why not: `not stationary`, `stopped at the bound of <name>`,
  /// `<name> is not identified`, `iteration cap reached`, …
  pub message: String,
}

impl GevFit {
  /// Return level $z_m$ for a return `period` of $m$ blocks; NaN unless `period > 1`.
  pub fn return_level(&self, period: f64) -> f64 {
    if period.is_nan() || period <= 1.0 {
      return f64::NAN;
    }
    let yp = -(1.0 - 1.0 / period).ln();
    if self.xi.abs() < 1e-12 {
      self.mu - self.sigma * yp.ln()
    } else {
      self.mu - self.sigma / self.xi * (1.0 - yp.powf(-self.xi))
    }
  }
}

/// Negative GEV log-likelihood; [`NO_VALUE`] outside the feasible region.
fn negative_log_likelihood(z: &[f64], mu: f64, sigma: f64, xi: f64) -> f64 {
  if !(sigma > 0.0 && sigma.is_finite()) {
    return NO_VALUE;
  }
  let n = z.len() as f64;
  if xi.abs() < 1e-12 {
    let mut acc = 0.0;
    for &v in z {
      let w = (v - mu) / sigma;
      acc += w + (-w).exp();
    }
    return n * sigma.ln() + acc;
  }
  let mut log_terms = 0.0;
  let mut power_terms = 0.0;
  for &v in z {
    let w = xi * (v - mu) / sigma;
    if w <= -1.0 {
      return NO_VALUE;
    }
    let log_t = w.ln_1p();
    log_terms += log_t;
    power_terms += (-log_t / xi).exp();
  }
  n * sigma.ln() + (1.0 / xi + 1.0) * log_terms + power_terms
}

/// GEV maximum likelihood of block maxima at unit scale; `Err` below 10, on a non-finite maximum, a
/// constant sample or an estimate past the double range; NaN errors at a singular Hessian.
pub fn fit_gev<T: FloatExt>(maxima: ArrayView1<T>) -> Result<GevFit, StatsError> {
  let z = maxima
    .iter()
    .map(|x| x.to_f64().unwrap())
    .collect::<Vec<f64>>();
  validate(&z, 10)?;
  reject_constant(&z, "constant block maxima")?;
  let (z, unit) = unit_scaled(z);
  let n = z.len();
  let mean = z.iter().sum::<f64>() / n as f64;
  let sd = (z.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n as f64 - 1.0)).sqrt();

  let objective = |p: &[f64]| negative_log_likelihood(&z, p[0], p[1].exp(), p[2]);
  let sigma0 = 6.0_f64.sqrt() * sd / std::f64::consts::PI;
  let mu0 = mean - EULER_MASCHERONI * sigma0;
  let starts = [-0.2, 0.0, 0.1, 0.3].map(|xi0| vec![mu0, sigma0.ln(), xi0]);
  let coordinates = [
    Coordinate::location(mean, sd),
    Coordinate::log(sd.ln()),
    Coordinate::free(),
  ];
  let found = minimise(objective, by_differences, &starts, &coordinates, n);
  let theta = &found.at;
  let (mu, sigma, xi) = (theta[0], theta[1].exp(), theta[2]);
  let log_likelihood = -negative_log_likelihood(&z, mu, sigma, xi);

  let nll = |p: &[f64]| negative_log_likelihood(&z, p[0], p[1], p[2]);
  let covariance = information_inverse(&nll, &[mu, sigma, xi], &[sd, sd, 1.0]);
  let fit = fit_in_data_units(
    [mu, sigma, xi],
    &covariance,
    log_likelihood,
    n,
    unit,
    [1, 1, 0],
  )?;
  let [mu, sigma, xi] = fit.params;
  Ok(GevFit {
    mu,
    sigma,
    xi,
    std_errors: fit.std_errors,
    covariance: fit.covariance,
    log_likelihood: fit.log_likelihood,
    aic: 6.0 - 2.0 * fit.log_likelihood,
    bic: 3.0 * (n as f64).ln() - 2.0 * fit.log_likelihood,
    nobs: n,
    iterations: found.iterations,
    converged: found.stop.converged(),
    message: found.stop.message(&["mu", "sigma", "xi"]),
  })
}

/// Maxima of consecutive blocks of `block_size` observations, a trailing partial block dropped;
/// panics unless `block_size >= 1`.
pub fn block_maxima<T: FloatExt>(data: ArrayView1<T>, block_size: usize) -> Array1<f64> {
  assert!(
    block_size >= 1,
    "block_size must satisfy `block_size >= 1`, got block_size = {block_size:?}"
  );
  let x: Vec<f64> = data.iter().map(|v| v.to_f64().unwrap()).collect();
  x.chunks_exact(block_size)
    .map(|block| f64::max_or_nan_fold(block.iter().copied()))
    .collect()
}
