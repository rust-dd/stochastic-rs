//! GEV maximum likelihood for block maxima and the return-level map.
//!
//! For block maxima $z_1, \ldots, z_n$ the GEV log-likelihood is
//!
//! $$
//! \ell(\mu, \sigma, \xi) = -n\log\sigma - \Bigl(1 + \frac1\xi\Bigr)\sum_{i=1}^{n}\log t_i - \sum_{i=1}^{n} t_i^{-1/\xi},\qquad
//! t_i = 1 + \xi\,\frac{z_i - \mu}{\sigma} > 0
//! $$
//!
//! (Coles 2001, eq. 3.7; the Gumbel limit
//! $-n\log\sigma - \sum_i w_i - \sum_i e^{-w_i}$ with $w_i = (z_i - \mu)/\sigma$
//! at $\xi = 0$, eq. 3.9). The return level $z_m$ — the level exceeded on
//! average once every $m$ blocks — is the $1 - 1/m$ quantile (eq. 3.4):
//!
//! $$
//! z_m = \hat\mu - \frac{\hat\sigma}{\hat\xi}\Bigl[1 - \bigl(-\log(1 - 1/m)\bigr)^{-\hat\xi}\Bigr].
//! $$
//!
//! Standard errors are the inverse observed information from a
//! central-difference Hessian at the optimum.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use super::gpd_fit::information_inverse;
use crate::optim::nelder_mead_vec;
use crate::traits::FloatExt;

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
  /// Inverse observed information for `[mu, sigma, xi]`.
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
  /// Whether the simplex met its tolerance.
  pub converged: bool,
}

impl GevFit {
  /// Return level $z_m$ for a return `period` of $m > 1$ blocks.
  pub fn return_level(&self, period: f64) -> f64 {
    assert!(period > 1.0, "period must exceed 1 block, got {period}");
    let yp = -(1.0 - 1.0 / period).ln();
    if self.xi.abs() < 1e-12 {
      self.mu - self.sigma * yp.ln()
    } else {
      self.mu - self.sigma / self.xi * (1.0 - yp.powf(-self.xi))
    }
  }
}

/// Negative GEV log-likelihood; `1e300` outside the feasible region.
fn negative_log_likelihood(z: &[f64], mu: f64, sigma: f64, xi: f64) -> f64 {
  if !(sigma > 0.0 && sigma.is_finite()) {
    return 1e300;
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
      return 1e300;
    }
    let log_t = w.ln_1p();
    log_terms += log_t;
    power_terms += (-log_t / xi).exp();
  }
  n * sigma.ln() + (1.0 / xi + 1.0) * log_terms + power_terms
}

/// GEV maximum-likelihood fit to block maxima.
///
/// Returns a NaN covariance, and NaN standard errors with it, when the
/// observed-information Hessian at the optimum is singular.
///
/// # Panics
///
/// If there are fewer than 10 maxima, any is non-finite, or they are
/// constant.
pub fn gev_fit<T: FloatExt>(maxima: ArrayView1<T>) -> GevFit {
  let z: Vec<f64> = maxima
    .iter()
    .map(|x| x.to_f64().unwrap_or(f64::NAN))
    .collect();
  let n = z.len();
  assert!(n >= 10, "need at least 10 block maxima, got {n}");
  assert!(
    z.iter().all(|v| v.is_finite()),
    "block maxima must be finite"
  );
  let mean = z.iter().sum::<f64>() / n as f64;
  let sd = (z.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
  assert!(sd > 0.0, "block maxima must not be constant");

  let objective = |p: &[f64]| negative_log_likelihood(&z, p[0], p[1].exp(), p[2]);
  let sigma0 = 6.0_f64.sqrt() * sd / std::f64::consts::PI;
  let mu0 = mean - EULER_MASCHERONI * sigma0;
  let start = [-0.2, 0.0, 0.1, 0.3]
    .into_iter()
    .map(|xi0| [mu0, sigma0.ln(), xi0])
    .min_by(|a, b| objective(a).partial_cmp(&objective(b)).unwrap())
    .unwrap();
  let (theta, iters, converged) = nelder_mead_vec(&start, 5_000, objective);
  let (theta, iters_restart, converged_restart) = nelder_mead_vec(&theta, 5_000, objective);
  let (mu, sigma, xi) = (theta[0], theta[1].exp(), theta[2]);
  let log_likelihood = -negative_log_likelihood(&z, mu, sigma, xi);

  let nll = |p: &[f64]| negative_log_likelihood(&z, p[0], p[1], p[2]);
  let covariance = information_inverse(&nll, &[mu, sigma, xi], &[sd, sd, 1.0]);
  let std_errors = Array1::from_iter((0..3).map(|i| covariance[[i, i]].sqrt()));
  GevFit {
    mu,
    sigma,
    xi,
    std_errors,
    covariance,
    log_likelihood,
    aic: 6.0 - 2.0 * log_likelihood,
    bic: 3.0 * (n as f64).ln() - 2.0 * log_likelihood,
    nobs: n,
    iterations: iters + iters_restart,
    converged: converged || converged_restart,
  }
}

/// Maxima of consecutive blocks of `block_size` observations; a trailing
/// partial block is dropped.
///
/// # Panics
///
/// If `block_size` is zero.
pub fn block_maxima<T: FloatExt>(data: ArrayView1<T>, block_size: usize) -> Array1<f64> {
  assert!(block_size >= 1, "block_size must be at least 1");
  let x: Vec<f64> = data
    .iter()
    .map(|v| v.to_f64().unwrap_or(f64::NAN))
    .collect();
  x.chunks_exact(block_size)
    .map(|block| block.iter().copied().fold(f64::NEG_INFINITY, f64::max))
    .collect()
}
