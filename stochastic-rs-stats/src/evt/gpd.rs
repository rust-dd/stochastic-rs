//! Generalised-Pareto maximum likelihood for threshold excesses and the
//! peaks-over-threshold tail model built on it.
//!
//! For excesses $y_i = x_i - u > 0$ the GPD log-likelihood is
//!
//! $$
//! \ell(\sigma, \xi) = -n\log\sigma - \Bigl(1 + \frac1\xi\Bigr)\sum_{i=1}^{n}\log\Bigl(1 + \xi\frac{y_i}{\sigma}\Bigr)
//! $$
//!
//! ($-n\log\sigma - \sum_i y_i/\sigma$ at $\xi = 0$), maximised over
//! $\sigma > 0$ with $1 + \xi y_i/\sigma > 0$ (Coles 2001, eq. 4.10). With
//! $n_u$ of $n$ observations above $u$, the tail estimate
//! $\hat F(x) = 1 - (n_u/n)\bigl(1 + \hat\xi(x - u)/\hat\sigma\bigr)^{-1/\hat\xi}$
//! inverts to the Value-at-Risk and expected shortfall of McNeil, Frey and
//! Embrechts (2015), eq. 5.18 and 5.20:
//!
//! $$
//! \widehat{\mathrm{VaR}}_p = u + \frac{\hat\sigma}{\hat\xi}\Bigl[\Bigl(\frac{1-p}{n_u/n}\Bigr)^{-\hat\xi} - 1\Bigr],\qquad
//! \widehat{\mathrm{ES}}_p = \frac{\widehat{\mathrm{VaR}}_p}{1-\hat\xi} + \frac{\hat\sigma - \hat\xi u}{1-\hat\xi}.
//! $$
//!
//! Standard errors are the inverse observed information from a
//! central-difference Hessian at the optimum.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use stochastic_rs_distributions::linalg::inverse;

use crate::StatsError;
use crate::input::reject_constant;
use crate::input::validate;
use crate::input::validate_domain;
use crate::optim::best_start;
use crate::optim::nelder_mead_vec;
use crate::traits::FloatExt;

/// GPD maximum-likelihood fit to threshold excesses.
#[derive(Debug, Clone)]
pub struct GpdFit {
  /// Scale $\hat\sigma$.
  pub sigma: f64,
  /// Shape $\hat\xi$.
  pub xi: f64,
  /// Standard errors of `[sigma, xi]`.
  pub std_errors: Array1<f64>,
  /// Inverse observed information for `[sigma, xi]`.
  pub covariance: Array2<f64>,
  /// Maximised log-likelihood.
  pub log_likelihood: f64,
  /// $4 - 2\ell$.
  pub aic: f64,
  /// $2\log n - 2\ell$.
  pub bic: f64,
  /// Number of excesses.
  pub nobs: usize,
  /// Simplex iterations.
  pub iterations: usize,
  /// Whether the simplex met its tolerance.
  pub converged: bool,
}

/// Peaks-over-threshold tail model: a [`GpdFit`] on the excesses over
/// `threshold` plus the exceedance rate that anchors the tail quantiles.
#[derive(Debug, Clone)]
pub struct PotFit {
  /// Threshold $u$.
  pub threshold: f64,
  /// GPD fit to the excesses $x - u$.
  pub gpd: GpdFit,
  /// Number of observations above the threshold, $n_u$.
  pub n_exceedances: usize,
  /// Sample size $n$.
  pub nobs: usize,
  /// $n_u / n$.
  pub exceedance_rate: f64,
}

impl PotFit {
  /// Tail quantile $\widehat{\mathrm{VaR}}_p$ (McNeil–Frey–Embrechts eq. 5.18), meaningful for
  /// $p > 1 - n_u/n$; NaN unless `0 < p < 1`.
  pub fn quantile(&self, p: f64) -> f64 {
    if !(p > 0.0 && p < 1.0) {
      return f64::NAN;
    }
    let ratio = (1.0 - p) / self.exceedance_rate;
    let (sigma, xi) = (self.gpd.sigma, self.gpd.xi);
    if xi.abs() < 1e-12 {
      self.threshold - sigma * ratio.ln()
    } else {
      self.threshold + sigma / xi * (ratio.powf(-xi) - 1.0)
    }
  }

  /// $\widehat{\mathrm{ES}}_p$ (eq. 5.20); `+∞` when $\hat\xi \ge 1$; NaN unless `0 < p < 1`.
  pub fn expected_shortfall(&self, p: f64) -> f64 {
    if !(p > 0.0 && p < 1.0) {
      return f64::NAN;
    }
    let xi = self.gpd.xi;
    if xi >= 1.0 {
      return f64::INFINITY;
    }
    let var = self.quantile(p);
    var / (1.0 - xi) + (self.gpd.sigma - xi * self.threshold) / (1.0 - xi)
  }
}

/// Negative GPD log-likelihood; `1e300` outside the feasible region.
fn negative_log_likelihood(y: &[f64], sigma: f64, xi: f64) -> f64 {
  if !(sigma > 0.0 && sigma.is_finite()) {
    return 1e300;
  }
  let n = y.len() as f64;
  if xi.abs() < 1e-12 {
    return n * sigma.ln() + y.iter().sum::<f64>() / sigma;
  }
  let mut acc = 0.0;
  for &v in y {
    let w = xi * v / sigma;
    if w <= -1.0 {
      return 1e300;
    }
    acc += w.ln_1p();
  }
  n * sigma.ln() + (1.0 / xi + 1.0) * acc
}

/// GPD maximum-likelihood fit to excesses; `Err` below 10, on a negative or non-finite excess or a
/// constant sample. A singular Hessian at the optimum gives NaN standard errors.
pub fn fit_gpd<T: FloatExt>(exceedances: ArrayView1<T>) -> Result<GpdFit, StatsError> {
  let y = exceedances
    .iter()
    .map(|x| x.to_f64().unwrap_or(f64::NAN))
    .collect::<Vec<f64>>();
  validate_domain(&y, 10, "finite and >= 0", |v| v >= 0.0)?;
  reject_constant(&y, "constant excesses")?;
  let n = y.len();
  let mean = y.iter().sum::<f64>() / n as f64;
  let var = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
  if mean == 0.0 || var == 0.0 {
    return Err(StatsError::DegenerateData("constant excesses"));
  }

  let objective = |p: &[f64]| negative_log_likelihood(&y, p[0].exp(), p[1]);
  let ratio = mean * mean / var;
  let mut starts = vec![[
    (0.5 * mean * (1.0 + ratio)).ln(),
    (0.5 * (1.0 - ratio)).clamp(-0.45, 0.9),
  ]];
  for xi0 in [-0.2, 0.0, 0.2, 0.5] {
    starts.push([(mean * (1.0 - xi0)).ln(), xi0]);
  }
  let start = best_start(starts, objective).expect("a non-empty starting grid");
  let (theta, iters, converged) = nelder_mead_vec(&start, 5_000, objective);
  let (theta, iters_restart, converged_restart) = nelder_mead_vec(&theta, 5_000, objective);
  let sigma = theta[0].exp();
  let xi = theta[1];
  let log_likelihood = -negative_log_likelihood(&y, sigma, xi);

  let nll = |p: &[f64]| negative_log_likelihood(&y, p[0], p[1]);
  let covariance = information_inverse(&nll, &[sigma, xi], &[mean, 1.0]);
  let std_errors = Array1::from_iter((0..2).map(|i| covariance[[i, i]].sqrt()));
  Ok(GpdFit {
    sigma,
    xi,
    std_errors,
    covariance,
    log_likelihood,
    aic: 4.0 - 2.0 * log_likelihood,
    bic: 2.0 * (n as f64).ln() - 2.0 * log_likelihood,
    nobs: n,
    iterations: iters + iters_restart,
    converged: converged || converged_restart,
  })
}

/// Probability-weighted-moment estimate of the GPD (Hosking–Wallis 1987).
#[derive(Debug, Clone)]
pub struct GpdPwmFit {
  /// Scale $\hat\sigma$.
  pub sigma: f64,
  /// Shape $\hat\xi$.
  pub xi: f64,
  /// $a_0$, the sample mean.
  pub a0: f64,
  /// $a_1$, the estimate of $\mathbb E[X(1 - F(X))]$.
  pub a1: f64,
  /// Number of excesses.
  pub nobs: usize,
}

/// Hosking–Wallis probability-weighted-moment GPD estimates over the order statistics `x(i)`; `Err`
/// below 2, on a negative or non-finite excess or a constant sample.
/// $$
/// a_0 = \bar x,\qquad a_1 = \frac1n\sum_{i=1}^{n}\frac{n - i}{n - 1}\,x_{(i)},\qquad
/// \hat\xi = 2 - \frac{a_0}{a_0 - 2a_1},\qquad \hat\sigma = \frac{2a_0a_1}{a_0 - 2a_1}
/// $$
/// Hosking & Wallis (1987), Technometrics 29(3), 339-349. DOI: 10.1080/00401706.1987.10488243
pub fn fit_gpd_pwm<T: FloatExt>(exceedances: ArrayView1<T>) -> Result<GpdPwmFit, StatsError> {
  let mut y = exceedances
    .iter()
    .map(|x| x.to_f64().unwrap_or(f64::NAN))
    .collect::<Vec<f64>>();
  validate_domain(&y, 2, "finite and >= 0", |v| v >= 0.0)?;
  reject_constant(&y, "constant excesses")?;
  let n = y.len();
  y.sort_by(f64::total_cmp);
  let nf = n as f64;
  let a0 = y.iter().sum::<f64>() / nf;
  let a1 = y
    .iter()
    .enumerate()
    .map(|(i, v)| v * (nf - (i + 1) as f64) / (nf - 1.0))
    .sum::<f64>()
    / nf;
  let denominator = a0 - 2.0 * a1;
  Ok(GpdPwmFit {
    sigma: 2.0 * a0 * a1 / denominator,
    xi: 2.0 - a0 / denominator,
    a0,
    a1,
    nobs: n,
  })
}

/// Peaks-over-threshold fit of losses `data` above `threshold`; `Err` on a non-finite observation,
/// below 10 exceedances or as [`fit_gpd`] on them, and panics unless `threshold` is finite.
pub fn fit_pot<T: FloatExt>(data: ArrayView1<T>, threshold: f64) -> Result<PotFit, StatsError> {
  assert!(
    threshold.is_finite(),
    "threshold must satisfy `threshold.is_finite()`, got threshold = {threshold:?}"
  );
  let x = data
    .iter()
    .map(|v| v.to_f64().unwrap_or(f64::NAN))
    .collect::<Vec<f64>>();
  validate(&x, 0)?;
  let exceedances = x
    .iter()
    .filter(|v| **v > threshold)
    .map(|v| v - threshold)
    .collect::<Array1<f64>>();
  let n_exceedances = exceedances.len();
  if n_exceedances < 10 {
    return Err(StatsError::InsufficientData {
      needed: 10,
      got: n_exceedances,
    });
  }
  let gpd = fit_gpd(exceedances.view())?;
  Ok(PotFit {
    threshold,
    gpd,
    n_exceedances,
    nobs: data.len(),
    exceedance_rate: n_exceedances as f64 / data.len() as f64,
  })
}

/// Mean excess $e(u) = \mathbb E[X - u \mid X > u]$ at each threshold, the mean-residual-life plot
/// that picks one; NaN where nothing exceeds $u$, and at every $u$ when `data` holds a NaN.
pub fn mean_excess<T: FloatExt>(data: ArrayView1<T>, thresholds: ArrayView1<f64>) -> Array1<f64> {
  let x: Vec<f64> = data
    .iter()
    .map(|v| v.to_f64().unwrap_or(f64::NAN))
    .collect();
  if x.iter().any(|v| v.is_nan()) {
    return Array1::from_elem(thresholds.len(), f64::NAN);
  }
  thresholds
    .iter()
    .map(|&u| {
      let (sum, count) = x
        .iter()
        .filter(|v| **v > u)
        .fold((0.0, 0usize), |(s, c), v| (s + (v - u), c + 1));
      if count == 0 {
        f64::NAN
      } else {
        sum / count as f64
      }
    })
    .collect()
}

/// Inverse of the central-difference Hessian of a negative log-likelihood
/// at `theta`, with steps $10^{-4}\max(|\theta_i|, \text{scale}_i)$; NaN
/// entries when the Hessian is singular.
pub(crate) fn information_inverse<F: Fn(&[f64]) -> f64>(
  nll: &F,
  theta: &[f64],
  scales: &[f64],
) -> Array2<f64> {
  let k = theta.len();
  let steps: Vec<f64> = (0..k)
    .map(|i| 1e-4 * theta[i].abs().max(scales[i]))
    .collect();
  let shifted = |i: usize, si: f64, j: usize, sj: f64| {
    let mut v = theta.to_vec();
    v[i] += si;
    v[j] += sj;
    v
  };
  let f0 = nll(theta);
  let mut hessian = Array2::<f64>::zeros((k, k));
  for i in 0..k {
    let h = steps[i];
    hessian[[i, i]] =
      (nll(&shifted(i, h, i, 0.0)) - 2.0 * f0 + nll(&shifted(i, -h, i, 0.0))) / (h * h);
    for j in (i + 1)..k {
      let g = steps[j];
      let mixed =
        (nll(&shifted(i, h, j, g)) - nll(&shifted(i, h, j, -g)) - nll(&shifted(i, -h, j, g))
          + nll(&shifted(i, -h, j, -g)))
          / (4.0 * h * g);
      hessian[[i, j]] = mixed;
      hessian[[j, i]] = mixed;
    }
  }
  inverse(hessian.view()).unwrap_or_else(|| Array2::from_elem((k, k), f64::NAN))
}
