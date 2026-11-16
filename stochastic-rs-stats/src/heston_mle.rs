//! # Mle
//!
//! $$
//! \hat\theta=\arg\max_\theta \sum_{i=1}^n \log f(x_i\mid\theta)
//! $$
//!
use ndarray::ArrayView1;

use crate::StatsError;
use crate::input::as_slice_or_copy;
use crate::input::reject_constant;
use crate::input::validate_domain;
use crate::input::validate_view_domain;

#[derive(Clone, Debug)]
pub struct HestonMleFit {
  /// Initial variance/volatility level.
  pub v0: f64,
  /// Mean-reversion speed parameter.
  pub kappa: f64,
  /// Long-run target level parameter.
  pub theta: f64,
  /// Volatility / diffusion scale parameter.
  pub sigma: f64,
  /// Correlation: the moment estimate (Wang et al. eq. 23) held in `[-1, 1]`.
  pub rho: f64,
}

/// Wang et al. (2018) closed-form Heston NMLE from `(S_t, V_t)` over unit time; `Err` on unequal
/// lengths, < 4 points, constant lagged `V` or a datum or estimate not finite and `> 0`; panics on
/// a non-finite `r`.
/// Reference: Wang et al. (2018), Sci. China Inf. Sci. 61, 042202. DOI: 10.1007/s11432-017-9215-8
pub fn fit_heston_nmle(
  s: ArrayView1<f64>,
  v: ArrayView1<f64>,
  r: f64,
) -> Result<HestonMleFit, StatsError> {
  assert_r(r);
  validate_series(s, v)?;
  nmle(s, v, r, 1.0 / (v.len() - 1) as f64)
}

/// [`fit_heston_nmle`] at sampling interval `delta` (Wang et al. eq. 19-21 and the rho moment
/// estimator); `Err` as it, and panics unless `r` is finite and `delta` finite and positive.
pub fn fit_heston_nmle_with_delta(
  s: ArrayView1<f64>,
  v: ArrayView1<f64>,
  r: f64,
  delta: f64,
) -> Result<HestonMleFit, StatsError> {
  assert_r(r);
  assert_delta(delta);
  validate_series(s, v)?;
  nmle(s, v, r, delta)
}

fn nmle(
  s: ArrayView1<f64>,
  v: ArrayView1<f64>,
  r: f64,
  delta: f64,
) -> Result<HestonMleFit, StatsError> {
  let nobs = v.len();
  let n_f = (nobs - 1) as f64;

  // Following Wang et al. (2018), Eq. (19)-(21):
  // P = kappa * theta - sigma^2 / 4, with closed-form estimators from sqrt-variance dynamics.
  let mut sum_inv_v_prev = 0.0;
  let mut sum_v_prev = 0.0;
  let mut sum_sqrt_ratio = 0.0;
  let mut sum_sqrt_prod = 0.0;

  for i in 1..nobs {
    let v_prev = v[i - 1];
    let v_curr = v[i];

    sum_inv_v_prev += 1.0 / v_prev;
    sum_v_prev += v_prev;
    sum_sqrt_ratio += (v_curr / v_prev).sqrt();
    sum_sqrt_prod += (v_curr * v_prev).sqrt();
  }

  let denom_inner = n_f - (sum_v_prev * sum_inv_v_prev) / n_f;
  let denom_p = 0.5 * delta * denom_inner;
  if (sum_v_prev * sum_inv_v_prev / (n_f * n_f) - 1.0).abs() <= DENOMINATOR {
    return Err(StatsError::Numerical {
      stage: "drift coefficient",
    });
  }
  let p_hat = (sum_sqrt_prod - (sum_v_prev * sum_sqrt_ratio) / n_f) / denom_p;

  let kappa_hat =
    (2.0 / delta) * (1.0 + 0.5 * delta * p_hat * (sum_inv_v_prev / n_f) - (sum_sqrt_ratio / n_f));
  estimate(kappa_hat, "kappa", NO_MEAN_REVERSION)?;

  let mut sum_res2 = 0.0;
  for i in 1..nobs {
    let v_prev = v[i - 1];
    let v_curr = v[i];
    let drift_term = (delta / (2.0 * v_prev.sqrt())) * (p_hat - kappa_hat * v_prev);
    let resid = v_curr.sqrt() - v_prev.sqrt() - drift_term;
    sum_res2 += resid * resid;
  }

  let sigma2_hat = 4.0 * sum_res2 / (delta * n_f);
  estimate(sigma2_hat, "sigma", ZERO_RESIDUALS)?;
  let sigma_hat = sigma2_hat.sqrt();

  let theta_hat = (p_hat + 0.25 * sigma2_hat) / kappa_hat;
  estimate(theta_hat, "theta", NON_POSITIVE_LEVEL)?;

  Ok(HestonMleFit {
    v0: v[0],
    kappa: kappa_hat,
    theta: theta_hat,
    sigma: sigma_hat,
    rho: rho_moment(s, v, r, delta, kappa_hat, theta_hat, sigma_hat)?,
  })
}

/// Wang et al. (2018) closed-form PMLE of Heston from `(S_t, V_t)` over one unit of time; `Err` as
/// [`fit_heston_nmle`], and panics on a non-finite `r`.
pub fn fit_heston_pmle(
  s: ArrayView1<f64>,
  v: ArrayView1<f64>,
  r: f64,
) -> Result<HestonMleFit, StatsError> {
  assert_r(r);
  validate_series(s, v)?;
  pmle(s, v, r, 1.0 / (v.len() - 1) as f64)
}

/// [`fit_heston_pmle`] at sampling interval `delta` (Wang et al.'s PMLE and the rho moment
/// estimator); `Err` as it, and panics unless `r` is finite and `delta` finite and positive.
pub fn fit_heston_pmle_with_delta(
  s: ArrayView1<f64>,
  v: ArrayView1<f64>,
  r: f64,
  delta: f64,
) -> Result<HestonMleFit, StatsError> {
  assert_r(r);
  assert_delta(delta);
  validate_series(s, v)?;
  pmle(s, v, r, delta)
}

fn pmle(
  s: ArrayView1<f64>,
  v: ArrayView1<f64>,
  r: f64,
  delta: f64,
) -> Result<HestonMleFit, StatsError> {
  let nobs = v.len();
  let n_f = (nobs - 1) as f64;

  let mut sum_v_k = 0.0;
  let mut sum_v_prev = 0.0;
  let mut sum_inv_v_prev = 0.0;
  let mut sum_ratio = 0.0;

  for i in 1..nobs {
    let v_prev = v[i - 1];
    let v_k = v[i];
    sum_v_k += v_k;
    sum_v_prev += v_prev;
    sum_inv_v_prev += 1.0 / v_prev;
    sum_ratio += v_k / v_prev;
  }

  let mean_v_k = sum_v_k / n_f;
  let mean_v_prev = sum_v_prev / n_f;
  let mean_inv_v_prev = sum_inv_v_prev / n_f;
  let mean_ratio = sum_ratio / n_f;

  let beta1_num = mean_v_k * mean_inv_v_prev - mean_ratio;
  let beta1_den = mean_v_prev * mean_inv_v_prev - 1.0;
  if beta1_den.abs() <= DENOMINATOR {
    return Err(StatsError::Numerical { stage: "kappa" });
  }
  // beta1 = exp(-kappa * delta): 1 or above has no mean reversion, 0 or below no persistence.
  let beta1_hat = beta1_num / beta1_den;
  if !beta1_hat.is_finite() {
    return Err(StatsError::Numerical { stage: "kappa" });
  }
  if beta1_hat >= 1.0 {
    return Err(StatsError::DegenerateData(NO_MEAN_REVERSION));
  }
  if beta1_hat <= 0.0 {
    return Err(StatsError::DegenerateData(
      "variance series without positive autocorrelation",
    ));
  }

  let beta2_hat = (mean_ratio - beta1_hat) / ((1.0 - beta1_hat) * mean_inv_v_prev);
  estimate(beta2_hat, "theta", NON_POSITIVE_LEVEL)?;

  let mut sum_beta3 = 0.0;
  for i in 1..nobs {
    let v_prev = v[i - 1];
    let v_k = v[i];
    let residual = v_k - v_prev * beta1_hat - beta2_hat * (1.0 - beta1_hat);
    sum_beta3 += residual * residual / v_prev;
  }
  let beta3_hat = sum_beta3 / n_f;
  estimate(beta3_hat, "sigma", ZERO_RESIDUALS)?;

  let kappa_hat = -(beta1_hat.ln()) / delta;
  estimate(kappa_hat, "kappa", NO_MEAN_REVERSION)?;
  let theta_hat = beta2_hat;
  let sigma2_hat = (2.0 * kappa_hat * beta3_hat) / (1.0 - beta1_hat * beta1_hat);
  estimate(sigma2_hat, "sigma", ZERO_RESIDUALS)?;
  let sigma_hat = sigma2_hat.sqrt();

  Ok(HestonMleFit {
    v0: v[0],
    kappa: kappa_hat,
    theta: theta_hat,
    sigma: sigma_hat,
    rho: rho_moment(s, v, r, delta, kappa_hat, theta_hat, sigma_hat)?,
  })
}

/// Below this, the dimensionless `mean(V) mean(1/V) - 1` is rounding on a near-constant path.
const DENOMINATOR: f64 = 1e-12;

const NO_MEAN_REVERSION: &str = "variance series without mean reversion";

const NON_POSITIVE_LEVEL: &str = "non-positive long-run variance";

const ZERO_RESIDUALS: &str = "zero residual variance";

fn estimate(value: f64, stage: &'static str, degenerate: &'static str) -> Result<(), StatsError> {
  if !value.is_finite() {
    return Err(StatsError::Numerical { stage });
  }
  if value <= 0.0 {
    return Err(StatsError::DegenerateData(degenerate));
  }
  Ok(())
}

/// Wang et al. (2018) eq. (23), the moment estimate of `rho`, held in `[-1, 1]`.
fn rho_moment(
  s: ArrayView1<f64>,
  v: ArrayView1<f64>,
  r: f64,
  delta: f64,
  kappa_hat: f64,
  theta_hat: f64,
  sigma_hat: f64,
) -> Result<f64, StatsError> {
  let nobs = v.len();
  let n_f = (nobs - 1) as f64;
  let mut sum_dw1dw2 = 0.0;
  for i in 1..nobs {
    let v_prev = v[i - 1];
    let v_curr = v[i];
    let s_prev = s[i - 1];
    let s_curr = s[i];

    let dw1_i = (s_curr.ln() - s_prev.ln() - (r - 0.5 * v_prev) * delta) / v_prev.sqrt();
    let dw2_i =
      (v_curr - v_prev - kappa_hat * (theta_hat - v_prev) * delta) / (sigma_hat * v_prev.sqrt());
    sum_dw1dw2 += dw1_i * dw2_i;
  }
  let rho_hat = sum_dw1dw2 / (n_f * delta);
  if !rho_hat.is_finite() {
    return Err(StatsError::Numerical { stage: "rho" });
  }
  Ok(rho_hat.clamp(-1.0, 1.0))
}

/// Equal lengths, 4 or more points, all finite and positive (two drift coefficients fit three
/// exactly), and lagged variances `V_0..V_{n-1}` that move: closed forms divide by their spread.
fn validate_series(s: ArrayView1<f64>, v: ArrayView1<f64>) -> Result<(), StatsError> {
  if s.len() != v.len() {
    return Err(StatsError::DimensionMismatch {
      expected: s.len(),
      got: v.len(),
    });
  }
  validate_view_domain(s, 4, "finite and > 0", |x| x > 0.0)?;
  let v = as_slice_or_copy(v);
  validate_domain(&v, 4, "finite and > 0", |x| x > 0.0)?;
  reject_constant(&v[..v.len() - 1], "constant variance series")
}

#[track_caller]
fn assert_r(r: f64) {
  assert!(
    r.is_finite(),
    "r must satisfy `r.is_finite()`, got r = {r:?}"
  );
}

#[track_caller]
fn assert_delta(delta: f64) {
  assert!(
    delta.is_finite() && delta > 0.0,
    "delta must satisfy `delta.is_finite() && delta > 0.0`, got delta = {delta:?}"
  );
}

#[cfg(test)]
mod tests;
