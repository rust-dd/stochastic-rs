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
use crate::input::validate;
use crate::input::validate_view;

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
  /// Correlation parameter.
  pub rho: f64,
}

/// Wang et al. (2018) closed-form NMLE of Heston from `(S_t, V_t)` over one unit of time; `Err` on
/// unequal lengths, fewer than 2 or non-finite observations or a constant variance path.
/// Reference: Wang et al. (2018), Sci. China Inf. Sci. 61, 042202. DOI: 10.1007/s11432-017-9215-8
pub fn fit_heston_nmle(
  s: ArrayView1<f64>,
  v: ArrayView1<f64>,
  r: f64,
) -> Result<HestonMleFit, StatsError> {
  validate_series(s, v)?;
  Ok(nmle(s, v, r, 1.0 / (v.len() - 1) as f64))
}

/// [`fit_heston_nmle`] at sampling interval `delta` (Wang et al. eq. 19-21 and the rho moment
/// estimator); `Err` as it, and panics unless `delta` is finite and positive.
pub fn fit_heston_nmle_with_delta(
  s: ArrayView1<f64>,
  v: ArrayView1<f64>,
  r: f64,
  delta: f64,
) -> Result<HestonMleFit, StatsError> {
  assert_delta(delta);
  validate_series(s, v)?;
  Ok(nmle(s, v, r, delta))
}

fn nmle(s: ArrayView1<f64>, v: ArrayView1<f64>, r: f64, delta: f64) -> HestonMleFit {
  const EPS: f64 = 1e-12;

  let nobs = v.len();
  let n_f = (nobs - 1) as f64;

  // Following Wang et al. (2018), Eq. (19)-(21):
  // P = kappa * theta - sigma^2 / 4, with closed-form estimators from sqrt-variance dynamics.
  let mut sum_inv_v_prev = 0.0;
  let mut sum_v_prev = 0.0;
  let mut sum_sqrt_ratio = 0.0;
  let mut sum_sqrt_prod = 0.0;

  for i in 1..nobs {
    let v_prev = v[i - 1].max(EPS);
    let v_curr = v[i].max(EPS);

    sum_inv_v_prev += 1.0 / v_prev;
    sum_v_prev += v_prev;
    sum_sqrt_ratio += (v_curr / v_prev).sqrt();
    sum_sqrt_prod += (v_curr * v_prev).sqrt();
  }

  let denom_inner = n_f - (sum_v_prev * sum_inv_v_prev) / n_f;
  let denom_p = 0.5 * delta * denom_inner;
  let p_hat = if denom_p.abs() > EPS {
    (sum_sqrt_prod - (sum_v_prev * sum_sqrt_ratio) / n_f) / denom_p
  } else {
    0.0
  };

  let mut kappa_hat =
    (2.0 / delta) * (1.0 + 0.5 * delta * p_hat * (sum_inv_v_prev / n_f) - (sum_sqrt_ratio / n_f));
  if !kappa_hat.is_finite() {
    kappa_hat = EPS;
  }
  kappa_hat = kappa_hat.max(EPS);

  let mut sum_res2 = 0.0;
  for i in 1..nobs {
    let v_prev = v[i - 1].max(EPS);
    let v_curr = v[i].max(EPS);
    let drift_term = (delta / (2.0 * v_prev.sqrt())) * (p_hat - kappa_hat * v_prev);
    let resid = v_curr.sqrt() - v_prev.sqrt() - drift_term;
    sum_res2 += resid * resid;
  }

  let sigma2_hat = (4.0 * sum_res2 / (delta * n_f)).max(EPS);
  let sigma_hat = sigma2_hat.sqrt();

  let mut theta_hat = (p_hat + 0.25 * sigma2_hat) / kappa_hat;
  if !theta_hat.is_finite() {
    theta_hat = v[0].max(EPS);
  }
  theta_hat = theta_hat.max(EPS);

  let mut sum_dw1dw2 = 0.0;
  for i in 1..nobs {
    let v_prev = v[i - 1].max(EPS);
    let v_curr = v[i].max(EPS);
    let s_prev = s[i - 1].max(EPS);
    let s_curr = s[i].max(EPS);

    let dw1_i = (s_curr.ln() - s_prev.ln() - (r - 0.5 * v_prev) * delta) / v_prev.sqrt();
    let dw2_i =
      (v_curr - v_prev - kappa_hat * (theta_hat - v_prev) * delta) / (sigma_hat * v_prev.sqrt());
    sum_dw1dw2 += dw1_i * dw2_i;
  }

  let rho_hat = (sum_dw1dw2 / (n_f * delta)).clamp(-1.0, 1.0);

  HestonMleFit {
    v0: v[0].max(0.0),
    kappa: kappa_hat,
    theta: theta_hat,
    sigma: sigma_hat,
    rho: rho_hat,
  }
}

/// Wang et al. (2018) closed-form PMLE of Heston from `(S_t, V_t)` over one unit of time; `Err` as
/// [`fit_heston_nmle`].
pub fn fit_heston_pmle(
  s: ArrayView1<f64>,
  v: ArrayView1<f64>,
  r: f64,
) -> Result<HestonMleFit, StatsError> {
  validate_series(s, v)?;
  Ok(pmle(s, v, r, 1.0 / (v.len() - 1) as f64))
}

/// [`fit_heston_pmle`] at sampling interval `delta` (Wang et al.'s PMLE and the rho moment
/// estimator); `Err` as it, and panics unless `delta` is finite and positive.
pub fn fit_heston_pmle_with_delta(
  s: ArrayView1<f64>,
  v: ArrayView1<f64>,
  r: f64,
  delta: f64,
) -> Result<HestonMleFit, StatsError> {
  assert_delta(delta);
  validate_series(s, v)?;
  Ok(pmle(s, v, r, delta))
}

fn pmle(s: ArrayView1<f64>, v: ArrayView1<f64>, r: f64, delta: f64) -> HestonMleFit {
  const EPS: f64 = 1e-12;

  let nobs = v.len();
  let n_f = (nobs - 1) as f64;

  let mut sum_v_k = 0.0;
  let mut sum_v_prev = 0.0;
  let mut sum_inv_v_prev = 0.0;
  let mut sum_ratio = 0.0;

  for i in 1..nobs {
    let v_prev = v[i - 1].max(EPS);
    let v_k = v[i].max(EPS);
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
  let mut beta1_hat = if beta1_den.abs() > EPS {
    beta1_num / beta1_den
  } else {
    1.0 - 1e-6
  };
  beta1_hat = beta1_hat.clamp(EPS, 1.0 - 1e-6);

  let beta2_den = (1.0 - beta1_hat) * mean_inv_v_prev;
  let mut beta2_hat = if beta2_den.abs() > EPS {
    (mean_ratio - beta1_hat) / beta2_den
  } else {
    mean_v_k
  };
  beta2_hat = beta2_hat.max(EPS);

  let mut sum_beta3 = 0.0;
  for i in 1..nobs {
    let v_prev = v[i - 1].max(EPS);
    let v_k = v[i].max(EPS);
    let residual = v_k - v_prev * beta1_hat - beta2_hat * (1.0 - beta1_hat);
    sum_beta3 += residual * residual / v_prev;
  }
  let beta3_hat = (sum_beta3 / n_f).max(EPS);

  let kappa_hat = (-(beta1_hat.ln()) / delta).max(EPS);
  let theta_hat = beta2_hat;
  let sigma2_hat =
    ((2.0 * kappa_hat * beta3_hat) / (1.0 - beta1_hat * beta1_hat).max(EPS)).max(EPS);
  let sigma_hat = sigma2_hat.sqrt();

  let mut sum_dw1dw2 = 0.0;
  for i in 1..nobs {
    let v_prev = v[i - 1].max(EPS);
    let v_k = v[i].max(EPS);
    let s_prev = s[i - 1].max(EPS);
    let s_curr = s[i].max(EPS);

    let dw1_i = (s_curr.ln() - s_prev.ln() - (r - 0.5 * v_prev) * delta) / v_prev.sqrt();
    let dw2_i =
      (v_k - v_prev - kappa_hat * (theta_hat - v_prev) * delta) / (sigma_hat * v_prev.sqrt());
    sum_dw1dw2 += dw1_i * dw2_i;
  }

  let rho_hat = (sum_dw1dw2 / (n_f * delta)).clamp(-1.0, 1.0);

  HestonMleFit {
    v0: v[0].max(0.0),
    kappa: kappa_hat,
    theta: theta_hat,
    sigma: sigma_hat,
    rho: rho_hat,
  }
}

/// Equal lengths, two or more finite observations each, and a variance path that moves: the
/// closed forms divide by its spread.
fn validate_series(s: ArrayView1<f64>, v: ArrayView1<f64>) -> Result<(), StatsError> {
  if s.len() != v.len() {
    return Err(StatsError::DimensionMismatch {
      expected: s.len(),
      got: v.len(),
    });
  }
  validate_view(s, 2)?;
  let v = as_slice_or_copy(v);
  validate(&v, 2)?;
  reject_constant(&v, "constant variance series")
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
