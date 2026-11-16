//! Heterogeneous Autoregressive model of Realized Volatility (HAR-RV).
//!
//! $$
//! RV_{t+1}^{(d)} = c + \beta_d RV_t^{(d)} + \beta_w RV_t^{(w)} + \beta_m RV_t^{(m)} + \omega_{t+1},
//! $$
//! with $RV_t^{(w)} = \tfrac{1}{5}\sum_{i=0}^{4} RV_{t-i}^{(d)}$ and
//! $RV_t^{(m)} = \tfrac{1}{22}\sum_{i=0}^{21} RV_{t-i}^{(d)}$.
//!
//! Reference: Corsi, "A Simple Approximate Long-Memory Model of Realized
//! Volatility", Journal of Financial Econometrics, 7(2), 174-196 (2009).
//! DOI: 10.1093/jjfinec/nbp001
//!
//! The OLS least-squares fit runs on the pure-Rust `faer`.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use crate::StatsError;
use crate::input::as_slice_or_copy;
use crate::input::reject_constant;
use crate::input::validate;
use crate::input::validate_view;
use crate::linalg::lstsq;
use crate::traits::FloatExt;

/// Daily horizon for the HAR weekly component.
pub const WEEKLY_LAGS: usize = 5;
/// Daily horizon for the HAR monthly component.
pub const MONTHLY_LAGS: usize = 22;

/// Result of fitting an HAR-RV regression by ordinary least squares.
#[derive(Debug, Clone)]
pub struct HarFit {
  /// Intercept $c$.
  pub intercept: f64,
  /// Daily-lag coefficient $\beta_d$.
  pub beta_d: f64,
  /// Weekly-average-lag coefficient $\beta_w$.
  pub beta_w: f64,
  /// Monthly-average-lag coefficient $\beta_m$.
  pub beta_m: f64,
  /// In-sample residual sum of squares.
  pub rss: f64,
  /// Sample size used by the regression.
  pub nobs: usize,
  /// Coefficient of determination on the training window; NaN for a constant target (`0/0`).
  pub r_squared: f64,
}

/// Fitted HAR-RV forecaster suitable for one-step-ahead prediction from a
/// rolling daily-RV history.
#[derive(Debug, Clone)]
pub struct HarRv {
  /// Underlying OLS fit.
  pub fit: HarFit,
}

impl HarRv {
  /// HAR-RV by least squares on a daily realized-variance history; `Err` below `MONTHLY_LAGS + 6`
  /// observations (more rows than the four regressors), on a non-finite one or a constant series.
  pub fn fit<T: FloatExt>(daily_rv: ArrayView1<T>) -> Result<Self, StatsError> {
    let rv = as_slice_or_copy(daily_rv);
    validate(&rv, MONTHLY_LAGS + 6)?;
    reject_constant(&rv, "constant series")?;
    let (x, y) = build_design_matrix(daily_rv);
    let beta = ols(&x, &y);
    let resid = residuals(&x, &y, &beta);
    let rss: f64 = resid.iter().map(|v| v * v).sum();
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;
    let tss: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum();
    // Exact: a rounded mean can leave a constant target a positive `tss`.
    let r_squared = if y.iter().all(|&v| v == y[0]) {
      f64::NAN
    } else {
      1.0 - rss / tss
    };
    Ok(Self {
      fit: HarFit {
        intercept: beta[0],
        beta_d: beta[1],
        beta_w: beta[2],
        beta_m: beta[3],
        rss,
        nobs: y.len(),
        r_squared,
      },
    })
  }

  /// One-step-ahead forecast from the latest daily RVs; `Err` below [`MONTHLY_LAGS`] observations
  /// or on a non-finite one.
  pub fn forecast<T: FloatExt>(&self, recent_daily_rv: ArrayView1<T>) -> Result<f64, StatsError> {
    validate_view(recent_daily_rv, MONTHLY_LAGS)?;
    let n = recent_daily_rv.len();
    let last = recent_daily_rv[n - 1].to_f64().unwrap();
    let weekly = avg_last(recent_daily_rv, WEEKLY_LAGS);
    let monthly = avg_last(recent_daily_rv, MONTHLY_LAGS);
    Ok(
      self.fit.intercept
        + self.fit.beta_d * last
        + self.fit.beta_w * weekly
        + self.fit.beta_m * monthly,
    )
  }
}

/// The design `[1, RV(d), RV(w), RV(m)]` and next-day target of a daily RV history.
fn build_design_matrix<T: FloatExt>(daily_rv: ArrayView1<T>) -> (Array2<f64>, Array1<f64>) {
  let n = daily_rv.len();
  let rows = n - MONTHLY_LAGS - 1;
  let mut x = Array2::<f64>::zeros((rows, 4));
  let mut y = Array1::<f64>::zeros(rows);
  for r in 0..rows {
    let t = MONTHLY_LAGS + r;
    let daily = daily_rv[t].to_f64().unwrap();
    let weekly = avg_window(daily_rv, t + 1 - WEEKLY_LAGS, t + 1);
    let monthly = avg_window(daily_rv, t + 1 - MONTHLY_LAGS, t + 1);
    x[[r, 0]] = 1.0;
    x[[r, 1]] = daily;
    x[[r, 2]] = weekly;
    x[[r, 3]] = monthly;
    y[r] = daily_rv[t + 1].to_f64().unwrap();
  }
  (x, y)
}

fn avg_window<T: FloatExt>(v: ArrayView1<T>, lo: usize, hi: usize) -> f64 {
  let mut acc = 0.0;
  for j in lo..hi {
    acc += v[j].to_f64().unwrap();
  }
  acc / (hi - lo) as f64
}

fn avg_last<T: FloatExt>(v: ArrayView1<T>, window: usize) -> f64 {
  let n = v.len();
  let lo = n - window;
  let mut acc = 0.0;
  for j in lo..n {
    acc += v[j].to_f64().unwrap();
  }
  acc / window as f64
}

fn ols(x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
  lstsq(x, y)
}

fn residuals(x: &Array2<f64>, y: &Array1<f64>, beta: &Array1<f64>) -> Array1<f64> {
  let yhat = x.dot(beta);
  y - &yhat
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::DistributionSampler;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  fn simulate_har_path(n: usize, c: f64, bd: f64, bw: f64, bm: f64, seed: u64) -> Array1<f64> {
    let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
    let mut shocks = vec![0.0_f64; n];
    dist.fill_slice(&mut shocks);
    let mut rv = Array1::<f64>::from_elem(n, c / (1.0 - bd - bw - bm).max(1e-3));
    for t in MONTHLY_LAGS..n {
      let d = rv[t - 1];
      let w: f64 = (1..=WEEKLY_LAGS).map(|i| rv[t - i]).sum::<f64>() / WEEKLY_LAGS as f64;
      let m: f64 = (1..=MONTHLY_LAGS).map(|i| rv[t - i]).sum::<f64>() / MONTHLY_LAGS as f64;
      let mean = c + bd * d + bw * w + bm * m;
      rv[t] = (mean + 0.05 * mean.abs() * shocks[t]).max(1e-8);
    }
    rv
  }

  #[test]
  fn har_recovers_simulation_coefficients_within_5_percent() {
    let c = 0.0001;
    let bd = 0.40;
    let bw = 0.30;
    let bm = 0.20;
    let rv = simulate_har_path(2_000, c, bd, bw, bm, 17);
    let model = HarRv::fit(rv.view()).unwrap();
    assert!((model.fit.beta_d - bd).abs() < 0.05);
    assert!((model.fit.beta_w - bw).abs() < 0.10);
    assert!((model.fit.beta_m - bm).abs() < 0.15);
    assert!(model.fit.r_squared > 0.5);
  }

  #[test]
  fn har_forecast_matches_one_step_design_row() {
    let rv = simulate_har_path(500, 0.0001, 0.40, 0.30, 0.20, 23);
    let model = HarRv::fit(rv.view()).unwrap();
    let recent = rv.slice(ndarray::s![rv.len() - MONTHLY_LAGS..]).to_owned();
    let f = model.forecast(recent.view()).unwrap();
    let last = recent[MONTHLY_LAGS - 1];
    let w: f64 = (1..=WEEKLY_LAGS)
      .map(|i| recent[MONTHLY_LAGS - i])
      .sum::<f64>()
      / WEEKLY_LAGS as f64;
    let m: f64 = (1..=MONTHLY_LAGS)
      .map(|i| recent[MONTHLY_LAGS - i])
      .sum::<f64>()
      / MONTHLY_LAGS as f64;
    let expected =
      model.fit.intercept + model.fit.beta_d * last + model.fit.beta_w * w + model.fit.beta_m * m;
    assert!((f - expected).abs() < 1e-12);
  }

  /// The regression rows must outnumber the four regressors.
  #[test]
  fn too_short_histories_are_insufficient_data() {
    let rv = Array1::from_iter((0..MONTHLY_LAGS + 6).map(|t| 1.0 + 0.5 * (t as f64 * 0.7).sin()));
    assert_eq!(
      HarRv::fit(rv.slice(ndarray::s![..MONTHLY_LAGS + 5])).unwrap_err(),
      StatsError::InsufficientData {
        needed: 28,
        got: 27
      }
    );
    assert_eq!(HarRv::fit(rv.view()).unwrap().fit.nobs, 5);
    let model = HarRv::fit(simulate_har_path(200, 0.0001, 0.4, 0.3, 0.2, 29).view()).unwrap();
    assert_eq!(
      model.forecast(rv.slice(ndarray::s![..5])).unwrap_err(),
      StatsError::InsufficientData {
        needed: MONTHLY_LAGS,
        got: 5
      }
    );
  }

  /// `1 − RSS/TSS` is `0/0` when every target equals the first.
  #[test]
  fn r_squared_is_nan_for_a_constant_target() {
    let mut rv =
      Array1::from_iter((0..MONTHLY_LAGS + 8).map(|t| 1.0 + 0.5 * (t as f64 * 0.7).sin()));
    rv.slice_mut(ndarray::s![MONTHLY_LAGS + 1..]).fill(0.75);
    assert!(HarRv::fit(rv.view()).unwrap().fit.r_squared.is_nan());
  }
}
