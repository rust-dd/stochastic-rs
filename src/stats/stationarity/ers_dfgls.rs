use super::common::CriticalValues;
use super::common::DeterministicTerm;
use super::common::LagSelection;
use super::common::choose_lag_for_adf;
use super::common::dfgls_critical_values;
use super::common::fit_adf;
use super::common::ols;
use super::common::schwert_max_lags;
use super::common::validate_series;

/// Deterministic specification for ERS / DF-GLS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ERSTrend {
  /// Constant-only GLS detrending.
  Constant,
  /// Constant + linear trend GLS detrending.
  ConstantTrend,
}

/// Configuration for the Elliott-Rothenberg-Stock DF-GLS test.
#[derive(Debug, Clone, Copy)]
pub struct ERSConfig {
  /// Trend specification used in GLS detrending.
  pub trend: ERSTrend,
  /// Lag-order selection strategy for the ADF-on-detrended stage.
  pub lag_selection: LagSelection,
  /// Maximum lag considered by automatic lag selection.
  pub max_lags: Option<usize>,
  /// Significance level used to compute `reject_unit_root`.
  pub alpha: f64,
}

impl Default for ERSConfig {
  fn default() -> Self {
    Self {
      trend: ERSTrend::Constant,
      lag_selection: LagSelection::Aic,
      max_lags: None,
      alpha: 0.05,
    }
  }
}

/// Result of the Elliott-Rothenberg-Stock DF-GLS test.
#[derive(Debug, Clone, Copy)]
pub struct ERSResult {
  /// DF-GLS test statistic.
  pub statistic: f64,
  /// Selected lag order.
  pub used_lags: usize,
  /// Number of observations used in the terminal regression.
  pub nobs: usize,
  /// Critical values at 1%, 5%, 10% levels.
  pub critical_values: CriticalValues,
  /// Whether the null (unit root) is rejected at `alpha`.
  pub reject_unit_root: bool,
}

fn gls_detrend(y: &[f64], trend: ERSTrend) -> Vec<f64> {
  validate_series(y, 20);
  let n = y.len();
  let n_f = n as f64;

  let include_trend = matches!(trend, ERSTrend::ConstantTrend);
  let cbar = if include_trend { -13.5 } else { -7.0 };
  let alpha = 1.0 + cbar / n_f;

  let mut z = Vec::with_capacity(n);
  for t in 0..n {
    if include_trend {
      z.push(vec![1.0, (t + 1) as f64]);
    } else {
      z.push(vec![1.0]);
    }
  }

  let mut delta_y = Vec::with_capacity(n - 1);
  let mut delta_z = Vec::with_capacity(n - 1);
  for t in 1..n {
    delta_y.push(y[t] - alpha * y[t - 1]);
    let row_t = &z[t];
    let row_tm1 = &z[t - 1];
    let mut row = vec![0.0; row_t.len()];
    for i in 0..row.len() {
      row[i] = row_t[i] - alpha * row_tm1[i];
    }
    delta_z.push(row);
  }

  let fit = ols(&delta_y, &delta_z);
  let beta = fit.beta;

  let mut detrended = vec![0.0; n];
  for t in 0..n {
    let trend_val = z[t]
      .iter()
      .zip(beta.iter())
      .map(|(a, b)| a * b)
      .sum::<f64>();
    detrended[t] = y[t] - trend_val;
  }
  detrended
}

/// Elliott-Rothenberg-Stock DF-GLS test.
///
/// # Panics
/// Panics on invalid inputs (non-finite series, too-short sample, invalid config).
pub fn ers_dfgls_test(y: &[f64], cfg: ERSConfig) -> ERSResult {
  validate_series(y, 20);
  assert!(
    cfg.alpha > 0.0 && cfg.alpha < 1.0,
    "alpha must be in (0, 1)"
  );

  let y_detrended = gls_detrend(y, cfg.trend);
  let max_possible_lag = y_detrended.len().saturating_sub(5);
  let max_lags = cfg
    .max_lags
    .unwrap_or_else(|| schwert_max_lags(y_detrended.len()))
    .min(max_possible_lag);

  let used_lags = match cfg.lag_selection {
    LagSelection::Fixed(p) => {
      assert!(
        p <= max_possible_lag,
        "fixed lag order too large for sample"
      );
      p
    }
    _ => choose_lag_for_adf(
      &y_detrended,
      DeterministicTerm::None,
      cfg.lag_selection,
      max_lags,
    ),
  };

  let fit = fit_adf(&y_detrended, used_lags, DeterministicTerm::None);
  let critical_values = dfgls_critical_values(matches!(cfg.trend, ERSTrend::ConstantTrend));
  let reject_unit_root = fit.statistic < critical_values.value_at(cfg.alpha);

  ERSResult {
    statistic: fit.statistic,
    used_lags,
    nobs: fit.nobs,
    critical_values,
    reject_unit_root,
  }
}

#[cfg(test)]
mod tests {
  use super::ERSConfig;
  use super::ers_dfgls_test;
  use crate::distributions::normal::SimdNormal;
  use crate::stats::stationarity::common::LagSelection;
  use crate::stats::stationarity::ers_dfgls::ERSTrend;

  fn simulate_ar1(phi: f64, n: usize) -> Vec<f64> {
    let innovations = {
      let dist = SimdNormal::<f64>::new(0.0, 1.0);
      let mut rng = rand::rng();
      let mut eps = vec![0.0; n];
      dist.fill_slice(&mut rng, &mut eps);
      eps
    };

    let mut x = vec![0.0; n];
    for t in 1..n {
      x[t] = phi * x[t - 1] + innovations[t];
    }
    x
  }

  fn simulate_random_walk(n: usize) -> Vec<f64> {
    let innovations = {
      let dist = SimdNormal::<f64>::new(0.0, 1.0);
      let mut rng = rand::rng();
      let mut eps = vec![0.0; n];
      dist.fill_slice(&mut rng, &mut eps);
      eps
    };

    let mut x = vec![0.0; n];
    for t in 1..n {
      x[t] = x[t - 1] + innovations[t];
    }
    x
  }

  #[test]
  fn ers_rejects_stationary_ar1() {
    let x = simulate_ar1(0.8, 2400);
    let cfg = ERSConfig {
      trend: ERSTrend::Constant,
      lag_selection: LagSelection::Fixed(4),
      ..ERSConfig::default()
    };
    let res = ers_dfgls_test(&x, cfg);
    assert!(res.reject_unit_root, "expected rejection, got {res:?}");
  }

  #[test]
  fn ers_keeps_unit_root_for_random_walk() {
    let x = simulate_random_walk(2400);
    let cfg = ERSConfig {
      trend: ERSTrend::Constant,
      lag_selection: LagSelection::Fixed(4),
      ..ERSConfig::default()
    };
    let res = ers_dfgls_test(&x, cfg);
    assert!(
      !res.reject_unit_root,
      "expected no rejection for random walk, got {res:?}"
    );
  }
}
