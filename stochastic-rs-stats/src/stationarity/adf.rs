//! Augmented Dickey-Fuller unit-root test.
//!
//! References:
//! - Dickey D. A. & Fuller W. A. (1979) — *Distribution of the
//!   Estimators for Autoregressive Time Series with a Unit Root*,
//!   Journal of the American Statistical Association 74(366), 427–431,
//!   DOI: 10.2307/2286348.
//! - Said S. E. & Dickey D. A. (1984) — *Testing for Unit Roots in
//!   Autoregressive-Moving Average Models of Unknown Order*, Biometrika
//!   71(3), 599–607, DOI: 10.1093/biomet/71.3.599 — the lagged-difference
//!   augmentation this file's `p`-lag regression implements.

use ndarray::ArrayView1;

use super::common::LagSelection;
use super::common::Trend;
use super::common::adf_critical_values;
use super::common::choose_lag_for_adf;
use super::common::fit_adf;
use super::common::schwert_max_lags;
use super::common::validate_series;
use crate::critical_values::CriticalValues;

/// Configuration for the Augmented Dickey-Fuller unit-root test.
#[derive(Debug, Clone, Copy)]
pub struct AdfConfig {
  /// Deterministic terms included in the test regression.
  pub trend: Trend,
  /// Lag-order selection strategy.
  pub lag_selection: LagSelection,
  /// Maximum lag considered by automatic lag selection.
  pub max_lags: Option<usize>,
  /// Significance level used to compute `rejects_null`.
  pub alpha: f64,
}

impl Default for AdfConfig {
  fn default() -> Self {
    Self {
      trend: Trend::Constant,
      lag_selection: LagSelection::Aic,
      max_lags: None,
      alpha: 0.05,
    }
  }
}

/// Augmented Dickey–Fuller test (null: a unit root); the table holds the 1 %, 5 % and 10 % levels.
#[derive(Debug, Clone)]
pub struct AdfTest {
  /// ADF t-statistic for the lagged level coefficient.
  pub statistic: f64,
  /// No p-value table is shipped; the decision is taken against `critical_values`.
  pub p_value: Option<f64>,
  pub critical_values: Option<CriticalValues>,
  /// Whether the null (unit root) is rejected at `alpha`.
  pub rejects_null: bool,
  /// Selected lag order.
  pub used_lags: usize,
  /// Number of regression observations used by the fitted model.
  pub nobs: usize,
}

impl crate::traits::HypothesisTest for AdfTest {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.rejects_null)
  }
}

/// Augmented Dickey-Fuller unit-root test.
///
/// # Panics
/// Panics on invalid inputs (non-finite series, too-short sample, invalid config).
pub fn adf_test(y: ArrayView1<f64>, cfg: AdfConfig) -> AdfTest {
  let y = y
    .as_slice()
    .expect("adf_test requires a contiguous ArrayView1");
  validate_series(y, 20);
  assert!(
    cfg.alpha > 0.0 && cfg.alpha < 1.0,
    "alpha must be in (0, 1)"
  );

  let max_possible_lag = y.len().saturating_sub(5);
  let max_lags = cfg
    .max_lags
    .unwrap_or_else(|| schwert_max_lags(y.len()))
    .min(max_possible_lag);

  let used_lags = match cfg.lag_selection {
    LagSelection::Fixed(p) => {
      assert!(
        p <= max_possible_lag,
        "fixed lag order too large for sample"
      );
      p
    }
    _ => choose_lag_for_adf(y, cfg.trend, cfg.lag_selection, max_lags),
  };

  let fit = fit_adf(y, used_lags, cfg.trend);
  let critical_values = adf_critical_values(cfg.trend);
  let rejects_null = fit.statistic < critical_values.value_at(cfg.alpha);

  AdfTest {
    statistic: fit.statistic,
    p_value: None,
    critical_values: Some(critical_values),
    rejects_null,
    used_lags,
    nobs: fit.nobs,
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::AdfConfig;
  use super::adf_test;
  use crate::stationarity::LagSelection;
  use crate::stationarity::Trend;

  fn simulate_ar1(phi: f64, n: usize, seed: u64) -> Vec<f64> {
    let innovations = {
      let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
      (0..n).map(|_| dist.sample()).collect::<Vec<_>>()
    };

    let mut x = vec![0.0; n];
    for t in 1..n {
      x[t] = phi * x[t - 1] + innovations[t];
    }
    x
  }

  fn simulate_random_walk(n: usize, seed: u64) -> Vec<f64> {
    let innovations = {
      let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
      (0..n).map(|_| dist.sample()).collect::<Vec<_>>()
    };

    let mut x = vec![0.0; n];
    for t in 1..n {
      x[t] = x[t - 1] + innovations[t];
    }
    x
  }

  #[test]
  fn adf_rejects_stationary_ar1() {
    let x = simulate_ar1(0.7, 2400, 0xADF1);
    let cfg = AdfConfig {
      trend: Trend::Constant,
      lag_selection: LagSelection::Fixed(4),
      ..AdfConfig::default()
    };
    let res = adf_test(ndarray::ArrayView1::from(&x), cfg);
    assert!(
      res.rejects_null,
      "expected unit-root rejection, got {res:?}"
    );
  }

  #[test]
  fn adf_random_walk_is_not_rejected_at_one_percent() {
    let x = simulate_random_walk(2400, 0xADF2);
    let cfg = AdfConfig {
      trend: Trend::Constant,
      lag_selection: LagSelection::Fixed(4),
      alpha: 0.01,
      ..AdfConfig::default()
    };
    let res = adf_test(ndarray::ArrayView1::from(&x), cfg);
    assert!(
      !res.rejects_null,
      "expected no rejection for random walk, got {res:?}"
    );
  }
}
