use super::common::CriticalValues;
use super::common::DeterministicTerm;
use super::common::LagSelection;
use super::common::adf_critical_values;
use super::common::choose_lag_for_adf;
use super::common::fit_adf;
use super::common::schwert_max_lags;
use super::common::validate_series;

/// Configuration for the Augmented Dickey-Fuller unit-root test.
#[derive(Debug, Clone, Copy)]
pub struct ADFConfig {
  /// Deterministic terms included in the test regression.
  pub deterministic: DeterministicTerm,
  /// Lag-order selection strategy.
  pub lag_selection: LagSelection,
  /// Maximum lag considered by automatic lag selection.
  pub max_lags: Option<usize>,
  /// Significance level used to compute `reject_unit_root`.
  pub alpha: f64,
}

impl Default for ADFConfig {
  fn default() -> Self {
    Self {
      deterministic: DeterministicTerm::Constant,
      lag_selection: LagSelection::Aic,
      max_lags: None,
      alpha: 0.05,
    }
  }
}

/// Result of the Augmented Dickey-Fuller test.
#[derive(Debug, Clone, Copy)]
pub struct ADFResult {
  /// ADF t-statistic for the lagged level coefficient.
  pub statistic: f64,
  /// Selected lag order.
  pub used_lags: usize,
  /// Number of regression observations used by the fitted model.
  pub nobs: usize,
  /// Critical values at 1%, 5%, 10% levels.
  pub critical_values: CriticalValues,
  /// Whether the null (unit root) is rejected at `alpha`.
  pub reject_unit_root: bool,
}

/// Augmented Dickey-Fuller unit-root test.
///
/// # Panics
/// Panics on invalid inputs (non-finite series, too-short sample, invalid config).
pub fn adf_test(y: &[f64], cfg: ADFConfig) -> ADFResult {
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
    _ => choose_lag_for_adf(y, cfg.deterministic, cfg.lag_selection, max_lags),
  };

  let fit = fit_adf(y, used_lags, cfg.deterministic);
  let critical_values = adf_critical_values(cfg.deterministic);
  let reject_unit_root = fit.statistic < critical_values.value_at(cfg.alpha);

  ADFResult {
    statistic: fit.statistic,
    used_lags,
    nobs: fit.nobs,
    critical_values,
    reject_unit_root,
  }
}

#[cfg(test)]
mod tests {
  use super::ADFConfig;
  use super::adf_test;
  use crate::distributions::normal::SimdNormal;
  use crate::stats::stationarity::common::DeterministicTerm;
  use crate::stats::stationarity::common::LagSelection;

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
  fn adf_rejects_stationary_ar1() {
    let x = simulate_ar1(0.7, 2400);
    let cfg = ADFConfig {
      deterministic: DeterministicTerm::Constant,
      lag_selection: LagSelection::Fixed(4),
      ..ADFConfig::default()
    };
    let res = adf_test(&x, cfg);
    assert!(
      res.reject_unit_root,
      "expected unit-root rejection, got {res:?}"
    );
  }

  #[test]
  fn adf_keeps_unit_root_for_random_walk() {
    let x = simulate_random_walk(2400);
    let cfg = ADFConfig {
      deterministic: DeterministicTerm::Constant,
      lag_selection: LagSelection::Fixed(4),
      ..ADFConfig::default()
    };
    let res = adf_test(&x, cfg);
    assert!(
      !res.reject_unit_root,
      "expected no rejection for random walk, got {res:?}"
    );
  }
}
