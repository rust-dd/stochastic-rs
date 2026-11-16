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
//! - Ng S. & Perron P. (2005) — *A Note on the Selection of Time Series
//!   Models*, Oxford Bulletin of Economics and Statistics 67(1), 115–134,
//!   DOI: 10.1111/j.1468-0084.2005.00113.x — every candidate lag scored on
//!   one common sample, as statsmodels' `adfuller` and arch's `ADF` do.

use ndarray::ArrayView1;

use super::common::LagSelection;
use super::common::Trend;
use super::common::adf_critical_values;
use super::common::choose_lag_for_adf;
use super::common::fit_adf;
use super::common::min_sample_size;
use super::common::schwert_max_lags;
use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::input::as_slice_or_copy;
use crate::input::assert_alpha;
use crate::input::reject_constant;
use crate::input::validate;

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

/// ADF test; `Err` below `max(20, 2p + c + 3)` points for the largest lag `p` and `c` terms, on
/// non-finite or constant data or a singular or exact fit; panics unless `0.01 <= alpha < 1`.
pub fn adf_test(y: ArrayView1<f64>, cfg: AdfConfig) -> Result<AdfTest, StatsError> {
  assert_alpha(cfg.alpha);
  let critical_values = adf_critical_values(cfg.trend);
  let critical = critical_values.value_at(cfg.alpha);
  let y = as_slice_or_copy(y);
  validate(&y, 20)?;
  reject_constant(&y, "constant series")?;

  let n = y.len();
  let max_lags = match cfg.lag_selection {
    LagSelection::Fixed(p) => p,
    _ => cfg.max_lags.unwrap_or_else(|| schwert_max_lags(n)),
  };
  // The lag-p ADF regression has the rows and columns of a levels autoregression of order p + 1.
  let needed = min_sample_size(max_lags.saturating_add(1), cfg.trend);
  if n < needed {
    return Err(StatsError::InsufficientData { needed, got: n });
  }

  let used_lags = choose_lag_for_adf(&y, cfg.trend, cfg.lag_selection, max_lags)?;
  let fit = fit_adf(&y, used_lags, cfg.trend, &y)?;

  Ok(AdfTest {
    statistic: fit.statistic,
    p_value: None,
    critical_values: Some(critical_values),
    rejects_null: fit.statistic < critical,
    used_lags,
    nobs: fit.nobs,
  })
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::AdfConfig;
  use super::adf_test;
  use crate::StatsError;
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
    let res = adf_test(ndarray::ArrayView1::from(&x), cfg).unwrap();
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
    let res = adf_test(ndarray::ArrayView1::from(&x), cfg).unwrap();
    assert!(
      !res.rejects_null,
      "expected no rejection for random walk, got {res:?}"
    );
  }

  #[test]
  fn a_trend_regression_with_the_default_lags_needs_twenty_one_points() {
    let x = simulate_ar1(0.5, 20, 0xADF3);
    let cfg = AdfConfig {
      trend: Trend::ConstantTrend,
      ..AdfConfig::default()
    };
    assert_eq!(
      adf_test(ndarray::ArrayView1::from(&x), cfg).unwrap_err(),
      StatsError::InsufficientData {
        needed: 21,
        got: 20
      }
    );
  }

  #[test]
  fn a_fixed_lag_needs_two_lags_plus_the_terms_plus_three_points() {
    let x = simulate_ar1(0.5, 20, 0xADF4);
    let fixed = |p| AdfConfig {
      lag_selection: LagSelection::Fixed(p),
      ..AdfConfig::default()
    };
    assert_eq!(
      adf_test(ndarray::ArrayView1::from(&x), fixed(15)).unwrap_err(),
      StatsError::InsufficientData {
        needed: 34,
        got: 20
      }
    );
    assert_eq!(
      adf_test(ndarray::ArrayView1::from(&x), fixed(9)).unwrap_err(),
      StatsError::InsufficientData {
        needed: 22,
        got: 20
      }
    );
    assert_eq!(
      adf_test(ndarray::ArrayView1::from(&x), fixed(8))
        .unwrap()
        .used_lags,
      8
    );
  }

  #[test]
  fn a_max_lag_beyond_the_sample_is_insufficient_data() {
    let x = simulate_ar1(0.5, 20, 0xADF5);
    let cfg = AdfConfig {
      max_lags: Some(15),
      ..AdfConfig::default()
    };
    assert_eq!(
      adf_test(ndarray::ArrayView1::from(&x), cfg).unwrap_err(),
      StatsError::InsufficientData {
        needed: 34,
        got: 20
      }
    );
  }

  #[test]
  fn a_lag_bound_at_usize_max_is_insufficient_data() {
    let x = simulate_ar1(0.5, 20, 0xADF6);
    let configs = [
      AdfConfig {
        lag_selection: LagSelection::Fixed(usize::MAX),
        ..AdfConfig::default()
      },
      AdfConfig {
        max_lags: Some(usize::MAX),
        ..AdfConfig::default()
      },
    ];
    for cfg in configs {
      assert_eq!(
        adf_test(ndarray::ArrayView1::from(&x), cfg).unwrap_err(),
        StatsError::InsufficientData {
          needed: usize::MAX,
          got: 20
        }
      );
    }
  }
}
