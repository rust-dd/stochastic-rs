//! Elliott-Rothenberg-Stock (1996) efficient (GLS-detrended) unit-root
//! test, also known as DF-GLS.
//!
//! Reference: Elliott G., Rothenberg T. J., Stock J. H. (1996) —
//! *Efficient Tests for an Autoregressive Unit Root*, Econometrica
//! 64(4), 813–836, DOI: 10.2307/2171846.

use ndarray::ArrayView1;

use super::common::LagSelection;
use super::common::Trend;
use super::common::choose_lag_for_adf;
use super::common::dfgls_critical_values;
use super::common::fit_adf;
use super::common::min_sample_size;
use super::common::ols;
use super::common::prepared;
use super::common::schwert_max_lags;
use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::input::assert_alpha;
use crate::input::validate;

/// Configuration for the Elliott-Rothenberg-Stock DF-GLS test.
#[derive(Debug, Clone, Copy)]
pub struct ErsConfig {
  /// GLS detrending: `Constant` or `ConstantTrend`.
  pub trend: Trend,
  /// How the lag is chosen: on the GLS-detrended series; arch's `DFGLS` takes the OLS-detrended one
  /// after Perron & Qu (2007), Economics Letters 94(1), 12–19, DOI: 10.1016/j.econlet.2006.06.009.
  pub lag_selection: LagSelection,
  /// Maximum lag considered by automatic lag selection.
  pub max_lags: Option<usize>,
  /// Significance level used to compute `rejects_null`.
  pub alpha: f64,
}

impl Default for ErsConfig {
  fn default() -> Self {
    Self {
      trend: Trend::Constant,
      lag_selection: LagSelection::Aic,
      max_lags: None,
      alpha: 0.05,
    }
  }
}

/// ERS DF-GLS test (null: a unit root); the table holds the 1 %, 5 % and 10 % levels.
#[derive(Debug, Clone)]
pub struct ErsTest {
  /// DF-GLS test statistic.
  pub statistic: f64,
  /// No p-value table is shipped; the decision is taken against `critical_values`.
  pub p_value: Option<f64>,
  pub critical_values: Option<CriticalValues>,
  /// Whether the null (unit root) is rejected at `alpha`.
  pub rejects_null: bool,
  /// Selected lag order.
  pub used_lags: usize,
  /// Number of observations used in the terminal regression.
  pub nobs: usize,
}

impl crate::traits::HypothesisTest for ErsTest {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.rejects_null)
  }
}

fn gls_detrend(y: &[f64], include_trend: bool) -> Result<Vec<f64>, StatsError> {
  validate(y, 20)?;
  let n = y.len();
  let n_f = n as f64;

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

  // Row one is `(y_1, z_1)` itself, as in Elliott, Rothenberg & Stock (1996) and arch's `DFGLS`.
  let mut delta_y = Vec::with_capacity(n);
  let mut delta_z = Vec::with_capacity(n);
  delta_y.push(y[0]);
  delta_z.push(z[0].clone());
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

  let fit = ols(&delta_y, &delta_z)?;
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
  Ok(detrended)
}

/// DF-GLS; `Err` below `max(20, 2p + 3)` points for the largest lag `p`, on non-finite or constant
/// data or a singular or exact fit; panics unless `trend != Trend::None` and `0.01 <= alpha < 1`.
pub fn ers_dfgls_test(y: ArrayView1<f64>, cfg: ErsConfig) -> Result<ErsTest, StatsError> {
  assert!(
    cfg.trend != Trend::None,
    "trend must satisfy `trend != Trend::None`, got trend = {:?}",
    cfg.trend
  );
  assert_alpha(cfg.alpha);
  let include_trend = cfg.trend == Trend::ConstantTrend;
  let critical_values = dfgls_critical_values(include_trend);
  let critical = critical_values.value_at(cfg.alpha);
  let y = prepared(y, 20)?;

  let n = y.len();
  let max_lags = match cfg.lag_selection {
    LagSelection::Fixed(p) => p,
    _ => cfg.max_lags.unwrap_or_else(|| schwert_max_lags(n)),
  };
  // The ADF stage on the detrended series is a levels autoregression of order p + 1, no terms.
  let needed = min_sample_size(max_lags.saturating_add(1), Trend::None);
  if n < needed {
    return Err(StatsError::InsufficientData { needed, got: n });
  }

  let y_detrended = gls_detrend(&y, include_trend)?;
  let used_lags = choose_lag_for_adf(&y_detrended, Trend::None, cfg.lag_selection, max_lags)?;
  let fit = fit_adf(&y_detrended, used_lags, Trend::None, &y)?;

  Ok(ErsTest {
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
  use stochastic_rs_distributions::DistributionSampler;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::ErsConfig;
  use super::ers_dfgls_test;
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

  #[test]
  fn ers_rejects_stationary_ar1() {
    let x = simulate_ar1(0.8, 2400, 0xE55A11);
    let cfg = ErsConfig {
      trend: Trend::Constant,
      lag_selection: LagSelection::Fixed(4),
      ..ErsConfig::default()
    };
    let res = ers_dfgls_test(ndarray::ArrayView1::from(&x), cfg).unwrap();
    assert!(res.rejects_null, "expected rejection, got {res:?}");
  }

  #[test]
  fn a_fixed_lag_needs_two_lags_plus_three_points() {
    let x = simulate_ar1(0.5, 20, 0xE55A12);
    let fixed = |p| ErsConfig {
      lag_selection: LagSelection::Fixed(p),
      ..ErsConfig::default()
    };
    assert_eq!(
      ers_dfgls_test(ndarray::ArrayView1::from(&x), fixed(9)).unwrap_err(),
      StatsError::InsufficientData {
        needed: 21,
        got: 20
      }
    );
    assert_eq!(
      ers_dfgls_test(ndarray::ArrayView1::from(&x), fixed(8))
        .unwrap()
        .used_lags,
      8
    );
  }

  /// arch 9.0.0's `DFGLS(y, trend, lags=2)` on walks `5 + 0.5 Σ z`, whose GLS regression keeps
  /// `(y_1, z_1)` as its first row, as Elliott, Rothenberg & Stock (1996) do.
  #[test]
  fn the_statistics_match_arch() {
    let walk = |seed: u64| {
      let mut z = vec![0.0; 200];
      SimdNormal::<f64>::new(0.0, 1.0)
        .seeded(&Deterministic::new(seed))
        .fill_slice(&mut z);
      let mut w = 0.0;
      z.into_iter()
        .map(|e| {
          w += e;
          5.0 + 0.5 * w
        })
        .collect::<Vec<_>>()
    };
    let cases = [
      (80_000, Trend::Constant, -0.8258479031397337),
      (80_000, Trend::ConstantTrend, -1.9136670415239592),
      (80_001, Trend::Constant, -2.1541778009705217),
      (80_001, Trend::ConstantTrend, -2.181113835761613),
      (80_002, Trend::Constant, -1.4745473727889928),
      (80_002, Trend::ConstantTrend, -1.8381066346233128),
      (80_003, Trend::Constant, -1.1963795015681389),
      (80_003, Trend::ConstantTrend, -1.4457474312335827),
    ];
    for (seed, trend, arch) in cases {
      let cfg = ErsConfig {
        trend,
        lag_selection: LagSelection::Fixed(2),
        ..ErsConfig::default()
      };
      let got = ers_dfgls_test(ndarray::ArrayView1::from(&walk(seed)), cfg)
        .unwrap()
        .statistic;
      assert!(
        (got - arch).abs() <= 1e-10 * arch.abs(),
        "seed {seed} {trend:?}: {got} against arch's {arch}"
      );
    }
  }

  #[test]
  fn a_lag_bound_at_usize_max_is_insufficient_data() {
    let x = simulate_ar1(0.5, 20, 0xE55A13);
    let configs = [
      ErsConfig {
        lag_selection: LagSelection::Fixed(usize::MAX),
        ..ErsConfig::default()
      },
      ErsConfig {
        max_lags: Some(usize::MAX),
        ..ErsConfig::default()
      },
    ];
    for cfg in configs {
      assert_eq!(
        ers_dfgls_test(ndarray::ArrayView1::from(&x), cfg).unwrap_err(),
        StatsError::InsufficientData {
          needed: usize::MAX,
          got: 20
        }
      );
    }
  }
}
