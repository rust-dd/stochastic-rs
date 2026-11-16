//! KPSS stationarity test: an LM statistic for the null of
//! (trend-)stationarity against a unit-root alternative, using a
//! Newey-West long-run variance estimate.
//!
//! References:
//! - Kwiatkowski D., Phillips P. C. B., Schmidt P., Shin Y. (1992) —
//!   *Testing the Null Hypothesis of Stationarity Against the
//!   Alternative of a Unit Root*, Journal of Econometrics 54(1-3),
//!   159–178, DOI: 10.1016/0304-4076(92)90104-Y. Critical values below
//!   are this paper's Table 1.
//! - Newey W. K. & West K. D. (1987) — *A Simple, Positive
//!   Semi-Definite, Heteroskedasticity and Autocorrelation Consistent
//!   Covariance Matrix*, Econometrica 55(3), 703–708,
//!   DOI: 10.2307/1913610.

use ndarray::ArrayView1;

use super::common::Trend;
use super::common::newey_west_long_run_variance;
use super::common::regress_on_deterministics;
use super::common::schwert_max_lags;
use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::input::as_slice_or_copy;
use crate::input::assert_alpha;
use crate::input::reject_constant;
use crate::input::validate;

/// Configuration for the KPSS stationarity test.
#[derive(Debug, Clone, Copy)]
pub struct KpssConfig {
  /// Deterministic component under the null: `Constant` (level) or `ConstantTrend`.
  pub trend: Trend,
  /// Newey-West lag length. If `None`, a Schwert-style default is used.
  pub lags: Option<usize>,
  /// Significance level used to compute `rejects_null`.
  pub alpha: f64,
}

impl Default for KpssConfig {
  fn default() -> Self {
    Self {
      trend: Trend::Constant,
      lags: None,
      alpha: 0.05,
    }
  }
}

/// KPSS test (null: stationarity); the table holds the upper-tail 1 %, 2.5 %, 5 % and 10 % levels.
#[derive(Debug, Clone)]
pub struct KpssTest {
  /// KPSS LM statistic.
  pub statistic: f64,
  /// No p-value table is shipped; the decision is taken against `critical_values`.
  pub p_value: Option<f64>,
  pub critical_values: Option<CriticalValues>,
  /// Whether the null (stationarity) is rejected at `alpha`.
  pub rejects_null: bool,
  /// Newey-West lag length used.
  pub used_lags: usize,
}

impl crate::traits::HypothesisTest for KpssTest {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.rejects_null)
  }
}

fn kpss_critical_values(include_trend: bool) -> CriticalValues {
  if include_trend {
    CriticalValues::new([(0.01, 0.216), (0.025, 0.176), (0.05, 0.146), (0.10, 0.119)])
  } else {
    CriticalValues::new([(0.01, 0.739), (0.025, 0.574), (0.05, 0.463), (0.10, 0.347)])
  }
}

/// KPSS test; `Err` below 20 points, on non-finite or constant data, a zero or non-finite long-run
/// variance or an overflowing statistic; panics unless `trend != Trend::None`, `0.01 <= alpha < 1`.
pub fn kpss_test(y: ArrayView1<f64>, cfg: KpssConfig) -> Result<KpssTest, StatsError> {
  assert!(
    cfg.trend != Trend::None,
    "trend must satisfy `trend != Trend::None`, got trend = {:?}",
    cfg.trend
  );
  assert_alpha(cfg.alpha);
  let include_trend = cfg.trend == Trend::ConstantTrend;
  let critical_values = kpss_critical_values(include_trend);
  let critical = critical_values.value_at(cfg.alpha);
  let y = as_slice_or_copy(y);
  validate(&y, 20)?;
  reject_constant(&y, "constant series")?;

  let reg = regress_on_deterministics(&y, include_trend)?;
  let resid = reg.residuals;
  let n = resid.len();
  let n_f = n as f64;

  let mut cum = 0.0;
  let mut eta = 0.0;
  for u in &resid {
    cum += *u;
    eta += cum * cum;
  }
  eta /= n_f * n_f;

  let used_lags = cfg.lags.unwrap_or_else(|| schwert_max_lags(n));
  let long_run_var = newey_west_long_run_variance(&resid, used_lags)?;
  let statistic = eta / long_run_var;
  if !statistic.is_finite() {
    return Err(StatsError::Numerical { stage: "statistic" });
  }

  Ok(KpssTest {
    statistic,
    p_value: None,
    critical_values: Some(critical_values),
    rejects_null: statistic > critical,
    used_lags,
  })
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::KpssConfig;
  use super::kpss_test;

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
  fn kpss_keeps_stationarity_for_ar1() {
    // KPSS rejects a genuinely stationary series at its nominal level, so one
    // path decides nothing; require that at least one of three holds.
    let runs = [0x4B505353u64, 0x4B505354, 0x4B505355].map(|seed| {
      let x = simulate_ar1(0.75, 2000, seed);
      (
        seed,
        kpss_test(ndarray::ArrayView1::from(&x), KpssConfig::default()).unwrap(),
      )
    });
    let ok = runs.iter().any(|(_, r)| !r.rejects_null);
    let report = runs
      .iter()
      .map(|(s, r)| format!("seed {s:#x}: stat={:.4}", r.statistic))
      .collect::<Vec<_>>()
      .join("; ");
    assert!(ok, "every seed rejected a stationary AR(1) — {report}");
  }

  #[test]
  fn kpss_rejects_stationarity_for_random_walk() {
    let x = simulate_random_walk(2000, 0x4B505354);
    let res = kpss_test(ndarray::ArrayView1::from(&x), KpssConfig::default()).unwrap();
    assert!(
      res.rejects_null,
      "expected rejection for random walk, got {res:?}"
    );
  }

  #[test]
  fn a_constant_series_is_degenerate_data() {
    let x = ndarray::Array1::<f64>::from_elem(64, 0.3);
    assert_eq!(
      kpss_test(x.view(), KpssConfig::default()).unwrap_err(),
      crate::StatsError::DegenerateData("constant series")
    );
  }
}
