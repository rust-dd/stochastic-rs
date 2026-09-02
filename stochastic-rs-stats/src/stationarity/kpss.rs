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

use super::common::newey_west_long_run_variance;
use super::common::regress_on_deterministics;
use super::common::schwert_max_lags;
use super::common::validate_series;

/// KPSS deterministic specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KpssTrend {
  /// Stationary around a constant mean.
  Level,
  /// Trend-stationary around a linear trend.
  Trend,
}

/// KPSS critical values.
#[derive(Debug, Clone, Copy)]
pub struct KpssCriticalValues {
  pub one_percent: f64,
  pub two_point_five_percent: f64,
  pub five_percent: f64,
  pub ten_percent: f64,
}

impl KpssCriticalValues {
  fn value_at(self, alpha: f64) -> f64 {
    if alpha <= 0.01 {
      self.one_percent
    } else if alpha <= 0.025 {
      self.two_point_five_percent
    } else if alpha <= 0.05 {
      self.five_percent
    } else {
      self.ten_percent
    }
  }
}

/// Configuration for the KPSS stationarity test.
#[derive(Debug, Clone, Copy)]
pub struct KpssConfig {
  /// Deterministic component under the null.
  pub trend: KpssTrend,
  /// Newey-West lag length. If `None`, a Schwert-style default is used.
  pub lags: Option<usize>,
  /// Significance level used to compute `reject_stationarity`.
  pub alpha: f64,
}

impl Default for KpssConfig {
  fn default() -> Self {
    Self {
      trend: KpssTrend::Level,
      lags: None,
      alpha: 0.05,
    }
  }
}

/// Result of the KPSS stationarity test.
#[derive(Debug, Clone, Copy)]
pub struct KpssResult {
  /// KPSS LM statistic.
  pub statistic: f64,
  /// Newey-West lag length used.
  pub used_lags: usize,
  /// Critical values for this trend specification.
  pub critical_values: KpssCriticalValues,
  /// Whether the null (stationarity) is rejected at `alpha`.
  pub reject_stationarity: bool,
}

impl crate::traits::HypothesisTest for KpssResult {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.reject_stationarity)
  }
}

fn kpss_critical_values(trend: KpssTrend) -> KpssCriticalValues {
  match trend {
    KpssTrend::Level => KpssCriticalValues {
      one_percent: 0.739,
      two_point_five_percent: 0.574,
      five_percent: 0.463,
      ten_percent: 0.347,
    },
    KpssTrend::Trend => KpssCriticalValues {
      one_percent: 0.216,
      two_point_five_percent: 0.176,
      five_percent: 0.146,
      ten_percent: 0.119,
    },
  }
}

/// KPSS stationarity test.
///
/// # Panics
/// Panics on invalid inputs (non-finite series, too-short sample, invalid config).
pub fn kpss_test(y: ArrayView1<f64>, cfg: KpssConfig) -> KpssResult {
  let y = y
    .as_slice()
    .expect("kpss_test requires a contiguous ArrayView1");
  validate_series(y, 20);
  assert!(
    cfg.alpha > 0.0 && cfg.alpha < 1.0,
    "alpha must be in (0, 1)"
  );

  let include_trend = matches!(cfg.trend, KpssTrend::Trend);
  let reg = regress_on_deterministics(y, include_trend);
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
  let long_run_var = newey_west_long_run_variance(&resid, used_lags).max(1e-12);
  let statistic = eta / long_run_var;

  let critical_values = kpss_critical_values(cfg.trend);
  let reject_stationarity = statistic > critical_values.value_at(cfg.alpha);

  KpssResult {
    statistic,
    used_lags,
    critical_values,
    reject_stationarity,
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::KpssConfig;
  use super::kpss_test;

  fn simulate_ar1(phi: f64, n: usize, seed: u64) -> Vec<f64> {
    let innovations = {
      let dist = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(seed));
      (0..n).map(|_| dist.sample_fast()).collect::<Vec<_>>()
    };

    let mut x = vec![0.0; n];
    for t in 1..n {
      x[t] = phi * x[t - 1] + innovations[t];
    }
    x
  }

  fn simulate_random_walk(n: usize, seed: u64) -> Vec<f64> {
    let innovations = {
      let dist = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(seed));
      (0..n).map(|_| dist.sample_fast()).collect::<Vec<_>>()
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
        kpss_test(ndarray::ArrayView1::from(&x), KpssConfig::default()),
      )
    });
    let ok = runs.iter().any(|(_, r)| !r.reject_stationarity);
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
    let res = kpss_test(ndarray::ArrayView1::from(&x), KpssConfig::default());
    assert!(
      res.reject_stationarity,
      "expected rejection for random walk, got {res:?}"
    );
  }
}
