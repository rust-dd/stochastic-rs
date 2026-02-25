use super::common::newey_west_long_run_variance;
use super::common::regress_on_deterministics;
use super::common::schwert_max_lags;
use super::common::validate_series;

/// KPSS deterministic specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KPSSTrend {
  /// Stationary around a constant mean.
  Level,
  /// Trend-stationary around a linear trend.
  Trend,
}

/// KPSS critical values.
#[derive(Debug, Clone, Copy)]
pub struct KPSSCriticalValues {
  pub one_percent: f64,
  pub two_point_five_percent: f64,
  pub five_percent: f64,
  pub ten_percent: f64,
}

impl KPSSCriticalValues {
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
pub struct KPSSConfig {
  /// Deterministic component under the null.
  pub trend: KPSSTrend,
  /// Newey-West lag length. If `None`, a Schwert-style default is used.
  pub lags: Option<usize>,
  /// Significance level used to compute `reject_stationarity`.
  pub alpha: f64,
}

impl Default for KPSSConfig {
  fn default() -> Self {
    Self {
      trend: KPSSTrend::Level,
      lags: None,
      alpha: 0.05,
    }
  }
}

/// Result of the KPSS stationarity test.
#[derive(Debug, Clone, Copy)]
pub struct KPSSResult {
  /// KPSS LM statistic.
  pub statistic: f64,
  /// Newey-West lag length used.
  pub used_lags: usize,
  /// Critical values for this trend specification.
  pub critical_values: KPSSCriticalValues,
  /// Whether the null (stationarity) is rejected at `alpha`.
  pub reject_stationarity: bool,
}

fn kpss_critical_values(trend: KPSSTrend) -> KPSSCriticalValues {
  match trend {
    KPSSTrend::Level => KPSSCriticalValues {
      one_percent: 0.739,
      two_point_five_percent: 0.574,
      five_percent: 0.463,
      ten_percent: 0.347,
    },
    KPSSTrend::Trend => KPSSCriticalValues {
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
pub fn kpss_test(y: &[f64], cfg: KPSSConfig) -> KPSSResult {
  validate_series(y, 20);
  assert!(
    cfg.alpha > 0.0 && cfg.alpha < 1.0,
    "alpha must be in (0, 1)"
  );

  let include_trend = matches!(cfg.trend, KPSSTrend::Trend);
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

  KPSSResult {
    statistic,
    used_lags,
    critical_values,
    reject_stationarity,
  }
}

#[cfg(test)]
mod tests {
  use super::KPSSConfig;
  use super::kpss_test;
  use crate::distributions::normal::SimdNormal;

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
  fn kpss_keeps_stationarity_for_ar1() {
    let x = simulate_ar1(0.75, 2000);
    let res = kpss_test(&x, KPSSConfig::default());
    assert!(
      !res.reject_stationarity,
      "expected no rejection for stationary series, got {res:?}"
    );
  }

  #[test]
  fn kpss_rejects_stationarity_for_random_walk() {
    let x = simulate_random_walk(2000);
    let res = kpss_test(&x, KPSSConfig::default());
    assert!(
      res.reject_stationarity,
      "expected rejection for random walk, got {res:?}"
    );
  }
}
