use ndarray::ArrayView1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::Normal;

use super::common::fit_ar;
use super::common::regress_on_deterministics;
use super::common::validate_series;

/// Deterministic specification for Leybourne-McCabe style testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LMTrend {
  /// Stationary around a fixed mean.
  Level,
  /// Trend-stationary around a linear trend.
  Trend,
}

/// Configuration for the Leybourne-McCabe stationarity test.
#[derive(Debug, Clone, Copy)]
pub struct LeybourneMcCabeConfig {
  /// Null deterministic structure.
  pub trend: LMTrend,
  /// AR prewhitening lag order.
  pub ar_lags: usize,
  /// Parametric bootstrap replications for p-value estimation.
  pub bootstrap_samples: usize,
  /// Seed used by bootstrap RNG.
  pub bootstrap_seed: u64,
  /// Significance level used to compute `reject_stationarity`.
  pub alpha: f64,
}

impl Default for LeybourneMcCabeConfig {
  fn default() -> Self {
    Self {
      trend: LMTrend::Level,
      ar_lags: 1,
      bootstrap_samples: 400,
      bootstrap_seed: 1234,
      alpha: 0.05,
    }
  }
}

/// Result of the Leybourne-McCabe stationarity test.
#[derive(Debug, Clone, Copy)]
pub struct LeybourneMcCabeResult {
  /// LM-style stationarity statistic.
  pub statistic: f64,
  /// AR lag order used for prewhitening.
  pub ar_lags: usize,
  /// Bootstrap p-value under stationarity null.
  pub p_value: f64,
  /// Whether stationarity is rejected at `alpha`.
  pub reject_stationarity: bool,
}

impl crate::traits::HypothesisTest for LeybourneMcCabeResult {
  fn statistic(&self) -> f64 {
    self.statistic
  }
  fn null_rejected(&self) -> Option<bool> {
    Some(self.reject_stationarity)
  }
}

fn detrend_series(y: &[f64], trend: LMTrend) -> Vec<f64> {
  match trend {
    LMTrend::Level => {
      let mean = y.iter().sum::<f64>() / y.len() as f64;
      y.iter().map(|v| v - mean).collect()
    }
    LMTrend::Trend => regress_on_deterministics(y, true).residuals,
  }
}

fn lm_statistic_from_residuals(residuals: &[f64]) -> f64 {
  let m = residuals.len();
  let m_f = m as f64;
  let sigma2 = residuals.iter().map(|u| u * u).sum::<f64>() / m_f;
  if sigma2 <= 0.0 || !sigma2.is_finite() {
    return f64::INFINITY;
  }

  let mut cum = 0.0;
  let mut sum_sq = 0.0;
  for &u in residuals {
    cum += u;
    sum_sq += cum * cum;
  }
  sum_sq / (m_f * m_f * sigma2)
}

fn lm_statistic_with_prewhitening(
  y: &[f64],
  trend: LMTrend,
  ar_lags: usize,
) -> (f64, Vec<f64>, f64) {
  let y_det = detrend_series(y, trend);
  let (phi, residuals, sigma) = fit_ar(&y_det, ar_lags);
  let stat = lm_statistic_from_residuals(&residuals);
  (stat, phi, sigma)
}

fn simulate_ar(phi: &[f64], sigma: f64, n: usize, rng: &mut StdRng) -> Vec<f64> {
  let p = phi.len();
  let burnin = (20 * p).max(200);
  let total = n + burnin;
  let mut x = vec![0.0; total];

  let noise = Normal::new(0.0, sigma.max(1e-12)).expect("normal params must be valid");

  for t in p..total {
    let mut value = noise.sample(rng);
    for i in 0..p {
      value += phi[i] * x[t - 1 - i];
    }
    if !value.is_finite() {
      value = 0.0;
    }
    value = value.clamp(-1e6, 1e6);
    x[t] = value;
  }

  x[burnin..].to_vec()
}

/// Leybourne-McCabe style stationarity test with AR prewhitening and bootstrap p-value.
///
/// # Panics
/// Panics on invalid inputs (non-finite series, too-short sample, invalid config).
pub fn leybourne_mccabe_test(
  y: ArrayView1<f64>,
  cfg: LeybourneMcCabeConfig,
) -> LeybourneMcCabeResult {
  let y = y
    .as_slice()
    .expect("leybourne_mccabe_test requires a contiguous ArrayView1");
  validate_series(y, 40);
  assert!(cfg.ar_lags > 0, "ar_lags must be positive");
  assert!(
    cfg.bootstrap_samples > 0,
    "bootstrap_samples must be positive"
  );
  assert!(
    cfg.alpha > 0.0 && cfg.alpha < 1.0,
    "alpha must be in (0, 1)"
  );

  let min_required = cfg.ar_lags + 15;
  assert!(
    y.len() >= min_required,
    "series is too short for the requested AR lag order"
  );

  let (obs_stat, phi, sigma) = lm_statistic_with_prewhitening(y, cfg.trend, cfg.ar_lags);

  let mut rng = StdRng::seed_from_u64(cfg.bootstrap_seed);
  let mut exceed = 0usize;
  for _ in 0..cfg.bootstrap_samples {
    let sim = simulate_ar(&phi, sigma, y.len(), &mut rng);
    let (sim_stat, _, _) = lm_statistic_with_prewhitening(&sim, cfg.trend, cfg.ar_lags);
    if sim_stat >= obs_stat {
      exceed += 1;
    }
  }

  let p_value = (exceed as f64 + 1.0) / (cfg.bootstrap_samples as f64 + 1.0);

  LeybourneMcCabeResult {
    statistic: obs_stat,
    ar_lags: cfg.ar_lags,
    p_value,
    reject_stationarity: p_value < cfg.alpha,
  }
}

#[cfg(test)]
mod tests {
  use rand::SeedableRng;
  use rand::rngs::StdRng;
  use rand_distr::Distribution;
  use rand_distr::Normal;

  use super::LeybourneMcCabeConfig;
  use super::leybourne_mccabe_test;

  fn simulate_ar1(phi: f64, n: usize, seed: u64) -> Vec<f64> {
    let innovations = {
      let dist = Normal::new(0.0, 1.0).unwrap();
      let mut rng = StdRng::seed_from_u64(seed);
      (0..n).map(|_| dist.sample(&mut rng)).collect::<Vec<_>>()
    };

    let mut x = vec![0.0; n];
    for t in 1..n {
      x[t] = phi * x[t - 1] + innovations[t];
    }
    x
  }

  fn simulate_random_walk(n: usize, seed: u64) -> Vec<f64> {
    let innovations = {
      let dist = Normal::new(0.0, 1.0).unwrap();
      let mut rng = StdRng::seed_from_u64(seed);
      (0..n).map(|_| dist.sample(&mut rng)).collect::<Vec<_>>()
    };

    let mut x = vec![0.0; n];
    for t in 1..n {
      x[t] = x[t - 1] + innovations[t];
    }
    x
  }

  #[test]
  fn leybourne_mccabe_keeps_stationary_ar1() {
    let x = simulate_ar1(0.7, 900, 0x1E7B0A);
    let cfg = LeybourneMcCabeConfig {
      ar_lags: 2,
      bootstrap_samples: 160,
      bootstrap_seed: 17,
      ..LeybourneMcCabeConfig::default()
    };
    let res = leybourne_mccabe_test(ndarray::ArrayView1::from(&x), cfg);
    assert!(
      !res.reject_stationarity,
      "expected no rejection for stationary series, got {res:?}"
    );
  }
}
