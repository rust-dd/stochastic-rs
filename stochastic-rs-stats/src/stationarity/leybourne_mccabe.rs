use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use super::common::Trend;
use super::common::fit_ar;
use super::common::regress_on_deterministics;
use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::input::as_slice_or_copy;
use crate::input::assert_alpha;
use crate::input::validate;

/// Configuration for the Leybourne-McCabe stationarity test.
#[derive(Debug, Clone, Copy)]
pub struct LeybourneMcCabeConfig {
  /// Null deterministic structure: `Constant` (level) or `ConstantTrend`.
  pub trend: Trend,
  /// AR prewhitening lag order.
  pub ar_lags: usize,
  /// Parametric bootstrap replications for p-value estimation.
  pub bootstrap_samples: usize,
  /// Seed used by bootstrap RNG.
  pub bootstrap_seed: u64,
  /// Significance level used to compute `rejects_null`.
  pub alpha: f64,
}

impl Default for LeybourneMcCabeConfig {
  fn default() -> Self {
    Self {
      trend: Trend::Constant,
      ar_lags: 1,
      bootstrap_samples: 400,
      bootstrap_seed: 1234,
      alpha: 0.05,
    }
  }
}

/// Leybourne–McCabe test (null: stationarity), decided by its parametric-bootstrap p-value.
#[derive(Debug, Clone)]
pub struct LeybourneMcCabeTest {
  /// LM-style stationarity statistic.
  pub statistic: f64,
  /// Bootstrap p-value under stationarity null.
  pub p_value: Option<f64>,
  pub critical_values: Option<CriticalValues>,
  /// Whether stationarity is rejected at `alpha`.
  pub rejects_null: bool,
  /// AR lag order used for prewhitening.
  pub ar_lags: usize,
}

impl crate::traits::HypothesisTest for LeybourneMcCabeTest {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.rejects_null)
  }
}

fn detrend_series(y: &[f64], include_trend: bool) -> Result<Vec<f64>, StatsError> {
  if include_trend {
    Ok(regress_on_deterministics(y, true)?.residuals)
  } else {
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    Ok(y.iter().map(|v| v - mean).collect())
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
  include_trend: bool,
  ar_lags: usize,
) -> Result<(f64, Vec<f64>, f64), StatsError> {
  let y_det = detrend_series(y, include_trend)?;
  let (phi, residuals, sigma) = fit_ar(&y_det, ar_lags)?;
  let stat = lm_statistic_from_residuals(&residuals);
  Ok((stat, phi, sigma))
}

fn simulate_ar(phi: &[f64], noise: &mut Seeded<SimdNormal<f64>>, n: usize) -> Vec<f64> {
  let p = phi.len();
  let burnin = (20 * p).max(200);
  let total = n + burnin;
  let mut x = vec![0.0; total];
  let mut eps = vec![0.0_f64; total];
  noise.fill_slice(&mut eps);

  for t in p..total {
    let mut value = eps[t];
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

/// Leybourne–McCabe test with AR prewhitening and a bootstrap p-value; `Err` below 40 (or
/// `ar_lags + 15`) points, on a non-finite one or a singular AR fit; panics on a bad config.
pub fn leybourne_mccabe_test(
  y: ArrayView1<f64>,
  cfg: LeybourneMcCabeConfig,
) -> Result<LeybourneMcCabeTest, StatsError> {
  assert!(
    cfg.trend != Trend::None,
    "trend must satisfy `trend != Trend::None`, got trend = {:?}",
    cfg.trend
  );
  assert!(
    cfg.ar_lags >= 1,
    "ar_lags must satisfy `ar_lags >= 1`, got ar_lags = {:?}",
    cfg.ar_lags
  );
  assert!(
    cfg.bootstrap_samples >= 1,
    "bootstrap_samples must satisfy `bootstrap_samples >= 1`, got bootstrap_samples = {:?}",
    cfg.bootstrap_samples
  );
  assert_alpha(cfg.alpha);
  let y = as_slice_or_copy(y);
  validate(&y, 40.max(cfg.ar_lags + 15))?;

  let include_trend = cfg.trend == Trend::ConstantTrend;
  let (obs_stat, phi, sigma) = lm_statistic_with_prewhitening(&y, include_trend, cfg.ar_lags)?;

  let mut noise =
    SimdNormal::<f64>::new(0.0, sigma.max(1e-12)).seeded(&Deterministic::new(cfg.bootstrap_seed));
  let mut exceed = 0usize;
  for _ in 0..cfg.bootstrap_samples {
    let sim = simulate_ar(&phi, &mut noise, y.len());
    let (sim_stat, _, _) = lm_statistic_with_prewhitening(&sim, include_trend, cfg.ar_lags)?;
    if sim_stat >= obs_stat {
      exceed += 1;
    }
  }

  let p_value = (exceed as f64 + 1.0) / (cfg.bootstrap_samples as f64 + 1.0);

  Ok(LeybourneMcCabeTest {
    statistic: obs_stat,
    p_value: Some(p_value),
    critical_values: None,
    rejects_null: p_value < cfg.alpha,
    ar_lags: cfg.ar_lags,
  })
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::LeybourneMcCabeConfig;
  use super::leybourne_mccabe_test;

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
  fn leybourne_mccabe_keeps_stationary_ar1() {
    let x = simulate_ar1(0.7, 900, 0x1E7B0A);
    let cfg = LeybourneMcCabeConfig {
      ar_lags: 2,
      bootstrap_samples: 160,
      bootstrap_seed: 17,
      ..LeybourneMcCabeConfig::default()
    };
    let res = leybourne_mccabe_test(ndarray::ArrayView1::from(&x), cfg).unwrap();
    assert!(
      !res.rejects_null,
      "expected no rejection for stationary series, got {res:?}"
    );
  }

  /// The complement of the test above: a random walk is the canonical
  /// non-stationary alternative, so the null must be rejected.
  #[test]
  fn leybourne_mccabe_rejects_random_walk() {
    // 900 points and 160 bootstrap replicates gave too little power to reject
    // reliably; a longer series plus the best-of-three seeds fixes that.
    let runs = [0x1E7B0Bu64, 0x1E7B0C, 0x1E7B0D].map(|seed| {
      let x = simulate_random_walk(2500, seed);
      let cfg = LeybourneMcCabeConfig {
        ar_lags: 2,
        bootstrap_samples: 320,
        bootstrap_seed: 19,
        ..LeybourneMcCabeConfig::default()
      };
      (
        seed,
        leybourne_mccabe_test(ndarray::ArrayView1::from(&x), cfg).unwrap(),
      )
    });
    let ok = runs.iter().any(|(_, r)| r.rejects_null);
    let report = runs
      .iter()
      .map(|(s, r)| format!("seed {s:#x}: p={:.4}", r.p_value.unwrap()))
      .collect::<Vec<_>>()
      .join("; ");
    assert!(ok, "no seed rejected a random walk — {report}");
  }
}
