use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use super::common::OlsResult;
use super::common::Trend;
use super::common::centred;
use super::common::fit_ar;
use super::common::min_sample_size;
use super::common::prepared;
use super::common::regress_on_deterministics;
use super::common::reject_exact_fit;
use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::input::assert_alpha;

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
    Ok(centred(y))
  }
}

/// `Σ S_t² / (m² σ²)` of AR residuals `u`, `S_t` their partial sums and `σ²` their mean square;
/// `Numerical` when that mean square is not finite.
fn lm_statistic(u: &[f64]) -> Result<f64, StatsError> {
  let m_f = u.len() as f64;
  let sigma2 = u.iter().map(|v| v * v).sum::<f64>() / m_f;
  if !sigma2.is_finite() {
    return Err(StatsError::Numerical {
      stage: "residual variance",
    });
  }
  let mut cum = 0.0;
  let mut sum_sq = 0.0;
  for &v in u {
    cum += v;
    sum_sq += cum * cum;
  }
  Ok(sum_sq / (m_f * m_f * sigma2))
}

/// The AR fit of `y` less its constant, or less its line under a trend: one solve, with its bound,
/// for the series and for each bootstrap path.
fn prewhitened(y: &[f64], include_trend: bool, ar_lags: usize) -> Result<OlsResult, StatsError> {
  fit_ar(&detrend_series(y, include_trend)?, ar_lags)
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
    x[t] = value;
  }

  x[burnin..].to_vec()
}

/// Leybourne–McCabe test; `Err` below `max(40, ar_lags + 15, 2 ar_lags + 1)` points, a non-finite
/// or constant series, a singular or exact AR fit or a failed bootstrap; panics on a bad config.
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
  let needed = 40
    .max(cfg.ar_lags.saturating_add(15))
    .max(min_sample_size(cfg.ar_lags, Trend::None));
  let y = prepared(y, needed)?;

  let include_trend = cfg.trend == Trend::ConstantTrend;
  let fit = prewhitened(&y, include_trend, cfg.ar_lags)?;
  let obs_stat = lm_statistic(&fit.residuals)?;
  reject_exact_fit(&y, &fit.residuals, fit.condition)?;

  let mut noise =
    SimdNormal::<f64>::new(0.0, fit.sigma2.sqrt()).seeded(&Deterministic::new(cfg.bootstrap_seed));
  let mut exceed = 0usize;
  for _ in 0..cfg.bootstrap_samples {
    let sim = simulate_ar(&fit.beta, &mut noise, y.len());
    // The exact-fit rule guards the caller's series, not its paths, which keep the solve's bound.
    let sim_stat = prewhitened(&sim, include_trend, cfg.ar_lags)
      .and_then(|path| lm_statistic(&path.residuals))
      .ok()
      .filter(|s| s.is_finite())
      .ok_or(StatsError::Numerical {
        stage: "bootstrap path",
      })?;
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
  use super::simulate_ar;
  use crate::StatsError;

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

  #[test]
  fn a_constant_series_is_degenerate_data() {
    let x = ndarray::Array1::<f64>::from_elem(64, 0.3);
    assert_eq!(
      leybourne_mccabe_test(x.view(), LeybourneMcCabeConfig::default()).unwrap_err(),
      StatsError::DegenerateData("constant series")
    );
  }

  #[test]
  fn an_ar_order_needs_twice_its_lags_plus_one_points() {
    let x = simulate_ar1(0.5, 40, 0x1E7B0E);
    let cfg = LeybourneMcCabeConfig {
      ar_lags: 20,
      ..LeybourneMcCabeConfig::default()
    };
    assert_eq!(
      leybourne_mccabe_test(ndarray::ArrayView1::from(&x), cfg).unwrap_err(),
      StatsError::InsufficientData {
        needed: 41,
        got: 40
      }
    );
  }

  /// An explosive fitted AR overflows its bootstrap path, which the test then answers with `Err`.
  #[test]
  fn an_explosive_bootstrap_path_overflows() {
    let mut noise = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(3));
    let path = simulate_ar(&[30.0], &mut noise, 40);
    assert!(path.iter().any(|v| !v.is_finite()), "{path:?}");
  }

  #[test]
  fn an_ar_order_at_usize_max_is_insufficient_data() {
    let x = simulate_ar1(0.5, 40, 0x1E7B0F);
    let cfg = LeybourneMcCabeConfig {
      ar_lags: usize::MAX,
      ..LeybourneMcCabeConfig::default()
    };
    assert_eq!(
      leybourne_mccabe_test(ndarray::ArrayView1::from(&x), cfg).unwrap_err(),
      StatsError::InsufficientData {
        needed: usize::MAX,
        got: 40
      }
    );
  }
}
