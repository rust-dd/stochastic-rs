use super::common::CriticalValues;
use super::common::DeterministicTerm;
use super::common::adf_critical_values;
use super::common::fit_adf;
use super::common::newey_west_long_run_variance;
use super::common::schwert_max_lags;
use super::common::validate_series;

/// Phillips-Perron test statistic variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PPTestType {
  /// Z-tau (t-ratio style) statistic.
  Tau,
  /// Z-rho statistic.
  Rho,
}

/// Configuration for the Phillips-Perron unit-root test.
#[derive(Debug, Clone, Copy)]
pub struct PhillipsPerronConfig {
  /// Deterministic terms in the test regression.
  pub deterministic: DeterministicTerm,
  /// PP statistic type.
  pub test_type: PPTestType,
  /// Newey-West lag length. If `None`, a Schwert-style default is used.
  pub lags: Option<usize>,
  /// Significance level for decision output (used for `Tau`).
  pub alpha: f64,
}

impl Default for PhillipsPerronConfig {
  fn default() -> Self {
    Self {
      deterministic: DeterministicTerm::Constant,
      test_type: PPTestType::Tau,
      lags: None,
      alpha: 0.05,
    }
  }
}

/// Result of the Phillips-Perron unit-root test.
#[derive(Debug, Clone, Copy)]
pub struct PhillipsPerronResult {
  /// PP statistic value.
  pub statistic: f64,
  /// Newey-West lag length used.
  pub used_lags: usize,
  /// Variant of the PP statistic.
  pub test_type: PPTestType,
  /// Critical values (available for `Tau` output).
  pub critical_values: Option<CriticalValues>,
  /// Unit-root rejection decision (available for `Tau` output).
  pub reject_unit_root: Option<bool>,
}

/// Phillips-Perron unit-root test.
///
/// # Panics
/// Panics on invalid inputs (non-finite series, too-short sample, invalid config).
pub fn phillips_perron_test(y: &[f64], cfg: PhillipsPerronConfig) -> PhillipsPerronResult {
  validate_series(y, 20);
  assert!(
    cfg.alpha > 0.0 && cfg.alpha < 1.0,
    "alpha must be in (0, 1)"
  );

  let fit = fit_adf(y, 0, cfg.deterministic);
  let u = fit.residuals;
  let nobs = u.len();
  let n_f = nobs as f64;

  let s2 = u.iter().map(|v| v * v).sum::<f64>() / n_f;
  let s = s2.max(1e-12).sqrt();
  let gamma0 = s2;

  let used_lags = cfg.lags.unwrap_or_else(|| schwert_max_lags(y.len()));
  let lam2 = newey_west_long_run_variance(&u, used_lags).max(1e-12);
  let lam = lam2.sqrt();

  // In the ADF parameterization, gamma = rho - 1.
  let rho = 1.0 + fit.gamma;

  let statistic = match cfg.test_type {
    PPTestType::Rho => {
      n_f * (rho - 1.0) - 0.5 * ((n_f * n_f * fit.sigma2 / s2.max(1e-12)) * (lam2 - s2))
    }
    PPTestType::Tau => {
      let se = fit.std_err_gamma.max(1e-12);
      (gamma0 / lam2).sqrt() * ((rho - 1.0) / se) - 0.5 * ((lam2 - s2) / lam) * (n_f * se / s)
    }
  };

  let (critical_values, reject_unit_root) = match cfg.test_type {
    PPTestType::Tau => {
      let cvals = adf_critical_values(cfg.deterministic);
      (Some(cvals), Some(statistic < cvals.value_at(cfg.alpha)))
    }
    PPTestType::Rho => (None, None),
  };

  PhillipsPerronResult {
    statistic,
    used_lags,
    test_type: cfg.test_type,
    critical_values,
    reject_unit_root,
  }
}

#[cfg(test)]
mod tests {
  use super::PPTestType;
  use super::PhillipsPerronConfig;
  use super::phillips_perron_test;
  use crate::distributions::normal::SimdNormal;
  use crate::stats::stationarity::common::DeterministicTerm;

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
  fn pp_tau_rejects_stationary_ar1() {
    let x = simulate_ar1(0.75, 2500);
    let cfg = PhillipsPerronConfig {
      deterministic: DeterministicTerm::Constant,
      test_type: PPTestType::Tau,
      lags: Some(12),
      ..PhillipsPerronConfig::default()
    };
    let res = phillips_perron_test(&x, cfg);
    assert_eq!(
      res.reject_unit_root,
      Some(true),
      "expected rejection, got {res:?}"
    );
  }

  #[test]
  fn pp_tau_keeps_unit_root_for_random_walk() {
    let x = simulate_random_walk(2500);
    let cfg = PhillipsPerronConfig {
      deterministic: DeterministicTerm::Constant,
      test_type: PPTestType::Tau,
      lags: Some(12),
      ..PhillipsPerronConfig::default()
    };
    let res = phillips_perron_test(&x, cfg);
    assert_eq!(
      res.reject_unit_root,
      Some(false),
      "expected no rejection, got {res:?}"
    );
  }
}
