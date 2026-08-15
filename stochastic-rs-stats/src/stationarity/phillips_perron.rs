//! Phillips-Perron unit-root test — corrects the (lag-0) Dickey-Fuller
//! regression for serial correlation and heteroskedasticity via a
//! Newey-West long-run variance estimate instead of augmenting the
//! regression with lagged differences. Implements both the `Tau`
//! (t-ratio-style) and `Rho` variants of the test statistic.
//!
//! References:
//! - Phillips P. C. B. & Perron P. (1988) — *Testing for a Unit Root in
//!   Time Series Regression*, Biometrika 75(2), 335–346,
//!   DOI: 10.1093/biomet/75.2.335.
//! - Newey W. K. & West K. D. (1987) — *A Simple, Positive
//!   Semi-Definite, Heteroskedasticity and Autocorrelation Consistent
//!   Covariance Matrix*, Econometrica 55(3), 703–708,
//!   DOI: 10.2307/1913610.

use ndarray::ArrayView1;

use super::common::CriticalValues;
use super::common::DeterministicTerm;
use super::common::adf_critical_values;
use super::common::fit_adf;
use super::common::newey_west_long_run_variance;
use super::common::schwert_max_lags;
use super::common::validate_series;

/// Phillips-Perron test statistic variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PpTestType {
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
  pub test_type: PpTestType,
  /// Newey-West lag length. If `None`, a Schwert-style default is used.
  pub lags: Option<usize>,
  /// Significance level for decision output (used for `Tau`).
  pub alpha: f64,
}

impl Default for PhillipsPerronConfig {
  fn default() -> Self {
    Self {
      deterministic: DeterministicTerm::Constant,
      test_type: PpTestType::Tau,
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
  pub test_type: PpTestType,
  /// Critical values (available for `Tau` output).
  pub critical_values: Option<CriticalValues>,
  /// Unit-root rejection decision (available for `Tau` output).
  pub reject_unit_root: Option<bool>,
}

impl crate::traits::HypothesisTest for PhillipsPerronResult {
  fn statistic(&self) -> f64 {
    self.statistic
  }
  fn null_rejected(&self) -> Option<bool> {
    self.reject_unit_root
  }
}

/// Phillips-Perron unit-root test.
///
/// # Panics
/// Panics on invalid inputs (non-finite series, too-short sample, invalid config).
pub fn phillips_perron_test(y: ArrayView1<f64>, cfg: PhillipsPerronConfig) -> PhillipsPerronResult {
  let y = y
    .as_slice()
    .expect("phillips_perron_test requires a contiguous ArrayView1");
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
    PpTestType::Rho => {
      n_f * (rho - 1.0) - 0.5 * ((n_f * n_f * fit.sigma2 / s2.max(1e-12)) * (lam2 - s2))
    }
    PpTestType::Tau => {
      let se = fit.std_err_gamma.max(1e-12);
      (gamma0 / lam2).sqrt() * ((rho - 1.0) / se) - 0.5 * ((lam2 - s2) / lam) * (n_f * se / s)
    }
  };

  let (critical_values, reject_unit_root) = match cfg.test_type {
    PpTestType::Tau => {
      let cvals = adf_critical_values(cfg.deterministic);
      (Some(cvals), Some(statistic < cvals.value_at(cfg.alpha)))
    }
    PpTestType::Rho => (None, None),
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
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::PhillipsPerronConfig;
  use super::PpTestType;
  use super::phillips_perron_test;
  use crate::stationarity::common::DeterministicTerm;

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
  fn pp_tau_rejects_stationary_ar1() {
    let x = simulate_ar1(0.75, 2500, 0xA11CE);
    let cfg = PhillipsPerronConfig {
      deterministic: DeterministicTerm::Constant,
      test_type: PpTestType::Tau,
      lags: Some(12),
      ..PhillipsPerronConfig::default()
    };
    let res = phillips_perron_test(ndarray::ArrayView1::from(&x), cfg);
    assert_eq!(
      res.reject_unit_root,
      Some(true),
      "expected rejection, got {res:?}"
    );
  }

  #[test]
  fn pp_tau_random_walk_is_not_rejected_at_one_percent() {
    let x = simulate_random_walk(2500, 0xBADC0DE);
    let cfg = PhillipsPerronConfig {
      deterministic: DeterministicTerm::Constant,
      test_type: PpTestType::Tau,
      lags: Some(12),
      alpha: 0.01,
    };
    let res = phillips_perron_test(ndarray::ArrayView1::from(&x), cfg);
    assert_eq!(
      res.reject_unit_root,
      Some(false),
      "expected no rejection, got {res:?}"
    );
  }
}
