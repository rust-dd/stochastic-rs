//! Cointegration tests — Engle-Granger 2-step here, the Johansen rank tests
//! and VECM estimation in [`johansen`].
//!
//! Reference: Engle, Granger, "Co-Integration and Error Correction:
//! Representation, Estimation, and Testing", Econometrica, 55(2), 251-276
//! (1987). DOI: 10.2307/1913236
//!
//! Reference: Johansen, "Statistical Analysis of Cointegration Vectors",
//! Journal of Economic Dynamics and Control, 12(2-3), 231-254 (1988).
//! DOI: 10.1016/0165-1889(88)90041-3

pub mod johansen;

pub use johansen::JohansenTest;
pub use johansen::VecmFit;
pub use johansen::fit_vecm;
pub use johansen::johansen_test;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use crate::critical_values::CriticalValues;
use crate::linalg::lstsq;
use crate::stationarity::adf::AdfConfig;
use crate::stationarity::adf::adf_test;

/// Configuration for [`engle_granger_test`].
#[derive(Debug, Clone, Copy)]
pub struct EngleGrangerConfig {
  /// Significance level used to compute `rejects_null`.
  pub alpha: f64,
}

impl Default for EngleGrangerConfig {
  fn default() -> Self {
    Self { alpha: 0.05 }
  }
}

/// Engle–Granger 2-step test for $y_t = a + \beta x_t + \varepsilon_t$ (null: no cointegration);
/// the Phillips–Ouliaris table holds the 1 %, 5 % and 10 % levels.
#[derive(Debug, Clone)]
pub struct EngleGrangerTest {
  /// ADF statistic computed on the residuals.
  pub statistic: f64,
  /// No p-value table is shipped; the decision is taken against `critical_values`.
  pub p_value: Option<f64>,
  pub critical_values: Option<CriticalValues>,
  /// Whether the no-cointegration null is rejected at `alpha`.
  pub rejects_null: bool,
  /// Estimated intercept $a$.
  pub intercept: f64,
  /// Estimated cointegration coefficient.
  pub beta: f64,
  /// Estimated regression residuals.
  pub residuals: Array1<f64>,
}

impl crate::traits::HypothesisTest for EngleGrangerTest {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.rejects_null)
  }
}

/// Engle-Granger 2-step cointegration test.
pub fn engle_granger_test(
  y: ArrayView1<f64>,
  x: ArrayView1<f64>,
  cfg: EngleGrangerConfig,
) -> EngleGrangerTest {
  assert!(
    cfg.alpha > 0.0 && cfg.alpha < 1.0,
    "alpha must satisfy `0.0 < alpha < 1.0`, got alpha = {:?}",
    cfg.alpha
  );
  let n = y.len();
  assert_eq!(n, x.len(), "y and x must have equal length");
  assert!(n >= 30, "need at least 30 observations");
  let mut design = Array2::<f64>::zeros((n, 2));
  for i in 0..n {
    design[[i, 0]] = 1.0;
    design[[i, 1]] = x[i];
  }
  let y_owned = y.to_owned();
  let sol = lstsq(&design, &y_owned);
  let intercept = sol[0];
  let beta = sol[1];
  let mut residuals = Array1::<f64>::zeros(n);
  for i in 0..n {
    residuals[i] = y[i] - intercept - beta * x[i];
  }
  let adf = adf_test(residuals.view(), AdfConfig::default());
  let critical_values = CriticalValues::new([(0.01, -3.96), (0.05, -3.37), (0.10, -3.07)]);
  let rejects_null = adf.statistic < critical_values.value_at(cfg.alpha);
  EngleGrangerTest {
    statistic: adf.statistic,
    p_value: None,
    critical_values: Some(critical_values),
    rejects_null,
    intercept,
    beta,
    residuals,
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::DistributionSampler;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  fn random_walk(seed: u64, n: usize, sigma: f64) -> Array1<f64> {
    let mut dist = SimdNormal::<f64>::new(0.0, sigma).seeded(&Deterministic::new(seed));
    let mut steps = vec![0.0_f64; n];
    dist.fill_slice(&mut steps);
    let mut out = Array1::<f64>::zeros(n);
    for i in 1..n {
      out[i] = out[i - 1] + steps[i];
    }
    out
  }

  #[test]
  fn engle_granger_rejects_under_cointegration() {
    let x = random_walk(7, 500, 1.0);
    let mut dist = SimdNormal::<f64>::new(0.0, 0.05).seeded(&Deterministic::new(11));
    let mut eps = vec![0.0_f64; 500];
    dist.fill_slice(&mut eps);
    let mut y = Array1::<f64>::zeros(500);
    for i in 0..500 {
      y[i] = 2.0 + 0.7 * x[i] + eps[i];
    }
    let res = engle_granger_test(y.view(), x.view(), EngleGrangerConfig::default());
    assert!(res.rejects_null);
    assert!((res.beta - 0.7).abs() < 0.05);
  }

  #[test]
  fn engle_granger_does_not_reject_independent_walks() {
    let x = random_walk(13, 500, 1.0);
    let y = random_walk(17, 500, 1.0);
    let res = engle_granger_test(y.view(), x.view(), EngleGrangerConfig::default());
    assert!(!res.rejects_null);
  }
}
