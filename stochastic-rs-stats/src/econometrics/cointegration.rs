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

pub use johansen::JohansenResult;
pub use johansen::Vecm;
pub use johansen::johansen_test;
pub use johansen::vecm_fit;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use crate::linalg::lstsq;
use crate::stationarity::adf::AdfConfig;
use crate::stationarity::adf::adf_test;

/// Result of an Engle-Granger 2-step test for $y_t = \alpha + \beta x_t + \varepsilon_t$.
#[derive(Debug, Clone)]
pub struct EngleGrangerResult {
  /// Estimated intercept.
  pub alpha: f64,
  /// Estimated cointegration coefficient.
  pub beta: f64,
  /// Estimated regression residuals.
  pub residuals: Array1<f64>,
  /// ADF statistic computed on the residuals.
  pub adf_statistic: f64,
  /// 1%, 5%, 10% critical values for the residual ADF test (Phillips-Ouliaris,
  /// finite-sample).
  pub critical_values: (f64, f64, f64),
  /// Whether the no-cointegration null is rejected at `alpha = 0.05`.
  pub reject_no_cointegration: bool,
}

/// Engle-Granger 2-step cointegration test.
pub fn engle_granger_test(y: ArrayView1<f64>, x: ArrayView1<f64>) -> EngleGrangerResult {
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
  let alpha = sol[0];
  let beta = sol[1];
  let mut residuals = Array1::<f64>::zeros(n);
  for i in 0..n {
    residuals[i] = y[i] - alpha - beta * x[i];
  }
  let cfg = AdfConfig::default();
  let adf = adf_test(residuals.view(), cfg);
  let crit = phillips_ouliaris_critical_values_two_var();
  let reject = adf.statistic < crit.1;
  EngleGrangerResult {
    alpha,
    beta,
    residuals,
    adf_statistic: adf.statistic,
    critical_values: crit,
    reject_no_cointegration: reject,
  }
}

fn phillips_ouliaris_critical_values_two_var() -> (f64, f64, f64) {
  (-3.96, -3.37, -3.07)
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  fn random_walk(seed: u64, n: usize, sigma: f64) -> Array1<f64> {
    let dist = SimdNormal::<f64>::new(0.0, sigma, &Deterministic::new(seed));
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
    let dist = SimdNormal::<f64>::new(0.0, 0.05, &Deterministic::new(11));
    let mut eps = vec![0.0_f64; 500];
    dist.fill_slice(&mut eps);
    let mut y = Array1::<f64>::zeros(500);
    for i in 0..500 {
      y[i] = 2.0 + 0.7 * x[i] + eps[i];
    }
    let res = engle_granger_test(y.view(), x.view());
    assert!(res.reject_no_cointegration);
    assert!((res.beta - 0.7).abs() < 0.05);
  }

  #[test]
  fn engle_granger_does_not_reject_independent_walks() {
    let x = random_walk(13, 500, 1.0);
    let y = random_walk(17, 500, 1.0);
    let res = engle_granger_test(y.view(), x.view());
    assert!(!res.reject_no_cointegration);
  }
}
