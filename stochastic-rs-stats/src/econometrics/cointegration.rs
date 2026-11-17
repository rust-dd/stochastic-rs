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
use stochastic_rs_distributions::linalg::lstsq;

use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::input::assert_alpha;
use crate::stationarity::Trend;
use crate::stationarity::adf::AdfConfig;
use crate::stationarity::adf::adf_test;
use crate::stationarity::common::prepared_with_unit;
use crate::stationarity::common::reject_exact_fit;

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

/// Engle–Granger 2-step test; `Err` below 30 points, on non-finite or constant data, unequal
/// lengths, a singular or exact fit or the residual ADF's `Err`; panics unless `0.01 <= alpha < 1`.
pub fn engle_granger_test(
  y: ArrayView1<f64>,
  x: ArrayView1<f64>,
  cfg: EngleGrangerConfig,
) -> Result<EngleGrangerTest, StatsError> {
  assert_alpha(cfg.alpha);
  let critical_values = CriticalValues::new([(0.01, -3.96), (0.05, -3.37), (0.10, -3.07)]);
  let critical = critical_values.value_at(cfg.alpha);
  if x.len() != y.len() {
    return Err(StatsError::DimensionMismatch {
      expected: y.len(),
      got: x.len(),
    });
  }
  let (y, y_unit) = prepared_with_unit(y, 30)?;
  let (x, x_unit) = prepared_with_unit(x, 30)?;
  let (y, x) = (Array1::from(y), Array1::from(x));
  let n = y.len();
  let mut design = Array2::<f64>::zeros((n, 2));
  for i in 0..n {
    design[[i, 0]] = 1.0;
    design[[i, 1]] = x[i];
  }
  let sol = lstsq(design.view(), y.view());
  let intercept = sol[0];
  let beta = sol[1];
  if !(intercept.is_finite() && beta.is_finite()) {
    return Err(StatsError::Singular("OLS design"));
  }
  let mut residuals = Array1::<f64>::zeros(n);
  for i in 0..n {
    residuals[i] = y[i] - intercept - beta * x[i];
  }
  reject_exact_fit(y.as_slice().unwrap(), residuals.as_slice().unwrap(), 1.0)?;
  // The residuals' ADF has no terms, as statsmodels' `coint` and arch's `engle_granger` run it,
  // and as MacKinnon's (2010) and Phillips & Ouliaris's (1990) critical values assume.
  let cfg = AdfConfig {
    trend: Trend::None,
    ..AdfConfig::default()
  };
  let adf = adf_test(residuals.view(), cfg)?;
  Ok(EngleGrangerTest {
    statistic: adf.statistic,
    p_value: None,
    critical_values: Some(critical_values),
    rejects_null: adf.statistic < critical,
    intercept: intercept / y_unit,
    beta: beta * x_unit / y_unit,
    residuals: residuals / y_unit,
  })
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
    let res = engle_granger_test(y.view(), x.view(), EngleGrangerConfig::default()).unwrap();
    assert!(res.rejects_null);
    assert!((res.beta - 0.7).abs() < 0.05);
  }

  #[test]
  fn engle_granger_does_not_reject_independent_walks() {
    let x = random_walk(13, 500, 1.0);
    let y = random_walk(17, 500, 1.0);
    let res = engle_granger_test(y.view(), x.view(), EngleGrangerConfig::default()).unwrap();
    assert!(!res.rejects_null);
  }

  /// A constant `x` leaves the regression singular, and a constant `y` fits it exactly.
  #[test]
  fn either_series_constant_is_degenerate_data() {
    let (walk, constant) = (random_walk(13, 200, 1.0), Array1::from_elem(200, 0.3));
    for (y, x) in [(&walk, &constant), (&constant, &walk)] {
      assert_eq!(
        engle_granger_test(y.view(), x.view(), EngleGrangerConfig::default()).unwrap_err(),
        StatsError::DegenerateData("constant series")
      );
    }
  }

  /// Both tests check each series in full before the next.
  #[test]
  fn a_constant_y_beside_a_nan_in_x_is_degenerate_data_as_in_granger() {
    let y = Array1::from_elem(200, 0.3);
    let mut x = random_walk(13, 200, 1.0);
    x[10] = f64::NAN;
    let constant = StatsError::DegenerateData("constant series");
    assert_eq!(
      engle_granger_test(y.view(), x.view(), EngleGrangerConfig::default()).unwrap_err(),
      constant
    );
    assert_eq!(
      crate::econometrics::granger::granger_causality_test(y.view(), x.view(), 1, 0.05)
        .unwrap_err(),
      constant
    );
  }

  #[test]
  fn series_of_unequal_lengths_are_a_dimension_mismatch() {
    let (y, x) = (random_walk(13, 40, 1.0), random_walk(17, 41, 1.0));
    assert_eq!(
      engle_granger_test(y.view(), x.view(), EngleGrangerConfig::default()).unwrap_err(),
      StatsError::DimensionMismatch {
        expected: 40,
        got: 41
      }
    );
  }
}
