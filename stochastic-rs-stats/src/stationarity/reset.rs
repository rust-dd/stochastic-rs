//! Ramsey (1969) RESET, the regression equation specification error test of functional form.
//!
//! $$
//! y_t = x_t^\top \beta + \sum_{j=2}^{p} \gamma_j \hat y_t^{\,j} + \varepsilon_t,\qquad
//! H_0: \gamma_2 = \dots = \gamma_p = 0
//! $$
//!
//! $$
//! F = \frac{(\mathrm{SSR}_r - \mathrm{SSR}_u)/q}{\mathrm{SSR}_u/(T - k - q)}
//! \overset{H_0}{\sim} F(q, T - k - q),\qquad q = p - 1
//! $$
//!
//! `ŷ` are the fitted values of `y` on `x`; the default powers, 2 and 3, are Ramsey's own.
//!
//! Reference: Ramsey (1969), JRSS B 31(2), 350-371, DOI: 10.1111/j.2517-6161.1969.tb00796.x

use ndarray::ArrayView1;
use ndarray::ArrayView2;
use stochastic_rs_distributions::special::beta_i;

use super::common::centred;
use super::common::nested_f;
use super::common::ols;
use super::common::prepared;
use super::common::reject_exact_fit;
use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::input::assert_alpha;
use crate::input::validate_rows;

/// $F(d_1, d_2)$ upper tail probability $1 - F(x)$ via the regularised
/// incomplete beta identity
/// $$
/// P(F \le x; d_1, d_2) = I_{d_1 x/(d_1 x + d_2)}(d_1/2, d_2/2).
/// $$
fn f_upper_tail(x: f64, d1: f64, d2: f64) -> f64 {
  if !x.is_finite() || x <= 0.0 {
    return 1.0;
  }
  let z = (d1 * x) / (d1 * x + d2);
  (1.0 - beta_i(d1 * 0.5, d2 * 0.5, z)).clamp(0.0, 1.0)
}

/// Configuration of the RESET test, which panics unless `max_power >= 2` and `0 < alpha < 1`.
#[derive(Debug, Clone, Copy)]
pub struct ResetConfig {
  /// Highest power of $\hat y$ added to the augmented regression. Must be
  /// $\ge 2$. Default `3` ($\hat y^2$ and $\hat y^3$).
  pub max_power: usize,
  /// Significance level used to compute `rejects_null`.
  pub alpha: f64,
}

impl Default for ResetConfig {
  fn default() -> Self {
    Self {
      max_power: 3,
      alpha: 0.05,
    }
  }
}

/// Ramsey RESET (null: a correct linear specification), decided by its F p-value.
#[derive(Debug, Clone)]
pub struct ResetTest {
  /// F-statistic for the joint restriction $\gamma_2 = \dots = \gamma_p = 0$.
  pub statistic: f64,
  /// Asymptotic p-value from the $F(q, T-k-q)$ distribution.
  pub p_value: Option<f64>,
  pub critical_values: Option<CriticalValues>,
  /// Whether the correct-specification null is rejected at `alpha`.
  pub rejects_null: bool,
  /// Numerator degrees of freedom $q = p - 1$.
  pub df_num: usize,
  /// Denominator degrees of freedom $T - k - q$.
  pub df_den: usize,
}

impl crate::traits::HypothesisTest for ResetTest {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.rejects_null)
  }
}

/// Ramsey (1969) RESET; `Err` below `k + q + 2` points, a non-finite entry, constant `y`,
/// mismatched rows, a singular or exact fit, a non-finite `F`; panics as [`ResetConfig`] says.
pub fn reset_test(
  y: ArrayView1<f64>,
  x: ArrayView2<f64>,
  cfg: ResetConfig,
) -> Result<ResetTest, StatsError> {
  assert!(
    cfg.max_power >= 2,
    "max_power must satisfy `max_power >= 2`, got max_power = {:?}",
    cfg.max_power
  );
  assert_alpha(cfg.alpha);
  if x.nrows() != y.len() {
    return Err(StatsError::DimensionMismatch {
      expected: y.len(),
      got: x.nrows(),
    });
  }
  let t_total = y.len();
  let k = x.ncols();
  let q = cfg.max_power - 1;
  // `y` at a power of two, which `F` does not see, keeps the powers of `ŷ` in range.
  let y_slice = prepared(y, k + q + 2)?;
  validate_rows(x)?;
  // A constant column absorbs `y`'s mean, so `F` is that of `y` less it: an exact subtraction,
  // which keeps the digits the level would round away.
  let constant = has_constant_column(x);
  let y_slice = if constant { centred(&y_slice) } else { y_slice };

  // Restricted OLS: y on x.
  let x_rows: Vec<Vec<f64>> = (0..t_total).map(|i| x.row(i).to_vec()).collect();
  let fit_r = ols(&y_slice, &x_rows)?;
  let sse_r = fit_r.sse;
  let beta_r = ndarray::Array1::from_vec(fit_r.beta.clone());

  // Fitted values from the restricted model.
  let fitted = x.dot(&beta_r).to_vec();
  // Ramsey (1969) adds powers of ŷ; with a constant in `x`, ŷ and the constant lie in its span, so
  // those of the standardised ŷ span the same columns: the same `F`, without their collinearity.
  let powered = if constant {
    standardised(&fitted)
  } else {
    // Without a constant the raw powers define the test, Singular from ŷ at 1e3 times its spread.
    fitted
  };

  let mut x_aug_rows = x_rows.clone();
  for (i, row) in x_aug_rows.iter_mut().enumerate() {
    let yhat = powered[i];
    let mut power = yhat * yhat;
    row.push(power);
    for _ in 3..=cfg.max_power {
      power *= yhat;
      row.push(power);
    }
  }
  let fit_u = ols(&y_slice, &x_aug_rows)?;
  reject_exact_fit(&y_slice, &fit_u.residuals, fit_u.condition)?;
  let sse_u = fit_u.sse;

  let df_num = q;
  let df_den = t_total - k - q;
  let f_statistic = nested_f(sse_r, sse_u, df_num as f64, df_den as f64)?;
  let p_value = f_upper_tail(f_statistic, df_num as f64, df_den as f64);

  Ok(ResetTest {
    statistic: f_statistic,
    p_value: Some(p_value),
    critical_values: None,
    rejects_null: p_value < cfg.alpha,
    df_num,
    df_den,
  })
}

/// Whether a column of `x` holds one nonzero value throughout, which puts the constant in its span.
fn has_constant_column(x: ArrayView2<f64>) -> bool {
  x.columns()
    .into_iter()
    .any(|c| c[0] != 0.0 && c.iter().all(|v| *v == c[0]))
}

fn standardised(v: &[f64]) -> Vec<f64> {
  let deviations = centred(v);
  let scale = (deviations.iter().map(|u| u * u).sum::<f64>() / v.len() as f64).sqrt();
  if scale > 0.0 {
    deviations.iter().map(|u| u / scale).collect()
  } else {
    deviations
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use ndarray::Array2;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::DistributionSampler;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  fn design_with_intercept(x: &Array1<f64>) -> Array2<f64> {
    let n = x.len();
    Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { x[i] })
  }

  fn normal_pair(n: usize, noise_sigma: f64, seed: u64) -> (Array1<f64>, Array1<f64>) {
    let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
    let mut x = Array1::zeros(n);
    dist.fill_slice(x.as_slice_mut().unwrap());
    let mut noise = Array1::zeros(n);
    dist.fill_slice(noise.as_slice_mut().unwrap());
    (x, noise * noise_sigma)
  }

  fn simulate_linear(n: usize, seed: u64) -> (Array1<f64>, Array1<f64>) {
    let (x, noise) = normal_pair(n, 0.5, seed);
    let y = 1.0 + 2.0 * &x + &noise;
    (x, y)
  }

  fn simulate_quadratic(n: usize, seed: u64) -> (Array1<f64>, Array1<f64>) {
    let (x, noise) = normal_pair(n, 0.3, seed);
    let y = 1.0 + &x + &(&x * &x) + &noise;
    (x, y)
  }

  fn simulate_cubic(n: usize, seed: u64) -> (Array1<f64>, Array1<f64>) {
    let (x, noise) = normal_pair(n, 0.3, seed);
    let y = 1.0 + &x + &(&x * &x * &x) + &noise;
    (x, y)
  }

  #[test]
  fn reset_does_not_reject_linear_dgp() {
    let (x, y) = simulate_linear(300, 42);
    let xm = design_with_intercept(&x);
    let res = reset_test(y.view(), xm.view(), ResetConfig::default()).unwrap();
    assert!(
      !res.rejects_null,
      "linear DGP: should not reject; F = {}, p = {:?}",
      res.statistic, res.p_value
    );
  }

  #[test]
  fn reset_rejects_quadratic_dgp() {
    let (x, y) = simulate_quadratic(300, 7);
    let xm = design_with_intercept(&x);
    let res = reset_test(y.view(), xm.view(), ResetConfig::default()).unwrap();
    assert!(
      res.rejects_null,
      "quadratic DGP: should reject; F = {}, p = {:?}",
      res.statistic, res.p_value
    );
  }

  #[test]
  fn reset_rejects_cubic_dgp() {
    let (x, y) = simulate_cubic(300, 11);
    let xm = design_with_intercept(&x);
    let res = reset_test(y.view(), xm.view(), ResetConfig::default()).unwrap();
    assert!(
      res.rejects_null,
      "cubic DGP: should reject; F = {}, p = {:?}",
      res.statistic, res.p_value
    );
  }

  const SCALES: [f64; 9] = [1e-300, 1e-200, 1e-160, 1e-6, 1.0, 1e6, 1e160, 1e200, 1e300];

  /// A line, exact or with noise far below `1e-12` of it, leaves both fits rounding at every scale
  /// of `y`, and with `x` scaled alike from `1e-6` to `1e6`.
  #[test]
  fn a_line_fitted_to_rounding_is_degenerate_data_at_every_scale() {
    let x: Array1<f64> = Array1::linspace(-1.0, 1.0, 50);
    let (_, noise) = normal_pair(50, 1e-15, 5);
    let scales = SCALES
      .map(|c| (c, 1.0))
      .into_iter()
      .chain([(1e-6, 1e-6), (1e6, 1e6)]);
    for y in [&x * 3.0 + 2.0, &x * 3.0 + 2.0 + &noise] {
      for (cy, cx) in scales.clone() {
        let xm = design_with_intercept(&x.mapv(|v| v * cx));
        assert_eq!(
          reset_test(y.mapv(|v| v * cy).view(), xm.view(), ResetConfig::default()).unwrap_err(),
          StatsError::DegenerateData("zero residual variance"),
          "y × {cy:e}, x × {cx:e}"
        );
      }
    }
  }

  /// `F` does not see the scale of `y`, which its powers of `ŷ` would under- or overflow unscaled.
  #[test]
  fn the_statistic_keeps_its_value_at_every_scale_of_y() {
    for (x, y) in [simulate_linear(300, 42), simulate_quadratic(300, 7)] {
      let xm = design_with_intercept(&x);
      let unit = reset_test(y.view(), xm.view(), ResetConfig::default())
        .unwrap()
        .statistic;
      for c in SCALES {
        let got = reset_test(y.mapv(|v| v * c).view(), xm.view(), ResetConfig::default())
          .map(|r| r.statistic);
        assert!(
          got.as_ref().is_ok_and(|f| (f - unit).abs() <= 1e-12 * unit),
          "y × {c:e}: {got:?} against {unit}"
        );
      }
    }
  }

  /// An exact line leaves both fits rounding, its regressor to `1e5` times its spread above zero.
  #[test]
  fn an_exact_line_far_above_its_spread_is_refused() {
    let (x, _) = normal_pair(50, 0.0, 7);
    for level in [10.0, 1e2, 1e3, 1e5] {
      let x = x.mapv(|v| v + level);
      let y = &x * 3.0 + 2.0;
      let got = reset_test(
        y.view(),
        design_with_intercept(&x).view(),
        ResetConfig::default(),
      );
      let refused = [
        StatsError::DegenerateData("zero residual variance"),
        StatsError::Singular("OLS design"),
      ];
      assert!(
        got.as_ref().is_err_and(|e| refused.contains(e)),
        "level {level}: {got:?}"
      );
    }
  }

  /// A level added to `y` leaves the span of the powers of `ŷ`, and so `F`, as it is.
  #[test]
  fn the_statistic_holds_with_fitted_values_far_above_their_spread() {
    for (x, y) in [simulate_linear(300, 42), simulate_quadratic(300, 7)] {
      let xm = design_with_intercept(&x);
      let at_zero = reset_test(y.view(), xm.view(), ResetConfig::default())
        .unwrap()
        .statistic;
      for level in [10.0, 30.0, 100.0, 1e3, 1e5] {
        let got = reset_test(
          y.mapv(|v| v + level).view(),
          xm.view(),
          ResetConfig::default(),
        )
        .map(|r| r.statistic);
        assert!(
          got
            .as_ref()
            .is_ok_and(|f| (f - at_zero).abs() <= 1e-6 * at_zero),
          "level {level}: {got:?} against {at_zero}"
        );
      }
    }
  }

  /// The linear model's `F` at a level, against the same data's `F` in double-double arithmetic.
  #[test]
  fn the_statistic_matches_its_double_double_value_at_a_level() {
    let (x, y) = simulate_linear(300, 42);
    let xm = design_with_intercept(&x);
    let cases = [
      (1e2, 2.433378958960329),
      (1e3, 2.43337895896031),
      (1e5, 2.4333789589603674),
    ];
    for (level, reference) in cases {
      let got = reset_test(
        y.mapv(|v| v + level).view(),
        xm.view(),
        ResetConfig::default(),
      )
      .map(|r| r.statistic);
      assert!(
        got
          .as_ref()
          .is_ok_and(|f| (f - reference).abs() <= 1e-11 * reference),
        "level {level:e}: {got:?} against {reference}"
      );
    }
  }

  /// A small `F` at a level of `1e5`, where the level's rounding in `y` took eight of its digits.
  #[test]
  fn a_small_statistic_matches_its_double_double_value_at_a_level() {
    let cases = [
      (300, 5, 0.00813005555155731),
      (1000, 7, 0.07390520170749103),
    ];
    for (n, seed, reference) in cases {
      let (x, e) = (
        normal_pair(n, 1.0, seed).0,
        normal_pair(n, 0.5, seed + 300).0,
      );
      let y = 1e5 + 2.0 * &x + 0.5 * &e;
      let xm = design_with_intercept(&x);
      let got = reset_test(y.view(), xm.view(), ResetConfig::default()).map(|r| r.statistic);
      assert!(
        got
          .as_ref()
          .is_ok_and(|f| (f - reference).abs() <= 1e-10 * reference),
        "n = {n}, seed {seed}: {got:?} against {reference}"
      );
    }
  }

  /// Without a constant column the powers of `ŷ` itself define the test, collinear at `3e3` above.
  #[test]
  fn without_a_constant_the_raw_powers_far_above_their_spread_are_singular() {
    let (x, noise) = normal_pair(300, 0.5, 42);
    let x = x.mapv(|v| v + 3e3);
    let y = &x * 2.0 + &noise;
    let xm = x.clone().insert_axis(ndarray::Axis(1));
    assert_eq!(
      reset_test(y.view(), xm.view(), ResetConfig::default()).unwrap_err(),
      StatsError::Singular("OLS design")
    );
  }

  #[test]
  fn a_design_with_other_rows_is_a_dimension_mismatch() {
    let (x, y) = simulate_linear(100, 3);
    let xm = design_with_intercept(&x.slice(ndarray::s![..99]).to_owned());
    assert_eq!(
      reset_test(y.view(), xm.view(), ResetConfig::default()).unwrap_err(),
      StatsError::DimensionMismatch {
        expected: 100,
        got: 99
      }
    );
  }

  #[test]
  fn a_non_finite_design_row_is_an_invalid_observation() {
    let (x, y) = simulate_linear(100, 3);
    let mut xm = design_with_intercept(&x);
    xm[[40, 1]] = f64::INFINITY;
    assert_eq!(
      reset_test(y.view(), xm.view(), ResetConfig::default()).unwrap_err(),
      StatsError::InvalidObservation {
        index: 40,
        value: f64::INFINITY,
        constraint: "finite"
      }
    );
  }

  #[test]
  fn a_constant_series_is_degenerate_data() {
    let y = Array1::<f64>::from_elem(64, 0.3);
    let xm = design_with_intercept(&Array1::linspace(-1.0, 1.0, 64));
    assert_eq!(
      reset_test(y.view(), xm.view(), ResetConfig::default()).unwrap_err(),
      StatsError::DegenerateData("constant series")
    );
  }

  /// `y = x` on `x ∈ {±1, ±2}` keeps every normal-equation step in exact dyadic arithmetic.
  #[test]
  fn an_exact_fit_is_degenerate_data() {
    let x = Array1::from_iter((0..18).map(|i| match i {
      16 => 2.0,
      17 => -2.0,
      _ if i % 2 == 0 => 1.0,
      _ => -1.0,
    }));
    let xm = x.clone().insert_axis(ndarray::Axis(1));
    assert_eq!(
      reset_test(x.view(), xm.view(), ResetConfig::default()).unwrap_err(),
      StatsError::DegenerateData("zero residual variance")
    );
  }

  #[test]
  fn reset_df_accounting_is_correct() {
    let (x, y) = simulate_linear(100, 99);
    let xm = design_with_intercept(&x);
    let res = reset_test(
      y.view(),
      xm.view(),
      ResetConfig {
        max_power: 4,
        alpha: 0.05,
      },
    )
    .unwrap();
    // p = 4 ⇒ q = 3; k = 2 (intercept + slope); T - k - q = 100 - 2 - 3 = 95.
    assert_eq!(res.df_num, 3, "df_num mismatch");
    assert_eq!(res.df_den, 95, "df_den mismatch");
  }
}
