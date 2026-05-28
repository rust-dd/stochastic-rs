//! Ramsey (1969) RESET (REgression Equation Specification Error Test) for
//! functional-form misspecification.
//!
//! Given the linear regression $y_t = x_t^\top \beta + \varepsilon_t$, fit OLS
//! to obtain fitted values $\hat y_t$ and augment with powers
//! $\hat y_t^2, \dots, \hat y_t^p$:
//! $$
//! y_t = x_t^\top \beta + \sum_{j=2}^{p} \gamma_j \hat y_t^{\,j}
//!       + \varepsilon_t.
//! $$
//! Under the null of correct linear specification, $H_0: \gamma_2 = \dots =
//! \gamma_p = 0$, the joint F-statistic
//! $$
//! F = \frac{(\mathrm{SSR}_r - \mathrm{SSR}_u)/q}{\mathrm{SSR}_u/(T - k - q)},
//! \qquad q = p - 1,
//! $$
//! is asymptotically $F(q, T - k - q)$. The default `powers = 2..=3` matches
//! Ramsey's original recommendation.
//!
//! Reference: Ramsey, J.B. (1969), "Tests for Specification Errors in
//! Classical Linear Least Squares Regression Analysis", *JRSS B* 31(2),
//! 350-371.

use ndarray::ArrayView1;
use ndarray::ArrayView2;
use stochastic_rs_distributions::special::beta_i;

use super::common::ols;
use super::common::validate_series;

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

/// Configuration for the RESET test.
#[derive(Debug, Clone, Copy)]
pub struct ResetConfig {
  /// Highest power of $\hat y$ added to the augmented regression. Must be
  /// $\ge 2$. Default `3` ($\hat y^2$ and $\hat y^3$).
  pub max_power: usize,
  /// Significance level used to compute `reject_correct_specification`.
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

/// Result of the RESET test.
#[derive(Debug, Clone, Copy)]
pub struct ResetResult {
  /// F-statistic for the joint restriction $\gamma_2 = \dots = \gamma_p = 0$.
  pub f_statistic: f64,
  /// Numerator degrees of freedom $q = p - 1$.
  pub df_num: usize,
  /// Denominator degrees of freedom $T - k - q$.
  pub df_den: usize,
  /// Asymptotic p-value from the $F(q, T-k-q)$ distribution.
  pub p_value: f64,
  /// Whether the correct-specification null is rejected at `alpha`.
  pub reject_correct_specification: bool,
}

impl crate::traits::HypothesisTest for ResetResult {
  fn statistic(&self) -> f64 {
    self.f_statistic
  }
  fn null_rejected(&self) -> Option<bool> {
    Some(self.reject_correct_specification)
  }
}

/// Ramsey (1969) RESET test.
///
/// `x` is the $T \times k$ design matrix (must include a constant column if
/// the model has an intercept) and `y` is the $T \times 1$ response.
///
/// # Panics
/// Panics on non-finite inputs, `max_power < 2`, $T \le k + (p-1)$, or
/// `alpha` outside $(0, 1)$.
pub fn reset_test(y: ArrayView1<f64>, x: ArrayView2<f64>, cfg: ResetConfig) -> ResetResult {
  let y_slice = y
    .as_slice()
    .expect("reset_test requires a contiguous ArrayView1");
  assert_eq!(y.len(), x.nrows(), "y/x row mismatch");
  let t_total = y.len();
  let k = x.ncols();
  let q = cfg.max_power.saturating_sub(1);
  assert!(cfg.max_power >= 2, "RESET requires max_power >= 2");
  assert!(t_total > k + q, "need T > k + q observations");
  validate_series(y_slice, k + q + 2);
  assert!(
    cfg.alpha > 0.0 && cfg.alpha < 1.0,
    "alpha must be in (0, 1)"
  );

  // Restricted OLS: y on x.
  let x_rows: Vec<Vec<f64>> = (0..t_total).map(|i| x.row(i).to_vec()).collect();
  let fit_r = ols(y_slice, &x_rows);
  let sse_r = fit_r.sse;
  let beta_r = ndarray::Array1::from_vec(fit_r.beta.clone());

  // Fitted values from the restricted model.
  let fitted = x.dot(&beta_r);

  // Augmented design: append ŷ^2, …, ŷ^p as additional regressors.
  let mut x_aug_rows = x_rows.clone();
  for (i, row) in x_aug_rows.iter_mut().enumerate() {
    let yhat = fitted[i];
    let mut power = yhat * yhat;
    row.push(power);
    for _ in 3..=cfg.max_power {
      power *= yhat;
      row.push(power);
    }
  }
  let fit_u = ols(y_slice, &x_aug_rows);
  let sse_u = fit_u.sse;

  let df_num = q;
  let df_den = t_total
    .checked_sub(k + q)
    .expect("RESET denominator d.o.f. underflowed");
  let num = ((sse_r - sse_u) / df_num as f64).max(0.0);
  let den = sse_u / df_den as f64;
  let f_statistic = if den > 0.0 { num / den } else { f64::INFINITY };

  let p_value = if f_statistic.is_finite() {
    f_upper_tail(f_statistic, df_num as f64, df_den as f64)
  } else {
    0.0
  };

  ResetResult {
    f_statistic,
    df_num,
    df_den,
    p_value,
    reject_correct_specification: p_value < cfg.alpha,
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use ndarray::Array2;
  use ndarray_rand::RandomExt;
  use ndarray_rand::rand_distr::Normal;
  use rand::SeedableRng;
  use rand::rngs::StdRng;

  use super::*;

  fn design_with_intercept(x: &Array1<f64>) -> Array2<f64> {
    let n = x.len();
    Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { x[i] })
  }

  fn simulate_linear(n: usize, seed: u64) -> (Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x = Array1::random_using(n, Normal::new(0.0, 1.0).unwrap(), &mut rng);
    let noise = Array1::random_using(n, Normal::new(0.0, 0.5).unwrap(), &mut rng);
    let y = 1.0 + 2.0 * &x + &noise;
    (x, y)
  }

  fn simulate_quadratic(n: usize, seed: u64) -> (Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x = Array1::random_using(n, Normal::new(0.0, 1.0).unwrap(), &mut rng);
    let noise = Array1::random_using(n, Normal::new(0.0, 0.3).unwrap(), &mut rng);
    let y = 1.0 + &x + &(&x * &x) + &noise;
    (x, y)
  }

  fn simulate_cubic(n: usize, seed: u64) -> (Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x = Array1::random_using(n, Normal::new(0.0, 1.0).unwrap(), &mut rng);
    let noise = Array1::random_using(n, Normal::new(0.0, 0.3).unwrap(), &mut rng);
    let y = 1.0 + &x + &(&x * &x * &x) + &noise;
    (x, y)
  }

  #[test]
  fn reset_does_not_reject_linear_dgp() {
    let (x, y) = simulate_linear(300, 42);
    let xm = design_with_intercept(&x);
    let res = reset_test(y.view(), xm.view(), ResetConfig::default());
    assert!(
      !res.reject_correct_specification,
      "linear DGP: should not reject; F = {}, p = {}",
      res.f_statistic, res.p_value
    );
  }

  #[test]
  fn reset_rejects_quadratic_dgp() {
    let (x, y) = simulate_quadratic(300, 7);
    let xm = design_with_intercept(&x);
    let res = reset_test(y.view(), xm.view(), ResetConfig::default());
    assert!(
      res.reject_correct_specification,
      "quadratic DGP: should reject; F = {}, p = {}",
      res.f_statistic, res.p_value
    );
  }

  #[test]
  fn reset_rejects_cubic_dgp() {
    let (x, y) = simulate_cubic(300, 11);
    let xm = design_with_intercept(&x);
    let res = reset_test(y.view(), xm.view(), ResetConfig::default());
    assert!(
      res.reject_correct_specification,
      "cubic DGP: should reject; F = {}, p = {}",
      res.f_statistic, res.p_value
    );
  }

  #[test]
  fn reset_f_stat_matches_hand_computed_fixture() {
    // Exactly linear DGP with no noise — augmented model fits identically and
    // F-stat must be ≈ 0 to numerical precision.
    let x: Array1<f64> = Array1::linspace(-1.0, 1.0, 50);
    let y: Array1<f64> = &x * 3.0 + 2.0;
    let xm = design_with_intercept(&x);
    let res = reset_test(y.view(), xm.view(), ResetConfig::default());
    assert!(
      res.f_statistic < 1e-6,
      "noiseless linear F should be ~ 0; got {}",
      res.f_statistic
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
    );
    // p = 4 ⇒ q = 3; k = 2 (intercept + slope); T - k - q = 100 - 2 - 3 = 95.
    assert_eq!(res.df_num, 3, "df_num mismatch");
    assert_eq!(res.df_den, 95, "df_den mismatch");
  }
}
