//! Brown–Durbin–Evans (1975) CUSUM and CUSUMQ structural-change tests on recursive residuals.
//!
//! $$
//! w_t = \frac{y_t - x_t^\top \hat\beta_{t-1}}{\sqrt{1 + x_t^\top
//! (X_{t-1}^\top X_{t-1})^{-1} x_t}} \overset{H_0}{\sim} \mathcal N(0, \sigma^2),
//! \qquad t = k+1, \dots, T
//! $$
//!
//! $$
//! W_t = \hat\sigma^{-1} \sum_{j=k+1}^t w_j,\qquad
//! |W_t| \le a_\alpha \sqrt{T-k} + \frac{2 a_\alpha (t-k)}{\sqrt{T-k}},\qquad
//! a_\alpha \in \{1.143, 0.948, 0.850\}
//! $$
//!
//! $$
//! S_t = \frac{\sum_{j=k+1}^t w_j^2}{\sum_{j=k+1}^T w_j^2},\qquad
//! \Bigl|S_t - \frac{t-k}{T-k}\Bigr| \le c_\alpha \sqrt{\frac{2}{T-k-2}},\qquad
//! c_\alpha \in \{1.07275, 0.84717, 0.74346\}
//! $$
//!
//! CUSUM watches the mean (Table 1), CUSUMQ the variance; `breakpoint` is the first crossing.
//!
//! Reference: Brown et al. (1975), JRSS B 37(2), 149-192, DOI: 10.1111/j.2517-6161.1975.tb01532.x
//! Reference: Durbin (1969), Biometrika 56(1), 1-15, DOI: 10.1093/biomet/56.1.1
//! Reference: Edgerton & Wells (1994), OBES 56(3), 355-365, 10.1111/j.1468-0084.1994.mp56003007.x

use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use stochastic_rs_distributions::linalg::inverse;

use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::input::as_slice_or_copy;
use crate::input::assert_alpha;
use crate::input::reject_constant;
use crate::input::validate;
use crate::input::validate_rows;
use crate::stationarity::common::regression_at_unit_scale;
use crate::traits::RealExt;

/// Choice of CUSUM variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CusumVariant {
  /// CUSUM of recursive residuals — sensitive to mean shifts.
  Cusum,
  /// CUSUM of squared recursive residuals (CUSUMSQ) — sensitive to variance
  /// shifts.
  Cusumq,
}

/// Configuration for the CUSUM / CUSUMQ test.
#[derive(Debug, Clone, Copy)]
pub struct CusumConfig {
  /// Test variant.
  pub variant: CusumVariant,
  /// Significance level used for the rejection boundary.
  pub alpha: f64,
}

impl Default for CusumConfig {
  fn default() -> Self {
    Self {
      variant: CusumVariant::Cusum,
      alpha: 0.05,
    }
  }
}

/// CUSUM / CUSUMQ test (null: constant regression parameters), decided against `critical_boundary`.
#[derive(Debug, Clone)]
pub struct CusumTest {
  /// Maximum absolute excess of the cumulative path over its boundary —
  /// positive when the path exits the band, non-positive otherwise.
  pub statistic: f64,
  pub p_value: Option<f64>,
  pub critical_values: Option<CriticalValues>,
  /// Whether the null of parameter constancy is rejected at `alpha`.
  pub rejects_null: bool,
  /// Variant that produced this result.
  pub variant: CusumVariant,
  /// First time index (in the original sample, 1-based) at which the
  /// boundary is crossed, or `None` when the null is not rejected.
  pub breakpoint: Option<usize>,
  /// Recursive residuals $w_{k+1}, \dots, w_T$.
  pub recursive_residuals: Vec<f64>,
  /// Cumulative-sum path values, length $T - k$.
  pub cusum_path: Vec<f64>,
  /// Two-sided rejection boundary `(lower, upper)` matching `cusum_path`.
  pub critical_boundary: Vec<(f64, f64)>,
}

impl crate::traits::HypothesisTest for CusumTest {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.rejects_null)
  }
}

/// CUSUM boundary coefficient $a_\alpha$ from Brown-Durbin-Evans (1975)
/// Table 1.
fn cusum_critical_coefficient(alpha: f64) -> f64 {
  if alpha <= 0.01 {
    1.143
  } else if alpha <= 0.05 {
    0.948
  } else {
    0.850
  }
}

/// CUSUMQ boundary coefficient $c_\alpha$, Edgerton–Wells' (1994) large-sample approximation to
/// Durbin's (1969) exact table; the band's half-width is $c_\alpha \sqrt{2/(T-k-2)}$.
fn cusumq_critical_coefficient(alpha: f64) -> f64 {
  if alpha <= 0.01 {
    1.07275
  } else if alpha <= 0.05 {
    0.84717
  } else {
    0.74346
  }
}

/// CUSUM / CUSUMQ test of `y` on `x`; `Err` below `k + 2` points, a non-finite or constant `y`, a
/// non-finite `x`, mismatched rows, a singular block; panics unless `k >= 1` and `0 < alpha < 1`.
pub fn cusum_test(
  y: ArrayView1<f64>,
  x: ArrayView2<f64>,
  cfg: CusumConfig,
) -> Result<CusumTest, StatsError> {
  let k = x.ncols();
  assert!(
    k >= 1,
    "x.ncols() must satisfy `x.ncols() >= 1`, got x.ncols() = {k:?}"
  );
  assert_alpha(cfg.alpha);
  if x.nrows() != y.len() {
    return Err(StatsError::DimensionMismatch {
      expected: y.len(),
      got: x.nrows(),
    });
  }
  let y_slice = as_slice_or_copy(y);
  validate(&y_slice, k + 2)?;
  validate_rows(x)?;
  reject_constant(&y_slice, "constant series")?;

  let t_total = y.len();
  let (y_unit, units, x_unit) = regression_at_unit_scale(y_slice.into_owned(), x);
  let w = recursive_residuals(ArrayView1::from(&y_unit), x_unit.view())?;
  debug_assert_eq!(w.len(), t_total - k);

  let n_tilde_usize = t_total - k;
  let n_tilde = n_tilde_usize as f64;

  let mut test = match cfg.variant {
    CusumVariant::Cusum => cusum_branch(w, n_tilde, k, cfg.alpha),
    CusumVariant::Cusumq => cusumq_branch(w, n_tilde, k, cfg.alpha),
  };
  let in_units = test.recursive_residuals.iter_mut();
  in_units.for_each(|w| *w = *w / units[1] / units[0]);
  Ok(test)
}

fn cusum_branch(w: Vec<f64>, n_tilde: f64, k: usize, alpha: f64) -> CusumTest {
  let w_mean = w.iter().sum::<f64>() / n_tilde;
  let sigma2 = w.iter().map(|wj| (wj - w_mean).powi(2)).sum::<f64>() / (n_tilde - 1.0).max(1.0);
  let sigma = sigma2.max(0.0).sqrt();
  let sigma_safe = if sigma > 0.0 { sigma } else { 1.0 };

  let a0 = cusum_critical_coefficient(alpha);
  let sqrt_n = n_tilde.sqrt();

  let mut path = Vec::with_capacity(w.len());
  let mut boundary = Vec::with_capacity(w.len());
  let mut excesses = Vec::with_capacity(w.len());
  let mut cum = 0.0_f64;
  let mut breakpoint: Option<usize> = None;

  for (i, wi) in w.iter().enumerate() {
    cum += wi / sigma_safe;
    path.push(cum);
    let upper = a0 * sqrt_n + 2.0 * a0 * (i as f64 + 1.0) / sqrt_n;
    boundary.push((-upper, upper));
    let excess = cum.abs() - upper;
    excesses.push(excess);
    if excess > 0.0 && breakpoint.is_none() {
      breakpoint = Some(k + i + 1);
    }
  }
  let max_excess = f64::max_or_nan_fold(excesses);

  CusumTest {
    statistic: max_excess,
    p_value: None,
    critical_values: None,
    rejects_null: max_excess > 0.0,
    variant: CusumVariant::Cusum,
    breakpoint,
    recursive_residuals: w,
    cusum_path: path,
    critical_boundary: boundary,
  }
}

fn cusumq_branch(w: Vec<f64>, n_tilde: f64, k: usize, alpha: f64) -> CusumTest {
  let denom: f64 = w.iter().map(|wj| wj * wj).sum();
  let denom_safe = if denom > 0.0 { denom } else { 1.0 };
  let c_alpha = cusumq_critical_coefficient(alpha);
  let half_band = c_alpha * (2.0_f64 / (n_tilde - 2.0).max(1.0)).sqrt();

  let mut path = Vec::with_capacity(w.len());
  let mut boundary = Vec::with_capacity(w.len());
  let mut excesses = Vec::with_capacity(w.len());
  let mut cum_sq = 0.0_f64;
  let mut breakpoint: Option<usize> = None;

  for (i, wi) in w.iter().enumerate() {
    cum_sq += wi * wi;
    let s = cum_sq / denom_safe;
    let mean = (i as f64 + 1.0) / n_tilde;
    path.push(s);
    boundary.push((mean - half_band, mean + half_band));
    let excess = (s - mean).abs() - half_band;
    excesses.push(excess);
    if excess > 0.0 && breakpoint.is_none() {
      breakpoint = Some(k + i + 1);
    }
  }
  let max_excess = f64::max_or_nan_fold(excesses);

  CusumTest {
    statistic: max_excess,
    p_value: None,
    critical_values: None,
    rejects_null: max_excess > 0.0,
    variant: CusumVariant::Cusumq,
    breakpoint,
    recursive_residuals: w,
    cusum_path: path,
    critical_boundary: boundary,
  }
}

/// Recursive residuals via incremental OLS with a Sherman-Morrison update
/// of $(X_t^\top X_t)^{-1}$.
fn recursive_residuals(y: ArrayView1<f64>, x: ArrayView2<f64>) -> Result<Vec<f64>, StatsError> {
  let t_total = y.len();
  let k = x.ncols();

  let x_init = x.slice(ndarray::s![..k, ..]).to_owned();
  let y_init = y.slice(ndarray::s![..k]).to_owned();
  let mut xtx_inv =
    inverse(x_init.t().dot(&x_init).view()).ok_or(StatsError::Singular("OLS design"))?;
  let mut xty = x_init.t().dot(&y_init);

  let mut w = Vec::with_capacity(t_total - k);
  let mut outer_buf = Array2::<f64>::zeros((k, k));

  for t in k..t_total {
    let xt = x.row(t).to_owned();
    let beta_prev = xtx_inv.dot(&xty);
    let pred = xt.dot(&beta_prev);
    let m_xt = xtx_inv.dot(&xt);
    let f_t = 1.0 + xt.dot(&m_xt);
    let d_t = f_t.max(f64::MIN_POSITIVE).sqrt();
    w.push((y[t] - pred) / d_t);

    for i in 0..k {
      for j in 0..k {
        outer_buf[[i, j]] = m_xt[i] * m_xt[j] / f_t;
      }
    }
    xtx_inv = &xtx_inv - &outer_buf;
    xty = &xty + &(&xt * y[t]);
  }

  Ok(w)
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::DistributionSampler;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  fn linear_design(n: usize) -> ndarray::Array2<f64> {
    Array2::from_shape_fn(
      (n, 2),
      |(i, j)| if j == 0 { 1.0 } else { i as f64 / n as f64 },
    )
  }

  fn standard_normal(n: usize, seed: u64) -> Array1<f64> {
    let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
    let mut out = Array1::zeros(n);
    dist.fill_slice(out.as_slice_mut().unwrap());
    out
  }

  fn simulate_stationary(n: usize, seed: u64) -> Array1<f64> {
    let x = linear_design(n);
    let beta = ndarray::array![0.5, 1.2];
    let noise = standard_normal(n, seed);
    x.dot(&beta) + noise
  }

  fn simulate_mean_shift(n: usize, shift_at: usize, shift: f64, seed: u64) -> Array1<f64> {
    let mut y = simulate_stationary(n, seed);
    for i in shift_at..n {
      y[i] += shift;
    }
    y
  }

  fn simulate_variance_shift(n: usize, shift_at: usize, sigma_high: f64, seed: u64) -> Array1<f64> {
    let x = linear_design(n);
    let beta = ndarray::array![0.5, 1.2];
    let mu = x.dot(&beta);
    let z = standard_normal(n, seed);
    let mut y = Array1::zeros(n);
    for i in 0..n {
      let s = if i < shift_at { 1.0 } else { sigma_high };
      y[i] = mu[i] + s * z[i];
    }
    y
  }

  #[test]
  fn cusum_no_reject_under_constant_dgp() {
    // `α = 0.01` keeps the type I error near 1 %, so three seeds stay well under 5 % jointly.
    let n = 200;
    let x = linear_design(n);
    for seed in [42_u64, 7, 99] {
      let y = simulate_stationary(n, seed);
      let res = cusum_test(
        y.view(),
        x.view(),
        CusumConfig {
          variant: CusumVariant::Cusum,
          alpha: 0.01,
        },
      )
      .unwrap();
      assert!(
        !res.rejects_null,
        "CUSUM should not reject constant DGP at seed {seed}: statistic = {}",
        res.statistic
      );
    }
  }

  #[test]
  fn cusum_rejects_mean_shift() {
    let n = 200;
    let y = simulate_mean_shift(n, 100, 3.0, 7);
    let x = linear_design(n);
    let res = cusum_test(
      y.view(),
      x.view(),
      CusumConfig {
        variant: CusumVariant::Cusum,
        alpha: 0.05,
      },
    )
    .unwrap();
    assert!(
      res.rejects_null,
      "CUSUM should reject mean shift: statistic = {}",
      res.statistic
    );
    let bp = res.breakpoint.expect("breakpoint must be reported");
    assert!(
      (bp as i64 - 100).abs() < 80,
      "breakpoint {bp} should be near the true shift at 100"
    );
  }

  #[test]
  fn cusumq_rejects_variance_shift() {
    let n = 200;
    let y = simulate_variance_shift(n, 100, 4.0, 11);
    let x = linear_design(n);
    let res = cusum_test(
      y.view(),
      x.view(),
      CusumConfig {
        variant: CusumVariant::Cusumq,
        alpha: 0.05,
      },
    )
    .unwrap();
    assert!(
      res.rejects_null,
      "CUSUMQ should reject variance shift: statistic = {}",
      res.statistic
    );
  }

  #[test]
  fn cusumq_does_not_reject_constant_dgp() {
    // α = 0.01 — see [`cusum_no_reject_under_constant_dgp`] for rationale.
    let n = 200;
    let x = linear_design(n);
    for seed in [11_u64, 12, 22] {
      let y = simulate_stationary(n, seed);
      let res = cusum_test(
        y.view(),
        x.view(),
        CusumConfig {
          variant: CusumVariant::Cusumq,
          alpha: 0.01,
        },
      )
      .unwrap();
      assert!(
        !res.rejects_null,
        "CUSUMQ should not reject constant DGP at seed {seed}: statistic = {}",
        res.statistic
      );
    }
  }

  #[test]
  fn a_design_with_other_rows_is_a_dimension_mismatch() {
    let y = simulate_stationary(100, 5);
    assert_eq!(
      cusum_test(y.view(), linear_design(99).view(), CusumConfig::default()).unwrap_err(),
      StatsError::DimensionMismatch {
        expected: 100,
        got: 99
      }
    );
  }

  #[test]
  fn a_constant_series_is_degenerate_data() {
    let y = Array1::<f64>::from_elem(100, 0.3);
    assert_eq!(
      cusum_test(y.view(), linear_design(100).view(), CusumConfig::default()).unwrap_err(),
      StatsError::DegenerateData("constant series")
    );
  }

  #[test]
  fn a_non_finite_design_row_is_an_invalid_observation() {
    let y = simulate_stationary(100, 5);
    let mut x = linear_design(100);
    x[[37, 1]] = f64::INFINITY;
    assert_eq!(
      cusum_test(y.view(), x.view(), CusumConfig::default()).unwrap_err(),
      StatsError::InvalidObservation {
        index: 37,
        value: f64::INFINITY,
        constraint: "finite"
      }
    );
  }

  #[test]
  fn the_maximum_excess_keeps_a_nan() {
    let w = vec![0.5, f64::NAN, -0.5, 0.25];
    assert!(cusum_branch(w.clone(), 4.0, 1, 0.05).statistic.is_nan());
    assert!(cusumq_branch(w, 4.0, 1, 0.05).statistic.is_nan());
  }

  #[test]
  fn cusum_recursive_residual_variance_matches_ols_sigma2() {
    let n = 100;
    let y = simulate_stationary(n, 99);
    let x = linear_design(n);
    let res = cusum_test(
      y.view(),
      x.view(),
      CusumConfig {
        variant: CusumVariant::Cusum,
        alpha: 0.05,
      },
    )
    .unwrap();
    let w = &res.recursive_residuals;
    let w_mean = w.iter().sum::<f64>() / w.len() as f64;
    let s2_w = w.iter().map(|wj| (wj - w_mean).powi(2)).sum::<f64>() / (w.len() as f64 - 1.0);

    // OLS σ² on the full sample.
    let beta = inverse(x.t().dot(&x).view()).unwrap().dot(&x.t().dot(&y));
    let resid = &y - &x.dot(&beta);
    let s2_ols = resid.iter().map(|r| r * r).sum::<f64>() / (n as f64 - 2.0);

    // Under the linear-Gaussian null both estimators are consistent for σ²;
    // require they agree to within 25 % at n = 100.
    let rel_err = (s2_w - s2_ols).abs() / s2_ols;
    assert!(
      rel_err < 0.25,
      "recursive σ² ({s2_w:.4}) far from OLS σ² ({s2_ols:.4}); rel_err = {rel_err:.4}"
    );
  }
}
