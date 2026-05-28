//! Brown-Durbin-Evans (1975) CUSUM and CUSUMQ structural-change tests.
//!
//! Given the linear regression $y_t = x_t^\top \beta + \varepsilon_t$ for
//! $t = 1, \dots, T$ with $k$ regressors, the **recursive residuals**
//! $$
//! w_t = \frac{y_t - x_t^\top \hat\beta_{t-1}}{\sqrt{1 + x_t^\top
//! (X_{t-1}^\top X_{t-1})^{-1} x_t}}, \qquad t = k+1, \dots, T
//! $$
//! are IID $\mathcal{N}(0, \sigma^2)$ under the null of parameter constancy.
//! Two cumulative-sum statistics monitor parameter stability:
//!
//! - **CUSUM** $W_t = \hat\sigma^{-1} \sum_{j=k+1}^t w_j$ — sensitive to a
//!   shift in the conditional mean. Rejection boundary (two-sided) is
//!   $\pm[a_\alpha \sqrt{T-k} + 2 a_\alpha (t-k)/\sqrt{T-k}]$ with
//!   $a_\alpha \in \{1.143, 0.948, 0.850\}$ for $\alpha \in \{0.01, 0.05, 0.10\}$
//!   (Brown-Durbin-Evans 1975, Table 1).
//! - **CUSUMQ** $S_t = \sum_{j=k+1}^t w_j^2 / \sum_{j=k+1}^T w_j^2$ —
//!   sensitive to a shift in the conditional variance. Boundary is parallel
//!   to the line $E[S_t] = (t-k)/(T-k)$ with half-width
//!   $c_\alpha \sqrt{2/(T-k-2)}$ where $c_\alpha \in \{1.07275, 0.84717,
//!   0.74346\}$ (Edgerton-Wells 1994 large-sample approximation to the
//!   Durbin 1969 exact table).
//!
//! Both tests reject when the corresponding cumulative path crosses its
//! boundary; the first crossing point is reported as `breakpoint`.
//!
//! References:
//! - Brown, R.L., Durbin, J., Evans, J.M. (1975), "Techniques for testing
//!   the constancy of regression relationships over time", *JRSS B* 37(2),
//!   149-192.
//! - Durbin, J. (1969), "Tests for serial correlation in regression analysis
//!   based on the periodogram of least-squares residuals", *Biometrika*
//!   56(1), 1-15.
//! - Edgerton, D., Wells, C. (1994), "Critical values for the CUSUMSQ
//!   statistic in medium and large sized samples", *Oxford Bull. Econ. Stat.*
//!   56(3), 355-365.

use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray_linalg::Inverse;

use super::common::validate_series;

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

/// Result of the CUSUM / CUSUMQ test.
#[derive(Debug, Clone)]
pub struct CusumResult {
  /// Variant that produced this result.
  pub variant: CusumVariant,
  /// Maximum absolute excess of the cumulative path over its boundary —
  /// positive when the path exits the band, non-positive otherwise. The
  /// natural reporting statistic mirrors the convention in Hansen (1992).
  pub statistic: f64,
  /// First time index (in the original sample, 1-based) at which the
  /// boundary is crossed, or `None` when the null is not rejected.
  pub breakpoint: Option<usize>,
  /// Recursive residuals $w_{k+1}, \dots, w_T$.
  pub recursive_residuals: Vec<f64>,
  /// Cumulative-sum path values, length $T - k$.
  pub cusum_path: Vec<f64>,
  /// Two-sided rejection boundary `(lower, upper)` matching `cusum_path`.
  pub critical_boundary: Vec<(f64, f64)>,
  /// Whether the null of parameter constancy is rejected at `alpha`.
  pub null_rejected: bool,
}

impl crate::traits::HypothesisTest for CusumResult {
  fn statistic(&self) -> f64 {
    self.statistic
  }
  fn null_rejected(&self) -> Option<bool> {
    Some(self.null_rejected)
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

/// CUSUMQ boundary coefficient $c_\alpha$ from the Edgerton-Wells (1994)
/// large-sample approximation; the half-width of the band is
/// $c_\alpha \sqrt{2/(T-k-2)}$.
fn cusumq_critical_coefficient(alpha: f64) -> f64 {
  if alpha <= 0.01 {
    1.07275
  } else if alpha <= 0.05 {
    0.84717
  } else {
    0.74346
  }
}

/// Brown-Durbin-Evans (1975) CUSUM / CUSUMQ structural-change test.
///
/// `x` is the $T \times k$ design matrix and `y` the $T \times 1$ response.
/// The initial $k$ observations are consumed to seed the recursive estimator,
/// producing $T - k$ residuals.
///
/// # Panics
/// Panics on non-finite inputs, when $T \le k + 1$, when the leading
/// $X_k$ block is singular, or when `alpha` is outside $(0, 1)$.
pub fn cusum_test(y: ArrayView1<f64>, x: ArrayView2<f64>, cfg: CusumConfig) -> CusumResult {
  let y_slice = y
    .as_slice()
    .expect("cusum_test requires a contiguous ArrayView1");
  assert_eq!(y.len(), x.nrows(), "y/x row mismatch");
  let k = x.ncols();
  assert!(k >= 1, "design matrix must have at least one regressor");
  validate_series(y_slice, k + 2);
  assert!(
    cfg.alpha > 0.0 && cfg.alpha < 1.0,
    "alpha must be in (0, 1)"
  );

  let t_total = y.len();
  let w = recursive_residuals(y.view(), x.view());
  debug_assert_eq!(w.len(), t_total - k);

  let n_tilde_usize = t_total - k;
  let n_tilde = n_tilde_usize as f64;

  match cfg.variant {
    CusumVariant::Cusum => cusum_branch(w, n_tilde, k, cfg.alpha),
    CusumVariant::Cusumq => cusumq_branch(w, n_tilde, k, cfg.alpha),
  }
}

fn cusum_branch(w: Vec<f64>, n_tilde: f64, k: usize, alpha: f64) -> CusumResult {
  let w_mean = w.iter().sum::<f64>() / n_tilde;
  let sigma2 = w.iter().map(|wj| (wj - w_mean).powi(2)).sum::<f64>() / (n_tilde - 1.0).max(1.0);
  let sigma = sigma2.max(0.0).sqrt();
  let sigma_safe = if sigma > 0.0 { sigma } else { 1.0 };

  let a0 = cusum_critical_coefficient(alpha);
  let sqrt_n = n_tilde.sqrt();

  let mut path = Vec::with_capacity(w.len());
  let mut boundary = Vec::with_capacity(w.len());
  let mut cum = 0.0_f64;
  let mut max_excess = f64::NEG_INFINITY;
  let mut breakpoint: Option<usize> = None;

  for (i, wi) in w.iter().enumerate() {
    cum += wi / sigma_safe;
    path.push(cum);
    let upper = a0 * sqrt_n + 2.0 * a0 * (i as f64 + 1.0) / sqrt_n;
    boundary.push((-upper, upper));
    let excess = cum.abs() - upper;
    if excess > max_excess {
      max_excess = excess;
    }
    if excess > 0.0 && breakpoint.is_none() {
      breakpoint = Some(k + i + 1);
    }
  }

  CusumResult {
    variant: CusumVariant::Cusum,
    statistic: max_excess,
    breakpoint,
    recursive_residuals: w,
    cusum_path: path,
    critical_boundary: boundary,
    null_rejected: max_excess > 0.0,
  }
}

fn cusumq_branch(w: Vec<f64>, n_tilde: f64, k: usize, alpha: f64) -> CusumResult {
  let denom: f64 = w.iter().map(|wj| wj * wj).sum();
  let denom_safe = if denom > 0.0 { denom } else { 1.0 };
  let c_alpha = cusumq_critical_coefficient(alpha);
  let half_band = c_alpha * (2.0_f64 / (n_tilde - 2.0).max(1.0)).sqrt();

  let mut path = Vec::with_capacity(w.len());
  let mut boundary = Vec::with_capacity(w.len());
  let mut cum_sq = 0.0_f64;
  let mut max_excess = f64::NEG_INFINITY;
  let mut breakpoint: Option<usize> = None;

  for (i, wi) in w.iter().enumerate() {
    cum_sq += wi * wi;
    let s = cum_sq / denom_safe;
    let mean = (i as f64 + 1.0) / n_tilde;
    path.push(s);
    boundary.push((mean - half_band, mean + half_band));
    let excess = (s - mean).abs() - half_band;
    if excess > max_excess {
      max_excess = excess;
    }
    if excess > 0.0 && breakpoint.is_none() {
      breakpoint = Some(k + i + 1);
    }
  }

  CusumResult {
    variant: CusumVariant::Cusumq,
    statistic: max_excess,
    breakpoint,
    recursive_residuals: w,
    cusum_path: path,
    critical_boundary: boundary,
    null_rejected: max_excess > 0.0,
  }
}

/// Recursive residuals via incremental OLS with a Sherman-Morrison update
/// of $(X_t^\top X_t)^{-1}$.
fn recursive_residuals(y: ArrayView1<f64>, x: ArrayView2<f64>) -> Vec<f64> {
  let t_total = y.len();
  let k = x.ncols();
  assert!(t_total > k, "need T > k observations");

  let x_init = x.slice(ndarray::s![..k, ..]).to_owned();
  let y_init = y.slice(ndarray::s![..k]).to_owned();
  let mut xtx_inv = x_init
    .t()
    .dot(&x_init)
    .inv()
    .expect("CUSUM: leading X_k block is singular");
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

  w
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use ndarray_rand::RandomExt;
  use ndarray_rand::rand_distr::Normal;
  use rand::SeedableRng;
  use rand::rngs::StdRng;

  use super::*;

  fn linear_design(n: usize) -> ndarray::Array2<f64> {
    Array2::from_shape_fn(
      (n, 2),
      |(i, j)| if j == 0 { 1.0 } else { i as f64 / n as f64 },
    )
  }

  fn simulate_stationary(n: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let x = linear_design(n);
    let beta = ndarray::array![0.5, 1.2];
    let noise = Array1::random_using(n, Normal::new(0.0, 1.0).unwrap(), &mut rng);
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
    let mut rng = StdRng::seed_from_u64(seed);
    let x = linear_design(n);
    let beta = ndarray::array![0.5, 1.2];
    let mu = x.dot(&beta);
    let mut y = Array1::zeros(n);
    for i in 0..n {
      let s = if i < shift_at { 1.0 } else { sigma_high };
      let eps =
        ndarray_rand::rand_distr::Distribution::sample(&Normal::new(0.0, s).unwrap(), &mut rng);
      y[i] = mu[i] + eps;
    }
    y
  }

  #[test]
  fn cusum_no_reject_under_constant_dgp() {
    // α = 0.01 keeps Type I error rate at ~1 %; with three seeds the joint
    // false-positive rate is well under 5 % even before any minimum-power
    // adjustment.
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
      );
      assert!(
        !res.null_rejected,
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
    );
    assert!(
      res.null_rejected,
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
    );
    assert!(
      res.null_rejected,
      "CUSUMQ should reject variance shift: statistic = {}",
      res.statistic
    );
  }

  #[test]
  fn cusumq_does_not_reject_constant_dgp() {
    // α = 0.01 — see [`cusum_no_reject_under_constant_dgp`] for rationale.
    let n = 200;
    let x = linear_design(n);
    for seed in [5_u64, 42, 99] {
      let y = simulate_stationary(n, seed);
      let res = cusum_test(
        y.view(),
        x.view(),
        CusumConfig {
          variant: CusumVariant::Cusumq,
          alpha: 0.01,
        },
      );
      assert!(
        !res.null_rejected,
        "CUSUMQ should not reject constant DGP at seed {seed}: statistic = {}",
        res.statistic
      );
    }
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
    );
    let w = &res.recursive_residuals;
    let w_mean = w.iter().sum::<f64>() / w.len() as f64;
    let s2_w = w.iter().map(|wj| (wj - w_mean).powi(2)).sum::<f64>() / (w.len() as f64 - 1.0);

    // OLS σ² on the full sample.
    let beta = x.t().dot(&x).inv().unwrap().dot(&x.t().dot(&y));
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
