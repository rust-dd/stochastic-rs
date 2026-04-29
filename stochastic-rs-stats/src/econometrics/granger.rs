//! Granger causality test based on a VAR(p) F-test of nested OLS.
//!
//! Tests $H_0$: lags of $X$ do not Granger-cause $Y$ versus $H_1$ that they
//! do, comparing restricted ($Y$ on its own lags) and unrestricted ($Y$ on
//! its own lags + lags of $X$) regression sums of squares.
//!
//! Reference: Granger, "Investigating Causal Relations by Econometric Models
//! and Cross-Spectral Methods", Econometrica, 37(3), 424-438 (1969).
//! DOI: 10.2307/1912791

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray_linalg::LeastSquaresSvd;
use stochastic_rs_distributions::special::beta_i as beta_reg;

/// Result of a Granger-causality F-test.
#[derive(Debug, Clone)]
pub struct GrangerResult {
  /// $F$ statistic.
  pub f_statistic: f64,
  /// $p$-value under $\mathcal F_{p, T-2p-1}$.
  pub p_value: f64,
  /// Number of lag terms tested.
  pub lags: usize,
  /// Effective sample size (rows of the VAR design).
  pub nobs: usize,
  /// Whether $H_0$ (no Granger causality) is rejected at the supplied
  /// significance level.
  pub reject_no_causality: bool,
}

/// Granger-causality test of "does X cause Y?" with `lags` autoregressive
/// terms on each variable.
pub fn granger_causality(
  y: ArrayView1<f64>,
  x: ArrayView1<f64>,
  lags: usize,
  alpha: f64,
) -> GrangerResult {
  assert_eq!(y.len(), x.len(), "y and x must have equal length");
  assert!(lags >= 1, "lags must be at least 1");
  assert!(alpha > 0.0 && alpha < 1.0, "alpha must lie in (0, 1)");
  let n_total = y.len();
  let n = n_total - lags;
  assert!(n > 2 * lags + 2, "not enough observations for given lag");

  let mut design_unrest = Array2::<f64>::zeros((n, 1 + 2 * lags));
  let mut target = Array1::<f64>::zeros(n);
  for i in 0..n {
    target[i] = y[lags + i];
    design_unrest[[i, 0]] = 1.0;
    for l in 1..=lags {
      design_unrest[[i, l]] = y[lags + i - l];
      design_unrest[[i, lags + l]] = x[lags + i - l];
    }
  }
  let mut design_rest = Array2::<f64>::zeros((n, 1 + lags));
  for i in 0..n {
    design_rest[[i, 0]] = 1.0;
    for l in 1..=lags {
      design_rest[[i, l]] = y[lags + i - l];
    }
  }
  let rss_unrest = ols_rss(&design_unrest, &target);
  let rss_rest = ols_rss(&design_rest, &target);
  let q = lags as f64;
  let dof = (n as f64) - (1.0 + 2.0 * q);
  let f = ((rss_rest - rss_unrest) / q) / (rss_unrest / dof.max(1.0));
  let p_value = if f.is_finite() && f > 0.0 {
    let d1 = q;
    let d2 = dof.max(1.0);
    let x = (d1 * f) / (d1 * f + d2);
    1.0 - beta_reg(d1 / 2.0, d2 / 2.0, x)
  } else {
    1.0
  };
  GrangerResult {
    f_statistic: f.max(0.0),
    p_value,
    lags,
    nobs: n,
    reject_no_causality: p_value < alpha,
  }
}

fn ols_rss(x: &Array2<f64>, y: &Array1<f64>) -> f64 {
  let sol = x.least_squares(y).expect("Granger OLS failed");
  let beta = sol.solution;
  let yhat = x.dot(&beta);
  (y - &yhat).iter().map(|v| v * v).sum::<f64>()
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  #[test]
  fn granger_does_not_reject_independent_series() {
    let dist = SimdNormal::<f64>::with_seed(0.0, 1.0, 7);
    let mut x_buf = vec![0.0_f64; 500];
    dist.fill_slice_fast(&mut x_buf);
    let dist2 = SimdNormal::<f64>::with_seed(0.0, 1.0, 13);
    let mut y_buf = vec![0.0_f64; 500];
    dist2.fill_slice_fast(&mut y_buf);
    let x = Array1::from(x_buf);
    let y = Array1::from(y_buf);
    let res = granger_causality(y.view(), x.view(), 4, 0.05);
    assert!(!res.reject_no_causality);
  }

  #[test]
  fn granger_rejects_when_x_drives_y() {
    let dist = SimdNormal::<f64>::with_seed(0.0, 1.0, 17);
    let mut x_buf = vec![0.0_f64; 500];
    dist.fill_slice_fast(&mut x_buf);
    let dist_eps = SimdNormal::<f64>::with_seed(0.0, 0.3, 19);
    let mut eps = vec![0.0_f64; 500];
    dist_eps.fill_slice_fast(&mut eps);
    let mut y = vec![0.0_f64; 500];
    for i in 2..500 {
      y[i] = 0.5 * y[i - 1] + 0.7 * x_buf[i - 1] - 0.3 * x_buf[i - 2] + eps[i];
    }
    let xa = Array1::from(x_buf);
    let ya = Array1::from(y);
    let res = granger_causality(ya.view(), xa.view(), 2, 0.05);
    assert!(res.reject_no_causality);
  }
}
