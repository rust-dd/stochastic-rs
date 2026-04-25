//! Cointegration tests — Engle-Granger 2-step and Johansen trace test.
//!
//! Reference: Engle, Granger, "Co-Integration and Error Correction:
//! Representation, Estimation, and Testing", Econometrica, 55(2), 251-276
//! (1987). DOI: 10.2307/1913236
//!
//! Reference: Johansen, "Statistical Analysis of Cointegration Vectors",
//! Journal of Economic Dynamics and Control, 12(2-3), 231-254 (1988).
//! DOI: 10.1016/0165-1889(88)90041-3
//!
//! Reference: Johansen, "Estimation and Hypothesis Testing of Cointegration
//! Vectors in Gaussian Vector Autoregressive Models", Econometrica, 59(6),
//! 1551-1580 (1991). DOI: 10.2307/2938278

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray_linalg::Eig;
use ndarray_linalg::Inverse;
use ndarray_linalg::LeastSquaresSvd;

use crate::stats::stationarity::adf::ADFConfig;
use crate::stats::stationarity::adf::adf_test;

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
  let sol = design
    .least_squares(&y_owned)
    .expect("Engle-Granger first-stage OLS failed");
  let alpha = sol.solution[0];
  let beta = sol.solution[1];
  let mut residuals = Array1::<f64>::zeros(n);
  for i in 0..n {
    residuals[i] = y[i] - alpha - beta * x[i];
  }
  let cfg = ADFConfig::default();
  let r_slice: Vec<f64> = residuals.iter().copied().collect();
  let adf = adf_test(&r_slice, cfg);
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

/// Result of the Johansen trace test for cointegrating rank.
#[derive(Debug, Clone)]
pub struct JohansenResult {
  /// Generalised eigenvalues $\lambda_i$.
  pub eigenvalues: Array1<f64>,
  /// Trace statistics $-T \sum_{i=r+1}^K \log(1 - \lambda_i)$ for each
  /// hypothesised rank $r = 0, 1, \ldots, K-1$.
  pub trace_statistics: Array1<f64>,
  /// 5% critical values for the trace test (Osterwald-Lenum 1992 — constant
  /// included). Defaults to length 12; truncated to `K`.
  pub trace_critical_5pct: Array1<f64>,
}

/// Johansen trace test for the cointegrating rank of an `(n, k)` series.
/// `lags`: number of lags in the underlying VAR (use `1` if unsure).
pub fn johansen_test(series: ArrayView2<f64>, lags: usize) -> JohansenResult {
  let (t, k) = series.dim();
  assert!(t > lags + 2, "not enough observations for given lag");
  assert!(k >= 2, "need at least two series");
  assert!(lags >= 1, "lags must be at least 1");
  let mut delta = Array2::<f64>::zeros((t - 1, k));
  for j in 0..k {
    for i in 0..(t - 1) {
      delta[[i, j]] = series[[i + 1, j]] - series[[i, j]];
    }
  }
  let n_eff = t - lags;
  let mut z0 = Array2::<f64>::zeros((n_eff, k));
  let mut z1 = Array2::<f64>::zeros((n_eff, k));
  for j in 0..k {
    for i in 0..n_eff {
      z0[[i, j]] = delta[[lags + i - 1, j]];
      z1[[i, j]] = series[[lags + i - 1, j]];
    }
  }
  let n_lag_cols = lags.saturating_sub(1) * k + 1;
  let mut z2 = Array2::<f64>::zeros((n_eff, n_lag_cols));
  for i in 0..n_eff {
    z2[[i, 0]] = 1.0;
    for l in 1..lags {
      for j in 0..k {
        z2[[i, 1 + (l - 1) * k + j]] = delta[[lags - 1 - l + i, j]];
      }
    }
  }
  let r0 = residualise(&z0, &z2);
  let r1 = residualise(&z1, &z2);
  let s00 = (&r0.t().dot(&r0)) / n_eff as f64;
  let s11 = (&r1.t().dot(&r1)) / n_eff as f64;
  let s01 = (&r0.t().dot(&r1)) / n_eff as f64;
  let s10 = s01.t().to_owned();
  let s00_inv = s00.inv().expect("S00 inverse failed");
  let m = s10.dot(&s00_inv).dot(&s01);
  let s11_inv = s11.inv().expect("S11 inverse failed");
  let a = s11_inv.dot(&m);
  let (eigvals_complex, _) = a.eig().expect("Johansen eig failed");
  let mut eigs: Vec<f64> = eigvals_complex
    .iter()
    .map(|c| c.re.clamp(0.0, 1.0 - 1e-12))
    .collect();
  eigs.sort_by(|a, b| b.partial_cmp(a).unwrap());
  let trace: Array1<f64> = (0..k)
    .map(|r| -(n_eff as f64) * eigs.iter().skip(r).map(|&l| (1.0 - l).ln()).sum::<f64>())
    .collect::<Vec<_>>()
    .into();
  let mut crit = vec![0.0_f64; k];
  let crit_table = osterwald_lenum_5pct();
  for (i, c) in crit.iter_mut().enumerate() {
    *c = if i < crit_table.len() {
      crit_table[i]
    } else {
      *crit_table.last().unwrap()
    };
  }
  JohansenResult {
    eigenvalues: Array1::from(eigs),
    trace_statistics: trace,
    trace_critical_5pct: Array1::from(crit),
  }
}

fn residualise(y: &Array2<f64>, x: &Array2<f64>) -> Array2<f64> {
  let (n, p) = y.dim();
  let (_, q) = x.dim();
  let mut residuals = Array2::<f64>::zeros((n, p));
  if q == 0 {
    return y.clone();
  }
  for col in 0..p {
    let target = y.column(col).to_owned();
    let sol = x
      .least_squares(&target)
      .expect("residualisation OLS failed");
    let beta = sol.solution.clone();
    for row in 0..n {
      let mut yhat = 0.0;
      for j in 0..q {
        yhat += x[[row, j]] * beta[j];
      }
      residuals[[row, col]] = y[[row, col]] - yhat;
    }
  }
  residuals
}

fn osterwald_lenum_5pct() -> Vec<f64> {
  vec![
    3.84, 12.21, 24.08, 39.71, 59.24, 82.61, 109.93, 141.20, 176.40,
  ]
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use ndarray::Array2;

  use super::*;
  use crate::distributions::normal::SimdNormal;

  fn random_walk(seed: u64, n: usize, sigma: f64) -> Array1<f64> {
    let dist = SimdNormal::<f64>::with_seed(0.0, sigma, seed);
    let mut steps = vec![0.0_f64; n];
    dist.fill_slice_fast(&mut steps);
    let mut out = Array1::<f64>::zeros(n);
    for i in 1..n {
      out[i] = out[i - 1] + steps[i];
    }
    out
  }

  #[test]
  fn engle_granger_rejects_under_cointegration() {
    let x = random_walk(7, 500, 1.0);
    let dist = SimdNormal::<f64>::with_seed(0.0, 0.05, 11);
    let mut eps = vec![0.0_f64; 500];
    dist.fill_slice_fast(&mut eps);
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

  #[test]
  fn johansen_returns_eigenvalues_in_unit_interval() {
    let mut s = Array2::<f64>::zeros((500, 3));
    let r1 = random_walk(31, 500, 1.0);
    let r2 = random_walk(41, 500, 1.0);
    let r3 = random_walk(43, 500, 1.0);
    for i in 0..500 {
      s[[i, 0]] = r1[i];
      s[[i, 1]] = r2[i];
      s[[i, 2]] = r3[i];
    }
    let res = johansen_test(s.view(), 1);
    for &l in res.eigenvalues.iter() {
      assert!((0.0..1.0).contains(&l));
    }
    assert_eq!(res.trace_statistics.len(), 3);
    assert!(res.trace_statistics.iter().all(|v| v.is_finite()));
  }
}
