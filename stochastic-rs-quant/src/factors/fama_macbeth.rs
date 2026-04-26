//! Fama-MacBeth two-pass cross-sectional regression for factor risk premia.
//!
//! 1. **Time-series regression** — for each asset $i$, estimate
//!    $R_{i,t} = a_i + \beta_i^\top F_t + \varepsilon_{i,t}$ by OLS.
//! 2. **Cross-sectional regression** — at each $t$, regress
//!    $R_{i,t} = \gamma_{0,t} + \gamma_t^\top \beta_i + \eta_{i,t}$ to obtain
//!    factor risk premia $\gamma_t$.
//! 3. **Inference** — average $\bar\gamma = \tfrac{1}{T}\sum_t\gamma_t$ with
//!    standard errors $\sqrt{\widehat{\mathrm{Var}}(\gamma_t) / T}$.
//!
//! Reference: Fama, MacBeth, "Risk, Return, and Equilibrium: Empirical Tests",
//! Journal of Political Economy, 81(3), 607-636 (1973). DOI: 10.1086/260061

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray_linalg::LeastSquaresSvd;

/// Result of a Fama-MacBeth two-pass regression.
#[derive(Debug, Clone)]
pub struct FamaMacBethResult {
  /// Mean cross-sectional risk-premium estimate, including the intercept term
  /// ($\bar\gamma_0, \bar\gamma_1, \dots$).
  pub gamma: Array1<f64>,
  /// Time-series-based standard errors $\sqrt{\widehat{\mathrm{Var}}(\gamma_t)/T}$.
  pub std_errors: Array1<f64>,
  /// $t$-statistics $\bar\gamma_k / SE_k$.
  pub t_statistics: Array1<f64>,
  /// Per-period risk-premium series $\gamma_t$ (rows = time, cols = factors+1).
  pub gamma_series: Array2<f64>,
  /// First-pass time-series betas (rows = assets, cols = factors+1).
  pub betas: Array2<f64>,
}

/// Run a Fama-MacBeth regression. `returns[t, i]` is asset `i`'s return at
/// time `t`; `factors[t, k]` is factor `k`'s realisation at time `t`.
pub fn fama_macbeth(returns: ArrayView2<f64>, factors: ArrayView2<f64>) -> FamaMacBethResult {
  let (t, n) = returns.dim();
  let (tt, k) = factors.dim();
  assert_eq!(
    t, tt,
    "returns and factors must have the same number of rows"
  );
  assert!(t >= k + 2, "not enough observations");
  let mut design_ts = Array2::<f64>::zeros((t, k + 1));
  for r in 0..t {
    design_ts[[r, 0]] = 1.0;
    for c in 0..k {
      design_ts[[r, c + 1]] = factors[[r, c]];
    }
  }
  let mut betas = Array2::<f64>::zeros((n, k + 1));
  for asset in 0..n {
    let y: Array1<f64> = returns.column(asset).to_owned();
    let sol = design_ts.least_squares(&y).expect("first-pass OLS failed");
    for j in 0..(k + 1) {
      betas[[asset, j]] = sol.solution[j];
    }
  }
  let mut design_xs = Array2::<f64>::zeros((n, k + 1));
  for asset in 0..n {
    design_xs[[asset, 0]] = 1.0;
    for j in 0..k {
      design_xs[[asset, j + 1]] = betas[[asset, j + 1]];
    }
  }
  let mut gamma_series = Array2::<f64>::zeros((t, k + 1));
  for time in 0..t {
    let y: Array1<f64> = returns.row(time).to_owned();
    let sol = design_xs.least_squares(&y).expect("second-pass OLS failed");
    for j in 0..(k + 1) {
      gamma_series[[time, j]] = sol.solution[j];
    }
  }
  let mut gamma_mean = Array1::<f64>::zeros(k + 1);
  for j in 0..(k + 1) {
    gamma_mean[j] = gamma_series.column(j).iter().sum::<f64>() / t as f64;
  }
  let mut std_errors = Array1::<f64>::zeros(k + 1);
  for j in 0..(k + 1) {
    let m = gamma_mean[j];
    let var = gamma_series
      .column(j)
      .iter()
      .map(|v| (v - m).powi(2))
      .sum::<f64>()
      / (t as f64 - 1.0);
    std_errors[j] = (var / t as f64).sqrt();
  }
  let mut t_stats = Array1::<f64>::zeros(k + 1);
  for j in 0..(k + 1) {
    t_stats[j] = if std_errors[j] > 0.0 {
      gamma_mean[j] / std_errors[j]
    } else {
      0.0
    };
  }
  FamaMacBethResult {
    gamma: gamma_mean,
    std_errors,
    t_statistics: t_stats,
    gamma_series,
    betas,
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use stochastic_rs_distributions::normal::SimdNormal;

  #[test]
  fn fama_macbeth_recovers_factor_premium() {
    let t = 600usize;
    let n = 30usize;
    let k = 1usize;
    let true_premium = 0.005_f64;
    let true_intercept = 0.001_f64;
    let factor_dist = SimdNormal::<f64>::with_seed(0.0, 0.02, 1);
    let mut factors_buf = vec![0.0_f64; t];
    factor_dist.fill_slice_fast(&mut factors_buf);
    let factors = Array2::from_shape_vec((t, k), factors_buf.clone()).unwrap();
    let beta_dist = SimdNormal::<f64>::with_seed(1.0, 0.3, 2);
    let mut betas_buf = vec![0.0_f64; n];
    beta_dist.fill_slice_fast(&mut betas_buf);
    let resid_dist = SimdNormal::<f64>::with_seed(0.0, 0.005, 3);
    let mut resid_buf = vec![0.0_f64; t * n];
    resid_dist.fill_slice_fast(&mut resid_buf);
    let mut returns = Array2::<f64>::zeros((t, n));
    for time in 0..t {
      for asset in 0..n {
        let f = factors_buf[time];
        let beta_i = betas_buf[asset];
        let resid = resid_buf[time * n + asset];
        returns[[time, asset]] = true_intercept + beta_i * (true_premium + f) + resid;
      }
    }
    let res = fama_macbeth(returns.view(), factors.view());
    assert!((res.gamma[1] - true_premium).abs() < 0.002);
    assert!(res.t_statistics[1].abs() > 1.5);
    assert!(res.std_errors.iter().all(|&v| v.is_finite()));
  }

  #[test]
  fn fama_macbeth_betas_match_first_pass_ols_dimensions() {
    let dist = SimdNormal::<f64>::with_seed(0.0, 0.01, 5);
    let mut buf = vec![0.0_f64; 120 * 5];
    dist.fill_slice_fast(&mut buf);
    let returns = Array2::from_shape_vec((120, 5), buf).unwrap();
    let mut fbuf = vec![0.0_f64; 120 * 2];
    dist.fill_slice_fast(&mut fbuf);
    let factors = Array2::from_shape_vec((120, 2), fbuf).unwrap();
    let res = fama_macbeth(returns.view(), factors.view());
    assert_eq!(res.betas.dim(), (5, 3));
    assert_eq!(res.gamma.len(), 3);
    assert_eq!(res.gamma_series.dim(), (120, 3));
  }
}
