use nalgebra::DMatrix;
use nalgebra::DVector;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeterministicTerm {
  None,
  Constant,
  ConstantTrend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LagSelection {
  Fixed(usize),
  Aic,
  Bic,
  TStat,
}

#[derive(Debug, Clone, Copy)]
pub struct CriticalValues {
  pub one_percent: f64,
  pub five_percent: f64,
  pub ten_percent: f64,
}

impl CriticalValues {
  pub fn value_at(self, alpha: f64) -> f64 {
    if alpha <= 0.01 {
      self.one_percent
    } else if alpha <= 0.05 {
      self.five_percent
    } else {
      self.ten_percent
    }
  }
}

#[derive(Debug, Clone)]
pub struct OlsResult {
  pub beta: Vec<f64>,
  pub std_err: Vec<f64>,
  pub residuals: Vec<f64>,
  pub sse: f64,
  pub sigma2: f64,
  pub nobs: usize,
  pub k: usize,
}

#[derive(Debug, Clone)]
pub struct AdfFit {
  pub lag: usize,
  pub statistic: f64,
  pub gamma: f64,
  pub std_err_gamma: f64,
  pub nobs: usize,
  pub residuals: Vec<f64>,
  pub sigma2: f64,
}

pub fn validate_series(y: &[f64], min_n: usize) {
  assert!(
    y.len() >= min_n,
    "series must have at least {min_n} observations"
  );
  assert!(
    y.iter().all(|v| v.is_finite()),
    "series must contain only finite values"
  );
}

pub fn difference(y: &[f64]) -> Vec<f64> {
  y.windows(2).map(|w| w[1] - w[0]).collect()
}

pub fn schwert_max_lags(n: usize) -> usize {
  if n <= 1 {
    return 0;
  }
  (12.0 * (n as f64 / 100.0).powf(0.25)).floor() as usize
}

pub fn adf_critical_values(det: DeterministicTerm) -> CriticalValues {
  match det {
    // Asymptotic MacKinnon-style benchmark values used widely in practice.
    DeterministicTerm::None => CriticalValues {
      one_percent: -2.58,
      five_percent: -1.95,
      ten_percent: -1.62,
    },
    DeterministicTerm::Constant => CriticalValues {
      one_percent: -3.43,
      five_percent: -2.86,
      ten_percent: -2.57,
    },
    DeterministicTerm::ConstantTrend => CriticalValues {
      one_percent: -3.96,
      five_percent: -3.41,
      ten_percent: -3.13,
    },
  }
}

pub fn dfgls_critical_values(include_trend: bool) -> CriticalValues {
  if include_trend {
    CriticalValues {
      one_percent: -3.77,
      five_percent: -3.19,
      ten_percent: -2.89,
    }
  } else {
    // ERS (constant-only) critical values are close to ADF with constant.
    CriticalValues {
      one_percent: -2.58,
      five_percent: -1.95,
      ten_percent: -1.62,
    }
  }
}

pub fn ols(y: &[f64], x: &[Vec<f64>]) -> OlsResult {
  assert!(!y.is_empty(), "OLS requires non-empty response");
  assert_eq!(y.len(), x.len(), "OLS y/x row mismatch");
  let n = y.len();
  let k = x[0].len();
  assert!(k > 0, "OLS requires at least one regressor");
  assert!(
    x.iter().all(|row| row.len() == k),
    "OLS design matrix must be rectangular"
  );
  assert!(n > k, "OLS requires nobs > number of regressors");

  let mut flat_x = Vec::with_capacity(n * k);
  for row in x {
    flat_x.extend_from_slice(row);
  }

  let x_mat = DMatrix::from_row_slice(n, k, &flat_x);
  let y_vec = DVector::from_row_slice(y);

  let xtx = x_mat.transpose() * &x_mat;
  let Some(xtx_inv) = xtx.clone().try_inverse() else {
    panic!("OLS failed: singular design matrix")
  };

  let beta = &xtx_inv * x_mat.transpose() * &y_vec;
  let fitted = &x_mat * &beta;
  let residuals_vec = y_vec - fitted;

  let residuals: Vec<f64> = residuals_vec.iter().copied().collect();
  let sse = residuals.iter().map(|u| u * u).sum::<f64>();
  let dof = (n - k) as f64;
  let sigma2 = (sse / dof).max(0.0);

  let cov = xtx_inv * sigma2;
  let mut std_err = vec![0.0; k];
  for i in 0..k {
    std_err[i] = cov[(i, i)].max(0.0).sqrt();
  }

  OlsResult {
    beta: beta.iter().copied().collect(),
    std_err,
    residuals,
    sse,
    sigma2,
    nobs: n,
    k,
  }
}

fn build_adf_design(
  y: &[f64],
  lags: usize,
  det: DeterministicTerm,
) -> (Vec<f64>, Vec<Vec<f64>>, usize) {
  validate_series(y, 3 + lags);
  let dy = difference(y);
  let n_dy = dy.len();
  assert!(n_dy > lags, "too many lags for sample length");

  let mut lhs = Vec::with_capacity(n_dy - lags);
  let mut rhs = Vec::with_capacity(n_dy - lags);

  for t in lags..n_dy {
    lhs.push(dy[t]);

    let mut row = Vec::with_capacity(match det {
      DeterministicTerm::None => 1 + lags,
      DeterministicTerm::Constant => 2 + lags,
      DeterministicTerm::ConstantTrend => 3 + lags,
    });

    match det {
      DeterministicTerm::None => {}
      DeterministicTerm::Constant => row.push(1.0),
      DeterministicTerm::ConstantTrend => {
        row.push(1.0);
        row.push((t + 1) as f64);
      }
    }

    // y_{t-1} term in ADF regression, with dy-index t corresponding to original time t+1.
    row.push(y[t]);

    for i in 1..=lags {
      row.push(dy[t - i]);
    }

    rhs.push(row);
  }

  let gamma_index = match det {
    DeterministicTerm::None => 0,
    DeterministicTerm::Constant => 1,
    DeterministicTerm::ConstantTrend => 2,
  };

  (lhs, rhs, gamma_index)
}

pub fn fit_adf(y: &[f64], lags: usize, det: DeterministicTerm) -> AdfFit {
  let (lhs, rhs, gamma_index) = build_adf_design(y, lags, det);
  let ols_fit = ols(&lhs, &rhs);

  let gamma = ols_fit.beta[gamma_index];
  let se = ols_fit.std_err[gamma_index];
  let statistic = if se > 0.0 { gamma / se } else { f64::NAN };

  AdfFit {
    lag: lags,
    statistic,
    gamma,
    std_err_gamma: se,
    nobs: ols_fit.nobs,
    residuals: ols_fit.residuals,
    sigma2: ols_fit.sigma2,
  }
}

pub fn aic_from_sse(sse: f64, nobs: usize, k: usize) -> f64 {
  let n = nobs as f64;
  n * (sse / n).ln() + 2.0 * k as f64
}

pub fn bic_from_sse(sse: f64, nobs: usize, k: usize) -> f64 {
  let n = nobs as f64;
  n * (sse / n).ln() + (k as f64) * n.ln()
}

pub fn choose_lag_for_adf(
  y: &[f64],
  det: DeterministicTerm,
  lag_selection: LagSelection,
  max_lags: usize,
) -> usize {
  if let LagSelection::Fixed(p) = lag_selection {
    return p;
  }

  let mut best_lag = 0usize;
  let mut best_score = f64::INFINITY;

  let mut candidates: Vec<(usize, f64, f64)> = Vec::new();
  // (lag, score_for_ic, tstat_last_lag)

  for lag in 0..=max_lags {
    let (lhs, rhs, _gamma_idx) = build_adf_design(y, lag, det);
    if rhs.is_empty() {
      continue;
    }
    let fit = ols(&lhs, &rhs);

    let ic = match lag_selection {
      LagSelection::Aic => aic_from_sse(fit.sse, fit.nobs, fit.k),
      LagSelection::Bic => bic_from_sse(fit.sse, fit.nobs, fit.k),
      LagSelection::Fixed(_) | LagSelection::TStat => 0.0,
    };

    let t_last = if lag > 0 {
      let idx = fit.k - 1;
      let se = fit.std_err[idx];
      if se > 0.0 { fit.beta[idx] / se } else { 0.0 }
    } else {
      0.0
    };

    if matches!(lag_selection, LagSelection::Aic | LagSelection::Bic) && ic < best_score {
      best_score = ic;
      best_lag = lag;
    }

    candidates.push((lag, ic, t_last));
  }

  match lag_selection {
    LagSelection::Aic | LagSelection::Bic => best_lag,
    LagSelection::TStat => {
      let mut ordered = candidates;
      ordered.sort_by_key(|(lag, _, _)| *lag);
      for (lag, _, t_last) in ordered.into_iter().rev() {
        if lag == 0 || t_last.abs() >= 1.644_853_626_951_472_2 {
          return lag;
        }
      }
      0
    }
    LagSelection::Fixed(_) => unreachable!(),
  }
}

pub fn newey_west_long_run_variance(u: &[f64], lags: usize) -> f64 {
  assert!(
    !u.is_empty(),
    "Newey-West requires non-empty residual series"
  );
  let n = u.len();
  let n_f = n as f64;

  let gamma0 = u.iter().map(|v| v * v).sum::<f64>() / n_f;
  let mut lr_var = gamma0;

  for j in 1..=lags {
    if j >= n {
      break;
    }
    let weight = 1.0 - (j as f64) / (lags as f64 + 1.0);
    let mut cov = 0.0;
    for t in j..n {
      cov += u[t] * u[t - j];
    }
    cov /= n_f;
    lr_var += 2.0 * weight * cov;
  }

  if lr_var <= 0.0 || !lr_var.is_finite() {
    gamma0.max(1e-12)
  } else {
    lr_var
  }
}

pub fn regress_on_deterministics(y: &[f64], include_trend: bool) -> OlsResult {
  validate_series(y, if include_trend { 3 } else { 2 });
  let n = y.len();
  let mut x = Vec::with_capacity(n);
  for t in 0..n {
    if include_trend {
      x.push(vec![1.0, (t + 1) as f64]);
    } else {
      x.push(vec![1.0]);
    }
  }
  ols(y, &x)
}

pub fn fit_ar(y: &[f64], lags: usize) -> (Vec<f64>, Vec<f64>, f64) {
  assert!(lags > 0, "AR lag order must be at least 1");
  validate_series(y, lags + 3);

  let n = y.len();
  let nobs = n - lags;
  let mut lhs = Vec::with_capacity(nobs);
  let mut rhs = Vec::with_capacity(nobs);

  for t in lags..n {
    lhs.push(y[t]);
    let mut row = Vec::with_capacity(lags);
    for i in 1..=lags {
      row.push(y[t - i]);
    }
    rhs.push(row);
  }

  let fit = ols(&lhs, &rhs);
  let sigma = fit.sigma2.max(0.0).sqrt();
  (fit.beta, fit.residuals, sigma)
}
