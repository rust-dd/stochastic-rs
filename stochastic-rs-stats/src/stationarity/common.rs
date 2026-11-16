use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_distributions::linalg::inverse;
use stochastic_rs_distributions::linalg::solve;

use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::input::validate;
use crate::traits::RealExt;

/// Deterministic terms of a unit-root or stationarity regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
  None,
  Constant,
  ConstantTrend,
}

/// The lag of the ADF and DF-GLS regressions: fixed, or by AIC, BIC or the largest lag whose
/// `|t| ≥ 1.645`, every candidate up to `max_lags` fitted on the rows of `max_lags`.
/// Reference: Ng & Perron (2005), DOI: 10.1111/j.1468-0084.2005.00113.x
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LagSelection {
  Fixed(usize),
  Aic,
  Bic,
  TStat,
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
  pub statistic: f64,
  pub gamma: f64,
  pub std_err_gamma: f64,
  pub nobs: usize,
  pub residuals: Vec<f64>,
  pub sigma2: f64,
}

pub fn difference(y: &[f64]) -> Vec<f64> {
  y.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Shortest series for a levels autoregression of order `lags` plus `trend`'s terms: it keeps
/// `n − lags` rows, which must outnumber its `lags + terms` regressors; saturates at `usize::MAX`.
pub fn min_sample_size(lags: usize, trend: Trend) -> usize {
  let terms = match trend {
    Trend::None => 0,
    Trend::Constant => 1,
    Trend::ConstantTrend => 2,
  };
  lags.saturating_mul(2).saturating_add(terms + 1)
}

pub fn schwert_max_lags(n: usize) -> usize {
  if n <= 1 {
    return 0;
  }
  (12.0 * (n as f64 / 100.0).powf(0.25)).floor() as usize
}

pub fn adf_critical_values(trend: Trend) -> CriticalValues {
  match trend {
    Trend::None => CriticalValues::new([(0.01, -2.58), (0.05, -1.95), (0.10, -1.62)]),
    Trend::Constant => CriticalValues::new([(0.01, -3.43), (0.05, -2.86), (0.10, -2.57)]),
    Trend::ConstantTrend => CriticalValues::new([(0.01, -3.96), (0.05, -3.41), (0.10, -3.13)]),
  }
}

pub fn dfgls_critical_values(include_trend: bool) -> CriticalValues {
  if include_trend {
    CriticalValues::new([(0.01, -3.77), (0.05, -3.19), (0.10, -2.89)])
  } else {
    CriticalValues::new([(0.01, -2.58), (0.05, -1.95), (0.10, -1.62)])
  }
}

/// OLS of `y` on the rows of `x`; `Err(Singular)` on a rank-deficient design, and a NaN standard
/// error where an overflow or rounding leaves its variance NaN or negative.
pub fn ols(y: &[f64], x: &[Vec<f64>]) -> Result<OlsResult, StatsError> {
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

  let x_mat = Array2::from_shape_fn((n, k), |(i, j)| x[i][j]);
  let y_vec = Array1::from_vec(y.to_vec());

  let xtx = x_mat.t().dot(&x_mat);
  let xty = x_mat.t().dot(&y_vec);

  let beta_arr = solve(xtx.view(), xty.view()).ok_or(StatsError::Singular("OLS design"))?;
  let fitted = x_mat.dot(&beta_arr);
  let residuals_arr = &y_vec - &fitted;

  let residuals: Vec<f64> = residuals_arr.to_vec();
  let sse = residuals.iter().map(|u| u * u).sum::<f64>();
  let dof = (n - k) as f64;
  let sigma2 = sse / dof;

  let xtx_inv = inverse(xtx.view()).ok_or(StatsError::Singular("OLS design"))?;
  let cov = &xtx_inv * sigma2;
  let std_err = (0..k).map(|i| cov[[i, i]].sqrt()).collect();

  Ok(OlsResult {
    beta: beta_arr.to_vec(),
    std_err,
    residuals,
    sse,
    sigma2,
    nobs: n,
    k,
  })
}

/// The regressand, the regressor rows and the column of `y_{t-1}` in them.
type AdfDesign = (Vec<f64>, Vec<Vec<f64>>, usize);

fn build_adf_design(y: &[f64], lags: usize, trend: Trend) -> Result<AdfDesign, StatsError> {
  validate(y, 3 + lags)?;
  let dy = difference(y);
  let n_dy = dy.len();

  let mut lhs = Vec::with_capacity(n_dy - lags);
  let mut rhs = Vec::with_capacity(n_dy - lags);

  for t in lags..n_dy {
    lhs.push(dy[t]);

    let mut row = Vec::with_capacity(match trend {
      Trend::None => 1 + lags,
      Trend::Constant => 2 + lags,
      Trend::ConstantTrend => 3 + lags,
    });

    match trend {
      Trend::None => {}
      Trend::Constant => row.push(1.0),
      Trend::ConstantTrend => {
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

  let gamma_index = match trend {
    Trend::None => 0,
    Trend::Constant => 1,
    Trend::ConstantTrend => 2,
  };

  Ok((lhs, rhs, gamma_index))
}

/// The ADF regression at a fixed lag order and the lagged level's t-statistic, NaN for an exactly
/// fitting regression; `Err` on too short a series or a singular design.
pub fn fit_adf(y: &[f64], lags: usize, trend: Trend) -> Result<AdfFit, StatsError> {
  let (lhs, rhs, gamma_index) = build_adf_design(y, lags, trend)?;
  let ols_fit = ols(&lhs, &rhs)?;

  let gamma = ols_fit.beta[gamma_index];
  let se = ols_fit.std_err[gamma_index];
  let statistic = if se > 0.0 { gamma / se } else { f64::NAN };

  Ok(AdfFit {
    statistic,
    gamma,
    std_err_gamma: se,
    nobs: ols_fit.nobs,
    residuals: ols_fit.residuals,
    sigma2: ols_fit.sigma2,
  })
}

pub fn aic_from_sse(sse: f64, nobs: usize, k: usize) -> f64 {
  let n = nobs as f64;
  n * (sse / n).ln() + 2.0 * k as f64
}

pub fn bic_from_sse(sse: f64, nobs: usize, k: usize) -> f64 {
  let n = nobs as f64;
  n * (sse / n).ln() + (k as f64) * n.ln()
}

/// The lag `lag_selection` takes from `0..=max_lags`, every candidate fitted on the rows of
/// `max_lags` so that their criteria compare; `Err` on a singular fit or a NaN criterion or `t`.
pub fn choose_lag_for_adf(
  y: &[f64],
  trend: Trend,
  lag_selection: LagSelection,
  max_lags: usize,
) -> Result<usize, StatsError> {
  if let LagSelection::Fixed(p) = lag_selection {
    return Ok(p);
  }
  let (lhs, rhs, gamma_index) = build_adf_design(y, max_lags, trend)?;
  let mut criteria = Vec::with_capacity(max_lags + 1);
  let mut significant = 0;
  for lag in 0..=max_lags {
    let columns = gamma_index + 1 + lag;
    let rows = rhs
      .iter()
      .map(|row| row[..columns].to_vec())
      .collect::<Vec<_>>();
    let fit = ols(&lhs, &rows)?;
    match lag_selection {
      LagSelection::Aic => criteria.push(aic_from_sse(fit.sse, fit.nobs, fit.k)),
      LagSelection::Bic => criteria.push(bic_from_sse(fit.sse, fit.nobs, fit.k)),
      _ => {
        if lag > 0 && t_is_significant(fit.beta[columns - 1], fit.std_err[columns - 1])? {
          significant = lag;
        }
      }
    }
  }
  if lag_selection == LagSelection::TStat {
    return Ok(significant);
  }
  least_criterion(&criteria)
}

/// Whether `|beta / se|` reaches the 10 % two-sided normal point; `Err` on a NaN `t`.
fn t_is_significant(beta: f64, se: f64) -> Result<bool, StatsError> {
  let t = (beta / se).abs();
  if t.is_nan() {
    return Err(StatsError::Numerical {
      stage: "lag selection",
    });
  }
  Ok(se > 0.0 && t >= 1.644_853_626_951_472_2)
}

/// The index of the smallest criterion, ties to the smaller lag; a NaN leaves the candidates with
/// no order.
fn least_criterion(criteria: &[f64]) -> Result<usize, StatsError> {
  if criteria.iter().any(|c| c.is_nan()) {
    return Err(StatsError::Numerical {
      stage: "lag selection",
    });
  }
  let mut best = 0;
  for (lag, &criterion) in criteria.iter().enumerate() {
    if criterion < criteria[best] {
      best = lag;
    }
  }
  Ok(best)
}

/// Bartlett-weighted long-run variance of `u`, `>= 0` in exact arithmetic; `Err` on an empty
/// series, `DegenerateData` when it is not positive and `Numerical` when it is not finite.
pub fn newey_west_long_run_variance(u: &[f64], lags: usize) -> Result<f64, StatsError> {
  if u.is_empty() {
    return Err(StatsError::InsufficientData { needed: 1, got: 0 });
  }
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

  if !lr_var.is_finite() {
    return Err(StatsError::Numerical {
      stage: "long-run variance",
    });
  }
  if lr_var <= 0.0 {
    return Err(StatsError::DegenerateData("zero long-run variance"));
  }
  Ok(lr_var)
}

/// An exact fit leaves a residual mean square near `ε² ≈ 5e-32` of the series' own variance; one
/// below this fraction is refused, as is a series whose variance an explosive tail dominates.
const EXACT_FIT: f64 = 1e-24;

/// `y` less its mean: a fit with a constant is the same, and keeps every digit of a variation that
/// is small against the level, as arch's `PhillipsPerron` does; scale `y` first, the sum overflows.
pub fn centred(y: &[f64]) -> Vec<f64> {
  let mean = y.iter().sum::<f64>() / y.len() as f64;
  y.iter().map(|v| v - mean).collect()
}

/// `y` times the power of two nearest the inverse of its largest magnitude, an exact scaling that
/// leaves a scale-free statistic as it is and puts it on the scale of the trend terms beside it.
pub fn unit_scaled(mut y: Vec<f64>) -> Vec<f64> {
  let spread = f64::max_or_nan_fold(y.iter().map(|v| v.abs()));
  let unit = 2f64.powi(-(spread.log2().round() as i32).clamp(-1022, 1022));
  for v in &mut y {
    *v *= unit;
  }
  y
}

/// `DegenerateData("zero residual variance")` when the mean square of `residuals`, of a fit to `y`,
/// is below `EXACT_FIT` of `y`'s variance; both are scaled by `y`'s largest deviation, so none overflows.
pub fn reject_exact_fit(y: &[f64], residuals: &[f64]) -> Result<(), StatsError> {
  let mean = y.iter().sum::<f64>() / y.len() as f64;
  let spread = f64::max_or_nan_fold(y.iter().map(|v| (v - mean).abs()));
  let mean_square = |values: &mut dyn Iterator<Item = f64>, n: usize| {
    values.map(|v| (v / spread) * (v / spread)).sum::<f64>() / n as f64
  };
  let own = mean_square(&mut y.iter().map(|v| v - mean), y.len());
  let left = mean_square(&mut residuals.iter().copied(), residuals.len());
  if left <= EXACT_FIT * own {
    return Err(StatsError::DegenerateData("zero residual variance"));
  }
  Ok(())
}

pub fn regress_on_deterministics(y: &[f64], include_trend: bool) -> Result<OlsResult, StatsError> {
  validate(y, if include_trend { 3 } else { 2 })?;
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

pub fn fit_ar(y: &[f64], lags: usize) -> Result<(Vec<f64>, Vec<f64>, f64), StatsError> {
  assert!(lags > 0, "AR lag order must be at least 1");
  validate(y, lags + 3)?;

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

  let fit = ols(&lhs, &rhs)?;
  let sigma = fit.sigma2.sqrt();
  Ok((fit.beta, fit.residuals, sigma))
}

#[cfg(test)]
mod tests {
  use super::least_criterion;
  use super::newey_west_long_run_variance;
  use super::t_is_significant;
  use crate::StatsError;

  /// Ties go to the smaller lag, an exact fit's `−∞` wins, and a NaN leaves no order to choose by.
  #[test]
  fn the_least_criterion_takes_the_smaller_lag_and_refuses_nan() {
    assert_eq!(least_criterion(&[3.0, 1.0, 1.0, 2.0]), Ok(1));
    assert_eq!(
      least_criterion(&[f64::INFINITY, 2.0, f64::NEG_INFINITY, f64::NEG_INFINITY]),
      Ok(2)
    );
    assert_eq!(least_criterion(&[f64::INFINITY, f64::INFINITY]), Ok(0));
    assert_eq!(
      least_criterion(&[1.0, f64::NAN, 0.5]),
      Err(StatsError::Numerical {
        stage: "lag selection"
      })
    );
  }

  /// A NaN `t` has no side of the threshold, so the chooser refuses it as it does a NaN criterion.
  #[test]
  fn a_nan_t_statistic_is_refused_where_a_finite_one_is_compared() {
    let refused = Err(StatsError::Numerical {
      stage: "lag selection",
    });
    assert_eq!(t_is_significant(2.0, 1.0), Ok(true));
    assert_eq!(t_is_significant(-1.0, 1.0), Ok(false));
    assert_eq!(t_is_significant(1.0, f64::NAN), refused);
    assert_eq!(t_is_significant(f64::NAN, 1.0), refused);
    assert_eq!(t_is_significant(0.0, 0.0), refused);
  }

  /// The residuals of an exact fit are all zero, so they carry no long-run variance.
  #[test]
  fn zero_residuals_have_no_long_run_variance() {
    assert_eq!(
      newey_west_long_run_variance(&[0.0; 30], 4).unwrap_err(),
      StatsError::DegenerateData("zero long-run variance")
    );
  }
}
