use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use stochastic_rs_distributions::linalg::inverse;
use stochastic_rs_distributions::linalg::solve;

use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::input::as_slice_or_copy;
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
  pub condition: f64,
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

/// OLS of `y` on the rows of `x`; `Err(Singular)` on a design singular or past `MAX_CONDITION`, and
/// a NaN standard error where an overflow or rounding leaves its variance NaN or negative.
pub fn ols(y: &[f64], x: &[Vec<f64>]) -> Result<OlsResult, StatsError> {
  ols_absorbing(y, x, 0)
}

/// Past this condition number of the equilibrated normal equations the design is numerically
/// singular: a coefficient errs by up to `κ ε`, and RESET's raw-power `F` by `2e-6` just below it.
const MAX_CONDITION: f64 = 1e13;

/// How far past `κ ε`, the solve's rounding of a coefficient, residuals must stay to be resolved.
const ROUNDING_MARGIN: f64 = 10.0;

/// `‖A‖₁ ‖A⁻¹‖₁` of `A = D X'X D` with `D = diag((X'X)_jj^{-1/2})`, which no column's scale moves.
fn equilibrated_condition(xtx: &Array2<f64>, xtx_inv: &Array2<f64>) -> f64 {
  let k = xtx.nrows();
  let s = xtx.diag().mapv(f64::sqrt);
  let a = Array2::from_shape_fn((k, k), |(i, j)| xtx[[i, j]] / (s[i] * s[j]));
  let a_inv = Array2::from_shape_fn((k, k), |(i, j)| xtx_inv[[i, j]] * (s[i] * s[j]));
  let norm = |m: &Array2<f64>| {
    f64::max_or_nan_fold(
      m.columns()
        .into_iter()
        .map(|c| c.iter().map(|v| v.abs()).sum::<f64>()),
    )
  };
  norm(&a) * norm(&a_inv)
}

/// [`ols`] of `y` and `x` that `absorbed` terms were partialled out of: its fit is the full
/// regression's, whose degrees of freedom and `k` count them.
fn ols_absorbing(y: &[f64], x: &[Vec<f64>], absorbed: usize) -> Result<OlsResult, StatsError> {
  assert!(!y.is_empty(), "OLS requires non-empty response");
  assert_eq!(y.len(), x.len(), "OLS y/x row mismatch");
  let n = y.len();
  let k = x[0].len();
  assert!(k > 0, "OLS requires at least one regressor");
  assert!(
    x.iter().all(|row| row.len() == k),
    "OLS design matrix must be rectangular"
  );
  assert!(n > k + absorbed, "OLS requires nobs > number of regressors");

  let x_mat = Array2::from_shape_fn((n, k), |(i, j)| x[i][j]);
  let y_vec = Array1::from_vec(y.to_vec());

  let xtx = x_mat.t().dot(&x_mat);
  let xty = x_mat.t().dot(&y_vec);

  let beta_arr = solve(xtx.view(), xty.view()).ok_or(StatsError::Singular("OLS design"))?;
  let fitted = x_mat.dot(&beta_arr);
  let residuals_arr = &y_vec - &fitted;

  let residuals: Vec<f64> = residuals_arr.to_vec();
  let sse = residuals.iter().map(|u| u * u).sum::<f64>();
  let dof = (n - k - absorbed) as f64;
  let sigma2 = sse / dof;

  let xtx_inv = inverse(xtx.view()).ok_or(StatsError::Singular("OLS design"))?;
  let condition = equilibrated_condition(&xtx, &xtx_inv);
  if condition.is_nan() || condition > MAX_CONDITION {
    return Err(StatsError::Singular("OLS design"));
  }
  let cov = &xtx_inv * sigma2;
  let std_err = (0..k).map(|i| cov[[i, i]].sqrt()).collect();

  Ok(OlsResult {
    beta: beta_arr.to_vec(),
    std_err,
    residuals,
    sse,
    sigma2,
    nobs: n,
    k: k + absorbed,
    condition,
  })
}

/// The regressand, the rows (the lagged level first) and how many terms are partialled out of both.
type AdfDesign = (Vec<f64>, Vec<Vec<f64>>, usize);

/// `v` less its fit on a constant and `centred_trend`, twice: the second pass takes off rounding of
/// the first, which a level far above the variation makes as large as the variation itself.
fn partial_out_terms(v: &mut [f64], centred_trend: &[f64]) {
  let tt = centred_trend.iter().map(|t| t * t).sum::<f64>();
  for _ in 0..2 {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    v.iter_mut().for_each(|x| *x -= mean);
    if !centred_trend.is_empty() {
      let slope = v.iter().zip(centred_trend).map(|(x, t)| x * t).sum::<f64>() / tt;
      v.iter_mut()
        .zip(centred_trend)
        .for_each(|(x, t)| *x -= slope * t);
    }
  }
}

/// Frisch–Waugh–Lovell (Lovell 1963, DOI: 10.1080/01621459.1963.10480682): partialling the terms
/// out keeps the fit, and keeps a level they nearly fit out of normal equations that square κ.
fn build_adf_design(y: &[f64], lags: usize, trend: Trend) -> Result<AdfDesign, StatsError> {
  validate(y, 3 + lags)?;
  let dy = difference(y);
  let n_dy = dy.len();
  let terms = match trend {
    Trend::None => 0,
    Trend::Constant => 1,
    Trend::ConstantTrend => 2,
  };
  // Row t carries the trend value t + 1, whose mean over the rows is exactly a multiple of ½.
  let centred_trend = match trend {
    Trend::ConstantTrend => {
      let centre = (lags + 1 + n_dy) as f64 / 2.0;
      (lags..n_dy).map(|t| (t + 1) as f64 - centre).collect()
    }
    _ => Vec::new(),
  };
  let partialled = |column: &[f64]| {
    let mut column = column.to_vec();
    if trend != Trend::None {
      partial_out_terms(&mut column, &centred_trend);
    }
    column
  };

  let lhs = partialled(&dy[lags..]);
  let level = partialled(&y[lags..n_dy]);
  // A lagged level that is a constant, or a line under a trend, leaves the regression no design.
  if trend != Trend::None {
    reject_exact_fit(y, &level, 1.0)?;
  }
  let differences = (1..=lags)
    .map(|i| partialled(&dy[lags - i..n_dy - i]))
    .collect::<Vec<_>>();
  let rhs = (0..n_dy - lags)
    .map(|r| {
      let mut row = Vec::with_capacity(1 + lags);
      row.push(level[r]);
      row.extend(differences.iter().map(|column| column[r]));
      row
    })
    .collect();

  Ok((lhs, rhs, terms))
}

/// The ADF regression at a fixed lag order and the lagged level's t-statistic; `Err` on too short a
/// series, a singular design or a standard error of no use, or as `reject_exact_fit` of `source`.
pub fn fit_adf(y: &[f64], lags: usize, trend: Trend, source: &[f64]) -> Result<AdfFit, StatsError> {
  let (lhs, rhs, terms) = build_adf_design(y, lags, trend)?;
  let ols_fit = ols_absorbing(&lhs, &rhs, terms)?;
  reject_exact_fit(source, &ols_fit.residuals, ols_fit.condition)?;

  let gamma = ols_fit.beta[0];
  let se = ols_fit.std_err[0];
  if !(se.is_finite() && se > 0.0) {
    return Err(StatsError::Numerical {
      stage: "standard error",
    });
  }
  let statistic = gamma / se;

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
/// `max_lags`; `Err` on a singular fit, a NaN criterion or `t`, or a `t` an exact fit leaves.
pub fn choose_lag_for_adf(
  y: &[f64],
  trend: Trend,
  lag_selection: LagSelection,
  max_lags: usize,
) -> Result<usize, StatsError> {
  if let LagSelection::Fixed(p) = lag_selection {
    return Ok(p);
  }
  let (lhs, rhs, terms) = build_adf_design(y, max_lags, trend)?;
  let mut criteria = Vec::with_capacity(max_lags + 1);
  let mut significant = 0;
  for lag in 0..=max_lags {
    let columns = 1 + lag;
    let rows = rhs
      .iter()
      .map(|row| row[..columns].to_vec())
      .collect::<Vec<_>>();
    let fit = ols_absorbing(&lhs, &rows, terms)?;
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

/// Whether `|beta / se|` reaches the 10 % two-sided normal point; `Err` on a NaN `t`, and on a zero
/// `se`, which only a candidate that fits exactly leaves.
fn t_is_significant(beta: f64, se: f64) -> Result<bool, StatsError> {
  if se == 0.0 {
    return Err(StatsError::DegenerateData("zero residual variance"));
  }
  let t = (beta / se).abs();
  if t.is_nan() {
    return Err(StatsError::Numerical {
      stage: "lag selection",
    });
  }
  Ok(t >= 1.644_853_626_951_472_2)
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

/// A series whose range is within this many `ε` of its largest magnitude is a constant's rounding:
/// neighbouring doubles, as a constant through two or three rounded operations leaves.
const ROUNDING_RANGE: f64 = 4.0;

/// `DegenerateData("constant series")` when `max − min <= ROUNDING_RANGE ε max|y|`: the variance
/// of such a series is rounding, which no relative rule can tell from a signal.
pub fn reject_rounded_constant(y: &[f64]) -> Result<(), StatsError> {
  let (low, high) = y
    .iter()
    .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), &v| {
      (low.min(v), high.max(v))
    });
  if high - low <= ROUNDING_RANGE * f64::EPSILON * low.abs().max(high.abs()) {
    return Err(StatsError::DegenerateData("constant series"));
  }
  Ok(())
}

/// `y` less its mean: a fit with a constant is the same, and keeps every digit of a variation that
/// is small against the level, as arch's `PhillipsPerron` does; scale `y` first, the sum overflows.
pub fn centred(y: &[f64]) -> Vec<f64> {
  let mean = y.iter().sum::<f64>() / y.len() as f64;
  y.iter().map(|v| v - mean).collect()
}

/// The power of two nearest the inverse of the largest magnitude in `y`.
pub fn unit_scale(y: &[f64]) -> f64 {
  let spread = f64::max_or_nan_fold(y.iter().map(|v| v.abs()));
  2f64.powi(-(spread.log2().round() as i32).clamp(-1022, 1022))
}

/// `y` times its [`unit_scale`], and that scale: exact, it leaves a scale-free statistic as it is
/// and puts it on the scale of the trend terms beside it.
pub fn unit_scaled(mut y: Vec<f64>) -> (Vec<f64>, f64) {
  let unit = unit_scale(&y);
  for v in &mut y {
    *v *= unit;
  }
  (y, unit)
}

/// [`unit_scaled`] `y` with `min_n` finite points or more and more than a constant's rounding; the
/// length comes first, as a series too short to compare is vacuously constant.
pub fn prepared_with_unit(y: ArrayView1<f64>, min_n: usize) -> Result<(Vec<f64>, f64), StatsError> {
  let y = as_slice_or_copy(y);
  validate(&y, min_n)?;
  reject_rounded_constant(&y)?;
  Ok(unit_scaled(y.into_owned()))
}

/// [`prepared_with_unit`]'s series.
pub fn prepared(y: ArrayView1<f64>, min_n: usize) -> Result<Vec<f64>, StatsError> {
  Ok(prepared_with_unit(y, min_n)?.0)
}

/// `DegenerateData("zero residual variance")` when `residuals` keep at most `EXACT_FIT` of `y`'s
/// variance, or `(ROUNDING_MARGIN κ ε)²` at the solve's `condition` κ; `Err` as
/// [`reject_rounded_constant`] of `y`.
pub fn reject_exact_fit(y: &[f64], residuals: &[f64], condition: f64) -> Result<(), StatsError> {
  reject_rounded_constant(y)?;
  let mean = y.iter().sum::<f64>() / y.len() as f64;
  let spread = f64::max_or_nan_fold(y.iter().map(|v| (v - mean).abs()));
  // Over the largest deviation, so that no square overflows.
  let square = |v: f64| (v / spread) * (v / spread);
  // Both sums run in index order, interleaved so that neither waits on the other.
  let (mut own, mut left) = (-0.0, -0.0);
  for (v, u) in y.iter().zip(residuals) {
    own += square(v - mean);
    left += square(*u);
  }
  for v in y.iter().skip(residuals.len()) {
    own += square(v - mean);
  }
  for u in residuals.iter().skip(y.len()) {
    left += square(*u);
  }
  let (own, left) = (own / y.len() as f64, left / residuals.len() as f64);
  let rounding = (ROUNDING_MARGIN * condition * f64::EPSILON).powi(2);
  if left <= EXACT_FIT.max_or_nan(rounding) * own {
    return Err(StatsError::DegenerateData("zero residual variance"));
  }
  Ok(())
}

/// `F` of a fit nested in one with `q` more regressors and `dof` residual degrees of freedom, from
/// both residual sums of squares; `Numerical { stage: "F statistic" }` when it is not finite.
pub fn nested_f(rss_nested: f64, rss_full: f64, q: f64, dof: f64) -> Result<f64, StatsError> {
  let f = ((rss_nested - rss_full) / q) / (rss_full / dof);
  if !f.is_finite() {
    return Err(StatsError::Numerical {
      stage: "F statistic",
    });
  }
  // The nested fit cannot fit better, so a negative `F` is the subtraction's rounding.
  Ok(f.max(0.0))
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

pub fn fit_ar(y: &[f64], lags: usize) -> Result<OlsResult, StatsError> {
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

  ols(&lhs, &rhs)
}

#[cfg(test)]
mod tests {
  use super::least_criterion;
  use super::newey_west_long_run_variance;
  use super::reject_exact_fit;
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
  }

  /// A zero standard error is an exactly fitting candidate, whatever its coefficient.
  #[test]
  fn a_zero_standard_error_is_an_exact_fit() {
    let exact = Err(StatsError::DegenerateData("zero residual variance"));
    assert_eq!(t_is_significant(1.0, 0.0), exact);
    assert_eq!(t_is_significant(0.0, 0.0), exact);
  }

  /// Residuals at `1e-20` of the variance pass after a solve of condition `1e4`, not `1e5`, whose
  /// rounding `(10 κ ε)²` is `4.9e-20`.
  #[test]
  fn the_exact_fit_threshold_rises_with_the_solve_s_rounding() {
    let y = (0..10).map(f64::from).collect::<Vec<_>>();
    let r = 8.25e-20f64.sqrt();
    let residuals = (0..10)
      .map(|i| if i % 2 == 0 { r } else { -r })
      .collect::<Vec<_>>();
    let exact = Err(StatsError::DegenerateData("zero residual variance"));
    assert_eq!(reject_exact_fit(&y, &residuals, 1.0), Ok(()));
    assert_eq!(reject_exact_fit(&y, &residuals, 1e4), Ok(()));
    assert_eq!(reject_exact_fit(&y, &residuals, 1e5), exact);
  }

  /// Neighbouring doubles about `0.7` are a constant's rounding whatever the residuals; the same
  /// steps four ulps apart are a series.
  #[test]
  fn a_series_of_neighbouring_doubles_is_a_constant_series() {
    let ulp = f64::EPSILON / 2.0;
    let steps = [0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, -1.0, 0.0, 1.0];
    let residuals = steps.map(|j| 0.4 * j * ulp + 0.1 * ulp);
    assert_eq!(
      reject_exact_fit(&steps.map(|j| 0.7 + j * ulp), &residuals, 1.0),
      Err(StatsError::DegenerateData("constant series"))
    );
    assert_eq!(
      reject_exact_fit(&steps.map(|j| 0.7 + 4.0 * j * ulp), &residuals, 1.0),
      Ok(())
    );
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
