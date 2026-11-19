//! Johansen cointegration: the trace and maximum-eigenvalue rank tests, and the ML VECM at a rank.
//!
//! $$
//! \Delta y_t = \mu + \Pi y_{t-1} + \sum_{i=1}^{p-1}\Gamma_i\,\Delta y_{t-i} + \varepsilon_t,
//! \qquad \Pi = \alpha\beta',
//! $$
//!
//! $$
//! R_{0t} = \Delta y_t - \hat E[\Delta y_t \mid 1, \Delta y_{t-1}, \ldots, \Delta y_{t-p+1}],\qquad
//! R_{1t} = y_{t-1} - \hat E[y_{t-1} \mid 1, \Delta y_{t-1}, \ldots, \Delta y_{t-p+1}],
//! $$
//!
//! $$
//! S_{ij} = T^{-1}\sum_t R_{it}R_{jt}',\qquad
//! |\lambda S_{11} - S_{10}S_{00}^{-1}S_{01}| = 0,\qquad
//! \hat\lambda_1 \ge \dots \ge \hat\lambda_K,
//! $$
//!
//! $$
//! \lambda_{\mathrm{trace}}(r) = -T\sum_{i=r+1}^{K}\log(1-\hat\lambda_i),\qquad
//! \lambda_{\max}(r) = -T\log(1-\hat\lambda_{r+1}),
//! $$
//!
//! $$
//! \hat\beta = (\hat v_1, \ldots, \hat v_r),\qquad \hat V' S_{11}\hat V = I,\qquad
//! \hat\alpha = S_{01}\hat\beta .
//! $$
//!
//! Reference: Johansen (1988), JEDC 12(2-3), 231-254, DOI: 10.1016/0165-1889(88)90041-3
//! Reference: Johansen (1991), Econometrica 59(6), 1551-1580, DOI: 10.2307/2938278
//! Reference: MacKinnon, Haug & Michelis (1999), J. Appl. Econometrics 14(5), 563-577, [DOI][mhm]
//!
//! [mhm]: https://doi.org/10.1002/(SICI)1099-1255(199909/10)14:5<563::AID-JAE530>3.0.CO;2-R

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray::s;
use stochastic_rs_distributions::linalg::inverse;
use stochastic_rs_distributions::linalg::lstsq;
use stochastic_rs_distributions::linalg::spd_cholesky_lower;
use stochastic_rs_distributions::linalg::symmetric_eigen;

use crate::StatsError;
use crate::input::as_slice_or_copy;
use crate::input::reject_constant;
use crate::input::validate_rows;

/// 5% critical values of the trace statistic for $K - r = 1, \ldots, 12$
/// (MacKinnon–Haug–Michelis 1999, unrestricted constant).
const TRACE_CRITICAL_5PCT: [f64; 12] = [
  3.8415, 15.4943, 29.7961, 47.8545, 69.8189, 95.7542, 125.6185, 159.5290, 197.3772, 239.2468,
  285.1402, 334.9795,
];

/// 5% critical values of the maximum-eigenvalue statistic for $K - r = 1, \ldots, 12$
/// (MacKinnon–Haug–Michelis 1999, unrestricted constant).
const MAX_EIG_CRITICAL_5PCT: [f64; 12] = [
  3.8415, 14.2639, 21.1314, 27.5858, 33.8777, 40.0763, 46.2299, 52.3622, 58.4332, 64.5040, 70.5392,
  76.5734,
];

/// Result of the Johansen trace and maximum-eigenvalue rank tests.
#[derive(Debug, Clone)]
pub struct JohansenTest {
  /// Eigenvalues $\hat\lambda_1 \ge \dots \ge \hat\lambda_K$.
  pub eigenvalues: Array1<f64>,
  /// Eigenvectors as columns, column `i` belonging to `eigenvalues[i]`,
  /// normalised $\hat V' S_{11}\hat V = I$.
  pub eigenvectors: Array2<f64>,
  /// Trace statistics for the hypothesised ranks $r = 0, \ldots, K-1$.
  pub trace_statistics: Array1<f64>,
  /// Maximum-eigenvalue statistics for $r = 0, \ldots, K-1$.
  pub max_eig_statistics: Array1<f64>,
  /// 5% critical values of the trace statistic, indexed by $r$; NaN once
  /// $K - r$ exceeds the tabulated 12.
  pub trace_critical_5pct: Array1<f64>,
  /// 5% critical values of the maximum-eigenvalue statistic, indexed by $r$.
  pub max_eig_critical_5pct: Array1<f64>,
  /// Rank chosen by the sequential trace test at 5%: the first $r$ whose
  /// null is not rejected.
  pub rank_trace: usize,
  /// Rank chosen by the sequential maximum-eigenvalue test at 5%.
  pub rank_max_eig: usize,
  /// Effective sample size $T$ (observations minus `lags`).
  pub nobs: usize,
}

/// Maximum-likelihood VECM estimate at a fixed cointegrating rank.
#[derive(Debug, Clone)]
pub struct VecmFit {
  /// Cointegrating rank $r$.
  pub rank: usize,
  /// VAR lag order $p$; the model carries $p - 1$ lagged differences.
  pub lags: usize,
  /// Cointegrating vectors as columns ($K \times r$), Johansen's
  /// normalisation $\hat\beta' S_{11}\hat\beta = I_r$.
  pub beta: Array2<f64>,
  /// Adjustment coefficients ($K \times r$), $\hat\alpha = S_{01}\hat\beta$.
  pub alpha: Array2<f64>,
  /// $\hat\Pi = \hat\alpha\hat\beta'$ ($K \times K$), invariant to how
  /// $\hat\beta$ is normalised.
  pub pi: Array2<f64>,
  /// Short-run matrices $\hat\Gamma_1, \ldots, \hat\Gamma_{p-1}$, each
  /// $K \times K$.
  pub gamma: Vec<Array2<f64>>,
  /// Unrestricted constant $\hat\mu$.
  pub intercept: Array1<f64>,
  /// Residuals $\hat\varepsilon_t$ ($T \times K$).
  pub residuals: Array2<f64>,
  /// ML residual covariance $\hat\Omega = T^{-1}\sum_t\hat\varepsilon_t\hat\varepsilon_t'$.
  pub sigma: Array2<f64>,
  /// Eigenvalues of the rank problem, descending.
  pub eigenvalues: Array1<f64>,
  /// Effective sample size $T$.
  pub nobs: usize,
}

/// The concentrated regression both entry points share: $\Delta y_t$, $y_{t-1}$, the
/// constant-plus-lagged-differences block, $S_{01}$ and the ordered eigen-solution.
struct Concentrated {
  z0: Array2<f64>,
  z1: Array2<f64>,
  z2: Array2<f64>,
  s01: Array2<f64>,
  eigenvalues: Array1<f64>,
  eigenvectors: Array2<f64>,
  n_eff: usize,
}

fn concentrate(series: ArrayView2<f64>, lags: usize) -> Result<Concentrated, StatsError> {
  let (t, k) = series.dim();
  assert!(
    lags >= 1,
    "lags must satisfy `lags >= 1`, got lags = {lags:?}"
  );
  if k < 2 {
    return Err(StatsError::InsufficientData { needed: 2, got: k });
  }
  // S00 needs `k` residual degrees of freedom beyond the `(lags − 1) k + 1` lag regressors.
  let needed = lags.saturating_mul(k.saturating_add(1)).saturating_add(1);
  if t < needed {
    return Err(StatsError::InsufficientData { needed, got: t });
  }
  validate_rows(series)?;
  for column in series.columns() {
    reject_constant(&as_slice_or_copy(column), "constant series")?;
  }
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
  let n = n_eff as f64;
  let s00 = r0.t().dot(&r0) / n;
  let s11 = r1.t().dot(&r1) / n;
  let s01 = r0.t().dot(&r1) / n;
  let s00_inv = inverse(s00.view()).ok_or(StatsError::Singular("S00"))?;
  let m = s01.t().dot(&s00_inv).dot(&s01);
  let (eigenvalues, eigenvectors) = ordered_eigen(&m, &s11)?;
  Ok(Concentrated {
    z0,
    z1,
    z2,
    s01,
    eigenvalues,
    eigenvectors,
    n_eff,
  })
}

/// Eigenpairs of $|\lambda S_{11} - M| = 0$ through $L^{-1} M L^{-\top} w = \lambda w$ with
/// $S_{11} = LL'$: eigenvalues descending in $[0, 1)$, $V = L^{-\top} W$ so that $V' S_{11} V = I$.
fn ordered_eigen(
  m: &Array2<f64>,
  s11: &Array2<f64>,
) -> Result<(Array1<f64>, Array2<f64>), StatsError> {
  let k = s11.nrows();
  let l = spd_cholesky_lower(s11.view()).ok_or(StatsError::Singular("S11"))?;
  let l_inv = inverse(l.view()).ok_or(StatsError::Singular("S11"))?;
  let whitened = l_inv.dot(m).dot(&l_inv.t());
  let symmetric = (&whitened + &whitened.t()) / 2.0;
  // faer answers a non-finite matrix with NaN eigenvalues, not an error.
  let (values, vectors) = symmetric_eigen(symmetric.view())
    .filter(|(values, _)| values.iter().all(|v| v.is_finite()))
    .ok_or(StatsError::Numerical {
      stage: "eigen decomposition",
    })?;
  let mut order: Vec<usize> = (0..k).collect();
  order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
  let raw = l_inv.t().dot(&vectors);
  let mut eigenvalues = Array1::<f64>::zeros(k);
  let mut eigenvectors = Array2::<f64>::zeros((k, k));
  for (slot, &idx) in order.iter().enumerate() {
    eigenvalues[slot] = values[idx].clamp(0.0, 1.0 - 1e-12);
    eigenvectors.column_mut(slot).assign(&raw.column(idx));
  }
  Ok((eigenvalues, eigenvectors))
}

fn residualise(y: &Array2<f64>, x: &Array2<f64>) -> Array2<f64> {
  let (n, p) = y.dim();
  let (_, q) = x.dim();
  let mut residuals = Array2::<f64>::zeros((n, p));
  if q == 0 {
    return y.clone();
  }
  for col in 0..p {
    let beta = lstsq(x.view(), y.column(col));
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

/// Critical values indexed by the hypothesised rank $r$: the table entry
/// for $K - r$, NaN once $K - r$ leaves the tabulated `1..=12`.
fn critical_values(table: &[f64; 12], k: usize) -> Array1<f64> {
  Array1::from_iter((0..k).map(|r| {
    let dim = k - r;
    if dim <= table.len() {
      table[dim - 1]
    } else {
      f64::NAN
    }
  }))
}

/// Johansen's sequential procedure: the first $r$ whose null is not
/// rejected (a NaN critical value never rejects).
fn sequential_rank(statistics: &Array1<f64>, critical: &Array1<f64>) -> usize {
  let mut r = 0;
  while r < statistics.len() && statistics[r] > critical[r] {
    r += 1;
  }
  r
}

/// Johansen trace and max-eigenvalue tests at VAR order `lags`; `Err` below 2 columns, below
/// `lags(k+1)+1` rows, on a non-finite or constant column or no finite fit; panics if `lags == 0`.
pub fn johansen_test(series: ArrayView2<f64>, lags: usize) -> Result<JohansenTest, StatsError> {
  let c = concentrate(series, lags)?;
  let k = c.eigenvalues.len();
  let t = c.n_eff as f64;
  let log_terms: Vec<f64> = c.eigenvalues.iter().map(|&l| (1.0 - l).ln()).collect();
  let trace_statistics = Array1::from_iter((0..k).map(|r| -t * log_terms[r..].iter().sum::<f64>()));
  let max_eig_statistics = Array1::from_iter((0..k).map(|r| -t * log_terms[r]));
  let trace_critical_5pct = critical_values(&TRACE_CRITICAL_5PCT, k);
  let max_eig_critical_5pct = critical_values(&MAX_EIG_CRITICAL_5PCT, k);
  let rank_trace = sequential_rank(&trace_statistics, &trace_critical_5pct);
  let rank_max_eig = sequential_rank(&max_eig_statistics, &max_eig_critical_5pct);
  Ok(JohansenTest {
    eigenvalues: c.eigenvalues,
    eigenvectors: c.eigenvectors,
    trace_statistics,
    max_eig_statistics,
    trace_critical_5pct,
    max_eig_critical_5pct,
    rank_trace,
    rank_max_eig,
    nobs: c.n_eff,
  })
}

/// ML VECM of a `(t, k)` series at cointegrating rank `rank` (`k`: the unrestricted VAR); `Err` as
/// [`johansen_test`], and panics unless `lags >= 1` and `rank <= k`.
pub fn fit_vecm(series: ArrayView2<f64>, lags: usize, rank: usize) -> Result<VecmFit, StatsError> {
  let k = series.ncols();
  assert!(
    rank <= k,
    "rank must satisfy `rank <= k`, got rank = {rank:?}, k = {k:?}"
  );
  let c = concentrate(series, lags)?;
  let beta = c.eigenvectors.slice(s![.., ..rank]).to_owned();
  let alpha = c.s01.dot(&beta);
  let pi = alpha.dot(&beta.t());
  let target = &c.z0 - &c.z1.dot(&pi.t());
  let (n_eff, n_cols) = c.z2.dim();
  let mut coef = Array2::<f64>::zeros((n_cols, k));
  for j in 0..k {
    coef
      .column_mut(j)
      .assign(&lstsq(c.z2.view(), target.column(j)));
  }
  let residuals = &target - &c.z2.dot(&coef);
  let sigma = residuals.t().dot(&residuals) / n_eff as f64;
  let intercept = coef.row(0).to_owned();
  let gamma = (1..lags)
    .map(|l| {
      coef
        .slice(s![1 + (l - 1) * k..1 + l * k, ..])
        .t()
        .to_owned()
    })
    .collect();
  Ok(VecmFit {
    rank,
    lags,
    beta,
    alpha,
    pi,
    gamma,
    intercept,
    residuals,
    sigma,
    eigenvalues: c.eigenvalues,
    nobs: n_eff,
  })
}

#[cfg(test)]
mod tests;
