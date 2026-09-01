//! Johansen cointegration: the trace and maximum-eigenvalue rank tests and
//! the maximum-likelihood VECM estimate at a chosen rank.
//!
//! The vector error-correction model with an unrestricted constant,
//!
//! $$
//! \Delta y_t = \mu + \Pi y_{t-1} + \sum_{i=1}^{p-1}\Gamma_i\,\Delta y_{t-i} + \varepsilon_t,
//! \qquad \Pi = \alpha\beta',
//! $$
//!
//! is concentrated on the residuals $R_{0t}$, $R_{1t}$ of $\Delta y_t$ and
//! $y_{t-1}$ regressed on the constant and the lagged differences, with
//! product moments $S_{ij} = T^{-1}\sum_t R_{it}R_{jt}'$. The ordered
//! eigenvalues $\hat\lambda_1 \ge \dots \ge \hat\lambda_K$ of
//! $|\lambda S_{11} - S_{10}S_{00}^{-1}S_{01}| = 0$ give the two rank
//! statistics for $H(r)$ against $H(K)$ and $H(r+1)$,
//!
//! $$
//! \lambda_{\mathrm{trace}}(r) = -T\sum_{i=r+1}^{K}\log(1-\hat\lambda_i),\qquad
//! \lambda_{\max}(r) = -T\log(1-\hat\lambda_{r+1}),
//! $$
//!
//! and the ML estimates $\hat\beta = (\hat v_1, \ldots, \hat v_r)$ with
//! $\hat V' S_{11}\hat V = I$ and $\hat\alpha = S_{01}\hat\beta$; the
//! short-run matrices and the constant follow by regressing
//! $\Delta y_t - \hat\Pi y_{t-1}$ on the remaining regressors.
//!
//! Critical values are the 5% asymptotic quantiles of MacKinnon, Haug and
//! Michelis for the unrestricted-constant case — the tables `statsmodels`
//! and EViews ship — indexed by $K - r$ and tabulated up to $K - r = 12$.
//!
//! Reference: Johansen, "Statistical Analysis of Cointegration Vectors",
//! Journal of Economic Dynamics and Control, 12(2-3), 231-254 (1988).
//! DOI: 10.1016/0165-1889(88)90041-3
//!
//! Reference: Johansen, "Estimation and Hypothesis Testing of Cointegration
//! Vectors in Gaussian Vector Autoregressive Models", Econometrica, 59(6),
//! 1551-1580 (1991). DOI: 10.2307/2938278
//!
//! Reference: MacKinnon, Haug, Michelis, "Numerical Distribution Functions
//! of Likelihood Ratio Tests for Cointegration", Journal of Applied
//! Econometrics, 14(5), 563-577 (1999).
//! DOI: 10.1002/(SICI)1099-1255(199909/10)14:5<563::AID-JAE530>3.0.CO;2-R

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray::s;

use crate::linalg::inverse;
use crate::linalg::lstsq;
use crate::linalg::spd_cholesky_lower;
use crate::linalg::symmetric_eigen;

/// 5% critical values of the trace statistic for $K - r = 1, \ldots, 12$
/// (MacKinnon–Haug–Michelis 1999, unrestricted constant).
const TRACE_CRITICAL_5PCT: [f64; 12] = [
  3.8415, 15.4943, 29.7961, 47.8545, 69.8189, 95.7542, 125.6185, 159.5290, 197.3772, 239.2468,
  285.1402, 334.9795,
];

/// 5% critical values of the maximum-eigenvalue statistic for
/// $K - r = 1, \ldots, 12$ (MacKinnon–Haug–Michelis 1999, unrestricted
/// constant).
const MAX_EIG_CRITICAL_5PCT: [f64; 12] = [
  3.8415, 14.2639, 21.1314, 27.5858, 33.8777, 40.0763, 46.2299, 52.3622, 58.4332, 64.5040, 70.5392,
  76.5734,
];

/// Result of the Johansen trace and maximum-eigenvalue rank tests.
#[derive(Debug, Clone)]
pub struct JohansenResult {
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
pub struct Vecm {
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

/// The concentrated regression both entry points share: $\Delta y_t$,
/// $y_{t-1}$, the constant-plus-lagged-differences block, $S_{01}$ and the
/// ordered eigen-solution.
struct Concentrated {
  z0: Array2<f64>,
  z1: Array2<f64>,
  z2: Array2<f64>,
  s01: Array2<f64>,
  eigenvalues: Array1<f64>,
  eigenvectors: Array2<f64>,
  n_eff: usize,
}

fn concentrate(series: ArrayView2<f64>, lags: usize) -> Concentrated {
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
  let n = n_eff as f64;
  let s00 = r0.t().dot(&r0) / n;
  let s11 = r1.t().dot(&r1) / n;
  let s01 = r0.t().dot(&r1) / n;
  let s00_inv = inverse(&s00).expect("S00 inverse failed");
  let m = s01.t().dot(&s00_inv).dot(&s01);
  let (eigenvalues, eigenvectors) = ordered_eigen(&m, &s11);
  Concentrated {
    z0,
    z1,
    z2,
    s01,
    eigenvalues,
    eigenvectors,
    n_eff,
  }
}

/// Eigenpairs of $|\lambda S_{11} - M| = 0$ through the Cholesky-whitened
/// symmetric problem $L^{-1} M L^{-\top} w = \lambda w$ with $S_{11} = LL'$:
/// eigenvalues descending and clamped to $[0, 1)$, eigenvectors
/// $V = L^{-\top} W$ so that $V' S_{11} V = I$.
fn ordered_eigen(m: &Array2<f64>, s11: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
  let k = s11.nrows();
  let l = spd_cholesky_lower(s11).expect("S11 is not positive definite");
  let l_inv = inverse(&l).expect("S11 Cholesky factor inverse failed");
  let whitened = l_inv.dot(m).dot(&l_inv.t());
  let symmetric = (&whitened + &whitened.t()) / 2.0;
  let (values, vectors) = symmetric_eigen(&symmetric).expect("Johansen eig failed");
  let mut order: Vec<usize> = (0..k).collect();
  order.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap());
  let raw = l_inv.t().dot(&vectors);
  let mut eigenvalues = Array1::<f64>::zeros(k);
  let mut eigenvectors = Array2::<f64>::zeros((k, k));
  for (slot, &idx) in order.iter().enumerate() {
    eigenvalues[slot] = values[idx].clamp(0.0, 1.0 - 1e-12);
    eigenvectors.column_mut(slot).assign(&raw.column(idx));
  }
  (eigenvalues, eigenvectors)
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
    let beta = lstsq(x, &target);
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

/// Johansen trace and maximum-eigenvalue tests for the cointegrating rank
/// of a `(t, k)` series. `lags` is the VAR order (use `1` if unsure).
pub fn johansen_test(series: ArrayView2<f64>, lags: usize) -> JohansenResult {
  let c = concentrate(series, lags);
  let k = c.eigenvalues.len();
  let t = c.n_eff as f64;
  let log_terms: Vec<f64> = c.eigenvalues.iter().map(|&l| (1.0 - l).ln()).collect();
  let trace_statistics = Array1::from_iter((0..k).map(|r| -t * log_terms[r..].iter().sum::<f64>()));
  let max_eig_statistics = Array1::from_iter((0..k).map(|r| -t * log_terms[r]));
  let trace_critical_5pct = critical_values(&TRACE_CRITICAL_5PCT, k);
  let max_eig_critical_5pct = critical_values(&MAX_EIG_CRITICAL_5PCT, k);
  let rank_trace = sequential_rank(&trace_statistics, &trace_critical_5pct);
  let rank_max_eig = sequential_rank(&max_eig_statistics, &max_eig_critical_5pct);
  JohansenResult {
    eigenvalues: c.eigenvalues,
    eigenvectors: c.eigenvectors,
    trace_statistics,
    max_eig_statistics,
    trace_critical_5pct,
    max_eig_critical_5pct,
    rank_trace,
    rank_max_eig,
    nobs: c.n_eff,
  }
}

/// Maximum-likelihood VECM estimate of a `(t, k)` series at cointegrating
/// rank `rank` (`0..=k`; `k` is the unrestricted VAR) with VAR order `lags`.
pub fn vecm_fit(series: ArrayView2<f64>, lags: usize, rank: usize) -> Vecm {
  let c = concentrate(series, lags);
  let k = c.eigenvalues.len();
  assert!(rank <= k, "rank must lie in 0..={k}, got {rank}");
  let beta = c.eigenvectors.slice(s![.., ..rank]).to_owned();
  let alpha = c.s01.dot(&beta);
  let pi = alpha.dot(&beta.t());
  let target = &c.z0 - &c.z1.dot(&pi.t());
  let (n_eff, n_cols) = c.z2.dim();
  let mut coef = Array2::<f64>::zeros((n_cols, k));
  for j in 0..k {
    let column = target.column(j).to_owned();
    coef.column_mut(j).assign(&lstsq(&c.z2, &column));
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
  Vecm {
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
  }
}

#[cfg(test)]
mod tests;
