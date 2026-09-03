//! Hill (1975) estimator of the tail index from the `k` largest positive
//! observations.
//!
//! With the positive observations ordered $X_{(1)} \ge X_{(2)} \ge \dots$,
//!
//! $$
//! \hat\xi_k = \frac1k\sum_{i=1}^{k}\log X_{(i)} - \log X_{(k+1)},\qquad
//! \hat\alpha_k = 1/\hat\xi_k,
//! $$
//!
//! estimates the extreme-value index $\xi > 0$ of a Pareto-type tail
//! $P(X > x) \sim x^{-\alpha}$; under second-order regular variation
//! $\sqrt k(\hat\xi_k - \xi) \to \mathcal N(0, \xi^2)$, so the reported
//! standard error is $\hat\xi_k/\sqrt k$.
//!
//! Reference: Hill, "A Simple General Approach to Inference About the Tail
//! of a Distribution", Annals of Statistics, 3(5), 1163-1174 (1975).
//! DOI: 10.1214/aos/1176343247

use ndarray::ArrayView1;

use crate::traits::FloatExt;

/// Result of the Hill estimator.
#[derive(Debug, Clone)]
pub struct HillResult {
  /// Tail index $\hat\xi_k$.
  pub xi: f64,
  /// Tail exponent $\hat\alpha_k = 1/\hat\xi_k$.
  pub alpha: f64,
  /// Asymptotic standard error $\hat\xi_k/\sqrt k$.
  pub std_error: f64,
  /// Number of upper order statistics used.
  pub k: usize,
  /// The $(k+1)$-th largest observation, the implicit threshold.
  pub threshold: f64,
  /// Number of positive observations the order statistics were taken from.
  pub nobs: usize,
}

/// Hill estimator on the `k` largest of the positive entries of `data`
/// (pass losses or absolute returns; non-positive values are ignored).
///
/// Returns a NaN tail index when infinite observations reach the threshold
/// order statistic.
///
/// # Panics
///
/// If `k` is zero or there are fewer than `k + 1` positive observations.
pub fn hill_estimator<T: FloatExt>(data: ArrayView1<T>, k: usize) -> HillResult {
  assert!(k >= 1, "k must be at least 1");
  let mut positive: Vec<f64> = data
    .iter()
    .map(|x| x.to_f64().unwrap_or(f64::NAN))
    .filter(|x| *x > 0.0)
    .collect();
  assert!(
    positive.len() > k,
    "need more than k = {k} positive observations, got {}",
    positive.len()
  );
  positive.sort_by(|a, b| b.partial_cmp(a).unwrap());
  let threshold = positive[k];
  let log_threshold = threshold.ln();
  let xi = positive[..k]
    .iter()
    .map(|x| x.ln() - log_threshold)
    .sum::<f64>()
    / k as f64;
  HillResult {
    xi,
    alpha: 1.0 / xi,
    std_error: xi / (k as f64).sqrt(),
    k,
    threshold,
    nobs: positive.len(),
  }
}
