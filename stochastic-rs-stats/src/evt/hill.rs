//! Hill (1975) estimator of the tail index from the `k` largest positive observations.
//!
//! $$
//! \hat\xi_k = \frac1k\sum_{i=1}^{k}\log X_{(i)} - \log X_{(k+1)},\qquad
//! \hat\alpha_k = 1/\hat\xi_k,\qquad X_{(1)} \ge X_{(2)} \ge \dots
//! $$
//!
//! $$
//! P(X > x) \sim x^{-\alpha},\qquad \sqrt k\,(\hat\xi_k - \xi) \to \mathcal N(0, \xi^2)
//! $$
//!
//! The standard error is `ξ̂_k / √k`, the limit's, which needs second-order regular variation.
//!
//! Reference: Hill (1975), Annals of Statistics 3(5), 1163-1174, DOI: 10.1214/aos/1176343247

use ndarray::ArrayView1;

use crate::StatsError;
use crate::input::validate;
use crate::traits::FloatExt;

/// Result of the Hill estimator.
#[derive(Debug, Clone)]
pub struct HillFit {
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

/// Hill estimator on the `k` largest positive entries of `data`, `ξ̂ = 0` if they tie; `Err` on a
/// non-finite entry or at most `k` positive ones (others are ignored); panics unless `k >= 1`.
pub fn fit_hill<T: FloatExt>(data: ArrayView1<T>, k: usize) -> Result<HillFit, StatsError> {
  assert!(k >= 1, "k must satisfy `k >= 1`, got k = {k:?}");
  let x = data
    .iter()
    .map(|v| v.to_f64().unwrap_or(f64::NAN))
    .collect::<Vec<f64>>();
  validate(&x, 0)?;
  let mut positive = x.into_iter().filter(|v| *v > 0.0).collect::<Vec<f64>>();
  if positive.len() <= k {
    return Err(StatsError::InsufficientData {
      needed: k.saturating_add(1),
      got: positive.len(),
    });
  }
  positive.sort_by(|a, b| b.total_cmp(a));
  let threshold = positive[k];
  let log_threshold = threshold.ln();
  let xi = positive[..k]
    .iter()
    .map(|x| x.ln() - log_threshold)
    .sum::<f64>()
    / k as f64;
  Ok(HillFit {
    xi,
    alpha: 1.0 / xi,
    std_error: xi / (k as f64).sqrt(),
    k,
    threshold,
    nobs: positive.len(),
  })
}
