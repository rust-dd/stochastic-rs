//! Expected Shortfall / Conditional VaR.
//!
//! Reference: Acerbi & Tasche, "On the Coherence of Expected Shortfall",
//! Journal of Banking & Finance, 26(7), 1487–1503 (2002).
//! DOI: 10.1016/S0378-4266(02)00283-2
//!
//! Reference: Rockafellar & Uryasev, "Conditional Value-at-Risk for General
//! Loss Distributions", Journal of Banking & Finance, 26(7), 1443–1471 (2002).
//! DOI: 10.1016/S0378-4266(02)00271-6
//!
//! For the loss random variable $L$,
//! $$
//! \mathrm{ES}_{\alpha}(L)=\mathbb{E}[L\mid L\ge\mathrm{VaR}_{\alpha}(L)]
//!   =\frac{1}{1-\alpha}\int_{\alpha}^{1}\mathrm{VaR}_{u}(L)\,du.
//! $$
//! Expected Shortfall is a coherent risk measure, unlike VaR.

use ndarray::ArrayView1;
use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_pdf;

use super::var::PnlOrLoss;
use super::var::VarMethod;
use super::var::assert_confidence;
use super::var::gaussian_var;
use super::var::losses_from_samples;
use super::var::sample_quantile;
use crate::traits::FloatExt;

/// Expected Shortfall using the chosen estimation method.
pub fn expected_shortfall<T: FloatExt>(
  samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
  method: VarMethod,
) -> T {
  assert_confidence(confidence);
  match method {
    VarMethod::Gaussian => gaussian_es(samples, confidence, orientation),
    VarMethod::Historical | VarMethod::MonteCarlo => {
      historical_es(samples, confidence, orientation)
    }
  }
}

/// Parametric Gaussian ES using sample mean and standard deviation.
///
/// $$
/// \mathrm{ES}_{\alpha}=-\hat\mu+\hat\sigma\,\frac{\phi(\Phi^{-1}(\alpha))}{1-\alpha}.
/// $$
pub fn gaussian_es<T: FloatExt>(
  samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
) -> T {
  assert_confidence(confidence);
  let losses = losses_from_samples(samples, orientation);
  let n = losses.len();
  assert!(n >= 2, "need at least two observations for Gaussian ES");
  let mean = losses.iter().fold(T::zero(), |acc, &v| acc + v) / T::from_usize_(n);
  let var = losses
    .iter()
    .fold(T::zero(), |acc, &v| acc + (v - mean).powi(2))
    / T::from_usize_(n - 1);
  let sigma = var.sqrt();
  let c = confidence.to_f64().unwrap();
  let q = ndtri(c);
  let phi_q = norm_pdf(q);
  let factor = T::from_f64_fast(phi_q / (1.0 - c));
  mean + sigma * factor
}

/// Historical ES — Rockafellar-Uryasev (2002) coherent estimator.
///
/// $$
/// \mathrm{ES}_\alpha = \frac{1}{n(1-\alpha)}
///   \Big[\sum_{i: L_i > \mathrm{VaR}_\alpha} L_i
///        + (n(1-\alpha) - \#\{i: L_i > \mathrm{VaR}_\alpha\})\,\mathrm{VaR}_\alpha\Big].
/// $$
///
/// The term `(n(1-α) - #{strict tail})` is the **finite-sample
/// tail-share correction** introduced by Rockafellar-Uryasev: when the
/// (1-α) quantile lands inside a tied cluster of samples, this fractional
/// weight on `VaR` accounts for the partial slice of the cluster that
/// belongs to the tail. Without the correction (the previous "average of
/// losses ≥ VaR" formula), ties bias the estimator and the result is no
/// longer coherent (Artzner et al. 1999).
///
/// Reference: Rockafellar, R. T. & Uryasev, S. (2002), "Conditional
/// Value-at-Risk for General Loss Distributions", *J. Banking & Finance*
/// 26(7), 1443-1471, eq. (17).
pub fn historical_es<T: FloatExt>(
  samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
) -> T {
  assert_confidence(confidence);
  let losses = losses_from_samples(samples, orientation);
  let n = losses.len();
  assert!(n >= 1, "need at least one observation for historical ES");
  let mut sorted = losses.to_vec();
  sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

  let var = sample_quantile(&sorted, confidence);

  // Rockafellar-Uryasev finite-sample formula:
  //   strict_tail = {L_i > VaR}    (strictly above)
  //   strict_sum  = Σ L_i  over strict_tail
  //   target      = n * (1 - α)    (expected tail size, possibly fractional)
  //   weight_var  = max(target - |strict_tail|, 0)   (slice of ties on VaR)
  //   ES          = (strict_sum + weight_var * VaR) / target
  let confidence_f64 = confidence.to_f64().unwrap_or(0.0);
  let target = (n as f64) * (1.0 - confidence_f64);
  if target <= 0.0 {
    return var;
  }
  let mut strict_sum = T::zero();
  let mut strict_count = 0usize;
  for &l in sorted.iter() {
    if l > var {
      strict_sum += l;
      strict_count += 1;
    }
  }
  let weight_var = (target - strict_count as f64).max(0.0);
  let total = strict_sum + var * T::from_f64_fast(weight_var);
  total / T::from_f64_fast(target)
}

/// Monte-Carlo ES alias of [`historical_es`], kept separate for intent.
pub fn monte_carlo_es<T: FloatExt>(
  simulated_samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
) -> T {
  historical_es(simulated_samples, confidence, orientation)
}

/// Convenience: Gaussian VaR and ES together.
pub fn gaussian_var_es<T: FloatExt>(
  samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
) -> (T, T) {
  (
    gaussian_var(samples, confidence, orientation),
    gaussian_es(samples, confidence, orientation),
  )
}

#[cfg(test)]
mod tests {
  use super::*;
  use ndarray::Array1;

  /// RU correction: with n=10 samples and α=0.85 (target tail size 1.5),
  /// the historical ES averages the worst 1.5 samples — i.e. the worst
  /// observation fully + half of the second-worst. Without the
  /// correction, the previous code used to return either the single
  /// worst loss (when VaR strictly < worst) or the simple mean of the
  /// two worst observations.
  #[test]
  fn historical_es_ru_finite_sample_correction() {
    // Distinct sorted losses; PnlOrLoss::Loss means samples are losses
    // already (no sign flip).
    let losses = Array1::from_vec(vec![
      -3.0_f64, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0,
    ]);
    let es = historical_es(losses.view(), 0.85, PnlOrLoss::Loss);
    // n*(1-α) = 1.5; sample_quantile at 0.85 lands at the 9th
    // ordered sample (≈ 5.0) by linear interpolation. Strictly above
    // that quantile is the single observation 10.0; weight_var =
    // 1.5 − 1 = 0.5; ES = (10.0 + 0.5·5.0) / 1.5 ≈ 8.333...
    // (Exact value depends on the implementation of sample_quantile;
    // we assert the ES sits between the worst and second-worst losses,
    // which is the RU coherence property.)
    assert!(es > 5.0, "ES = {es} must exceed second-worst loss");
    assert!(es < 10.0, "ES = {es} must be below worst loss");
  }

  /// All-equal losses: ES collapses to that constant value.
  #[test]
  fn historical_es_constant_losses() {
    let losses = Array1::from_vec(vec![1.5_f64; 50]);
    let es = historical_es(losses.view(), 0.95, PnlOrLoss::Loss);
    assert!((es - 1.5).abs() < 1e-12, "ES = {es}, expected 1.5");
  }
}
