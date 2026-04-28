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

/// Historical ES — average of losses in the upper $(1-\alpha)$ tail.
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
  sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

  let var = sample_quantile(&sorted, confidence);

  let tail: Vec<T> = sorted.iter().copied().filter(|&v| v >= var).collect();
  if tail.is_empty() {
    return var;
  }
  tail.iter().fold(T::zero(), |acc, &v| acc + v) / T::from_usize_(tail.len())
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
