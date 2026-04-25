//! Value at Risk: parametric (Gaussian), historical simulation, and Monte-Carlo.
//!
//! Reference: Jorion, "Value at Risk: The New Benchmark for Managing Financial
//! Risk", 3rd ed., McGraw-Hill (2007).
//!
//! Reference: Basel Committee on Banking Supervision, "International Convergence
//! of Capital Measurement and Capital Standards: A Revised Framework" (2006).
//!
//! The library adopts the standard *loss-positive* convention: VaR at level
//! $\alpha\in(0,1)$ is the smallest threshold $\ell$ such that the probability
//! of a loss larger than $\ell$ does not exceed $1-\alpha$:
//! $$
//! \mathrm{VaR}_{\alpha}(L)=\inf\{\ell\in\mathbb{R}:\mathbb{P}(L>\ell)\le 1-\alpha\}.
//! $$
//! When inputs are PnL samples (profit positive), pass `PnlOrLoss::Pnl` to
//! automatically negate before ranking.

use std::fmt::Display;

use ndarray::Array1;
use ndarray::ArrayView1;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

use crate::traits::FloatExt;

/// Interpretation of the input sample.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PnlOrLoss {
  /// Samples are PnL (higher is better); losses are $-x$.
  #[default]
  Pnl,
  /// Samples are already losses (higher is worse).
  Loss,
}

impl Display for PnlOrLoss {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Pnl => write!(f, "PnL"),
      Self::Loss => write!(f, "Loss"),
    }
  }
}

/// Supported VaR estimation methods.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VarMethod {
  /// Parametric Gaussian VaR using the sample mean and standard deviation.
  #[default]
  Gaussian,
  /// Historical simulation: empirical quantile of losses.
  Historical,
  /// Monte Carlo: empirical quantile of simulated losses (interface mirrors
  /// `Historical` but documents intent separately).
  MonteCarlo,
}

impl Display for VarMethod {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Gaussian => write!(f, "Gaussian"),
      Self::Historical => write!(f, "Historical simulation"),
      Self::MonteCarlo => write!(f, "Monte Carlo"),
    }
  }
}

/// Compute VaR for the given sample using the chosen method.
pub fn value_at_risk<T: FloatExt>(
  samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
  method: VarMethod,
) -> T {
  assert_confidence(confidence);
  match method {
    VarMethod::Gaussian => gaussian_var(samples, confidence, orientation),
    VarMethod::Historical | VarMethod::MonteCarlo => {
      historical_var(samples, confidence, orientation)
    }
  }
}

/// Parametric Gaussian VaR using the sample moments.
///
/// $$
/// \mathrm{VaR}_{\alpha}=-\hat\mu+\hat\sigma\,\Phi^{-1}(\alpha).
/// $$
pub fn gaussian_var<T: FloatExt>(
  samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
) -> T {
  assert_confidence(confidence);
  let losses = losses_from_samples(samples, orientation);
  let n = losses.len();
  assert!(n >= 2, "need at least two observations for Gaussian VaR");
  let mean = losses.iter().fold(T::zero(), |acc, &v| acc + v) / T::from_usize_(n);
  let var = losses
    .iter()
    .fold(T::zero(), |acc, &v| acc + (v - mean).powi(2))
    / T::from_usize_(n - 1);
  let sigma = var.sqrt();
  let z = T::from_f64_fast(standard_normal_quantile(confidence.to_f64().unwrap()));
  mean + sigma * z
}

/// Historical-simulation VaR (empirical loss quantile).
pub fn historical_var<T: FloatExt>(
  samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
) -> T {
  assert_confidence(confidence);
  let losses = losses_from_samples(samples, orientation);
  let n = losses.len();
  assert!(n >= 1, "need at least one observation for historical VaR");
  let mut sorted = losses.to_vec();
  sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
  sample_quantile(&sorted, confidence)
}

/// Monte-Carlo VaR (alias of [`historical_var`] — kept as a separate function
/// for clarity of intent in client code).
pub fn monte_carlo_var<T: FloatExt>(
  simulated_samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
) -> T {
  historical_var(simulated_samples, confidence, orientation)
}

pub(crate) fn losses_from_samples<T: FloatExt>(
  samples: ArrayView1<T>,
  orientation: PnlOrLoss,
) -> Array1<T> {
  match orientation {
    PnlOrLoss::Pnl => samples.mapv(|v| -v),
    PnlOrLoss::Loss => samples.to_owned(),
  }
}

pub(crate) fn sample_quantile<T: FloatExt>(sorted: &[T], confidence: T) -> T {
  let n = sorted.len();
  assert!(n >= 1, "empty sample");
  let idx = (confidence * T::from_usize_(n - 1)).to_f64().unwrap_or(0.0);
  let lo = idx.floor() as usize;
  let hi = idx.ceil() as usize;
  if lo == hi {
    sorted[lo]
  } else {
    let w = T::from_f64_fast(idx - idx.floor());
    sorted[lo] * (T::one() - w) + sorted[hi] * w
  }
}

pub(crate) fn assert_confidence<T: FloatExt>(c: T) {
  let v = c.to_f64().unwrap_or(0.0);
  assert!(v > 0.0 && v < 1.0, "confidence must lie in (0, 1); got {v}");
}

fn standard_normal_quantile(p: f64) -> f64 {
  Normal::new(0.0, 1.0)
    .expect("standard normal")
    .inverse_cdf(p)
}
