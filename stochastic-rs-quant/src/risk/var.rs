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
use stochastic_rs_distributions::special::ndtri;

use super::input::finite;
use super::input::validate_samples;
use crate::QuantError;
use crate::total_order::ascending;
use crate::traits::RealExt;

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
#[non_exhaustive]
pub enum VarMethod {
  /// Parametric Gaussian VaR using the sample mean and standard deviation.
  #[default]
  Gaussian,
  /// Historical simulation: the losses' type-7 quantile (Hyndman & Fan 1996, numpy's default).
  Historical,
  /// Monte Carlo: the `Historical` quantile of simulated losses.
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

/// VaR by `method`, with that method's errors; panics unless `0 < confidence < 1`.
pub fn value_at_risk<T: RealExt>(
  samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
  method: VarMethod,
) -> Result<T, QuantError> {
  assert_confidence(confidence);
  match method {
    VarMethod::Gaussian => gaussian_var(samples, confidence, orientation),
    VarMethod::Historical => historical_var(samples, confidence, orientation),
    VarMethod::MonteCarlo => monte_carlo_var(samples, confidence, orientation),
  }
}

/// Gaussian VaR from the sample moments; `InsufficientData` below two samples, `InvalidInput`
/// for a non-finite one, `Numerical` if a moment overflows; panics unless `0 < confidence < 1`.
///
/// $$
/// \mathrm{VaR}_{\alpha}=-\hat\mu+\hat\sigma\,\Phi^{-1}(\alpha).
/// $$
pub fn gaussian_var<T: RealExt>(
  samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
) -> Result<T, QuantError> {
  assert_confidence(confidence);
  let (mean, sigma) = loss_moments(samples, orientation, "gaussian VaR")?;
  let z = T::from_f64_fast(standard_normal_quantile(confidence.to_f64().unwrap()));
  Ok(mean + sigma * z)
}

/// Historical VaR, the losses' type-7 quantile (Hyndman & Fan 1996); `InsufficientData` for no
/// sample, `InvalidInput` for a non-finite one, and a panic unless `0 < confidence < 1`.
pub fn historical_var<T: RealExt>(
  samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
) -> Result<T, QuantError> {
  empirical_var(samples, confidence, orientation, "historical VaR")
}

/// Monte Carlo VaR: [`historical_var`] over simulated samples.
pub fn monte_carlo_var<T: RealExt>(
  simulated_samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
) -> Result<T, QuantError> {
  empirical_var(
    simulated_samples,
    confidence,
    orientation,
    "Monte Carlo VaR",
  )
}

/// [`monte_carlo_var`] over `n_samples` draws that continue `sampler`'s stream, so a heavy-tailed
/// law (`SimdStudentT`, `SimdAlphaStable`, `SimdGpd`) can replace the Gaussian assumption.
pub fn monte_carlo_var_with_sampler<T, D>(
  sampler: &mut D,
  n_samples: usize,
  confidence: T,
  orientation: PnlOrLoss,
) -> Result<T, QuantError>
where
  T: RealExt,
  D: stochastic_rs_distributions::traits::DistributionSampler<T>,
{
  let samples = sampler.sample_n(n_samples);
  monte_carlo_var(samples.view(), confidence, orientation)
}

fn empirical_var<T: RealExt>(
  samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
  what: &'static str,
) -> Result<T, QuantError> {
  assert_confidence(confidence);
  let sorted = sorted_losses(samples, orientation, what)?;
  Ok(sample_quantile(&sorted, confidence))
}

/// The losses sorted by [`ascending`], which puts NaN last and ±∞ at the ends, so the samples are
/// scanned for a non-finite one only when an end is not finite.
pub(crate) fn sorted_losses<T: RealExt>(
  samples: ArrayView1<T>,
  orientation: PnlOrLoss,
  what: &'static str,
) -> Result<Vec<T>, QuantError> {
  let mut sorted = match orientation {
    PnlOrLoss::Pnl => samples.iter().map(|&v| -v).collect::<Vec<_>>(),
    PnlOrLoss::Loss => samples.to_vec(),
  };
  sorted.sort_by(ascending);
  let finite_end = |end: Option<&T>| end.is_some_and(|v| v.is_finite());
  if !(finite_end(sorted.first()) && finite_end(sorted.last())) {
    validate_samples(samples, what, 1)?;
  }
  Ok(sorted)
}

/// Mean and Bessel-corrected deviation of the losses, `Numerical` if either overflows; the sum is
/// non-finite whenever a sample is, so only then are the samples scanned.
pub(crate) fn loss_moments<T: RealExt>(
  samples: ArrayView1<T>,
  orientation: PnlOrLoss,
  what: &'static str,
) -> Result<(T, T), QuantError> {
  let losses = losses_from_samples(samples, orientation);
  let n = losses.len();
  let sum = losses.iter().fold(T::zero(), |acc, &v| acc + v);
  if n < 2 || !sum.is_finite() {
    validate_samples(samples, what, 2)?;
  }
  let mean = finite(sum, what, "the sum")? / T::from_usize_(n);
  let var = losses
    .iter()
    .fold(T::zero(), |acc, &v| acc + (v - mean).powi(2))
    / T::from_usize_(n - 1);
  Ok((mean, finite(var, what, "the variance")?.sqrt()))
}

pub(crate) fn losses_from_samples<T: RealExt>(
  samples: ArrayView1<T>,
  orientation: PnlOrLoss,
) -> Array1<T> {
  match orientation {
    PnlOrLoss::Pnl => samples.mapv(|v| -v),
    PnlOrLoss::Loss => samples.to_owned(),
  }
}

pub(crate) fn sample_quantile<T: RealExt>(sorted: &[T], confidence: T) -> T {
  let n = sorted.len();
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

pub(crate) fn assert_confidence<T: RealExt>(confidence: T) {
  assert!(
    confidence > T::zero() && confidence < T::one(),
    "confidence must satisfy `confidence > 0.0 && confidence < 1.0`, got confidence = {confidence:?}"
  );
}

fn standard_normal_quantile(p: f64) -> f64 {
  ndtri(p)
}
