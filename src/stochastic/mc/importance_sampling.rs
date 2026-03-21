//! # Importance Sampling
//!
//! $$
//! \mathbb{E}_P[f(Z)] = \mathbb{E}_Q\!\left[f(Z)\,\frac{dP}{dQ}(Z)\right],\quad
//! \frac{dP}{dQ}(z)=\exp\!\bigl(-z\!\cdot\!\theta+\tfrac{|\theta|^2}{2}\bigr)
//! $$
//!
//! Gaussian mean-shift importance sampling.
//!
//! Reference: Glasserman (2003), *Monte Carlo Methods in Financial Engineering*, §4.6.
//! DOI: 10.1007/978-0-387-21617-1

use ndarray::Array1;

use super::McEstimate;
use crate::traits::FloatExt;

/// Importance sampling MC estimate with Gaussian mean-shift.
///
/// Shifts the sampling distribution from `N(0, I)` to `N(θ, I)`.
/// Useful for rare-event simulation (e.g., deep out-of-the-money options).
///
/// The `payoff` closure receives the shifted samples `Z + θ`.
pub fn estimate<T, F>(
  n_paths: usize,
  dim: usize,
  payoff: F,
  shift: &Array1<T>,
) -> McEstimate<T>
where
  T: FloatExt,
  F: Fn(&Array1<T>) -> T,
{
  assert_eq!(shift.len(), dim, "shift dimension must match dim");

  let two = T::from_f64_fast(2.0);
  let shift_norm_sq: T = shift.iter().map(|&s| s * s).sum();
  let mut sum = T::zero();
  let mut sum_sq = T::zero();

  for _ in 0..n_paths {
    let z_std = T::normal_array(dim, T::zero(), T::one());
    let z = &z_std + shift;
    let dot: T = z.iter().zip(shift.iter()).map(|(&zi, &si)| zi * si).sum();
    let log_weight = -dot + shift_norm_sq / two;
    let weight = log_weight.exp();
    let y = payoff(&z) * weight;
    sum += y;
    sum_sq += y * y;
  }

  let n = T::from_usize_(n_paths);
  let mean = sum / n;
  let variance = sum_sq / n - mean * mean;
  let std_err = (variance / n).sqrt();

  McEstimate {
    mean,
    std_err,
    n_samples: n_paths,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Zero shift should recover plain MC for E[max(Z,0)] = 1/√(2π).
  #[test]
  fn zero_shift_matches_plain_mc() {
    let n = 50_000;
    let dim = 1;
    let payoff = |z: &Array1<f64>| z[0].max(0.0);
    let shift = Array1::zeros(dim);

    let is = estimate(n, dim, payoff, &shift);
    let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();

    assert!(
      (is.mean - expected).abs() < 3.0 * is.std_err + 0.01,
      "IS mean {:.4} far from expected {expected:.4}",
      is.mean
    );
  }

  /// Shifting towards the tail should still give a correct (unbiased) estimate.
  #[test]
  fn shifted_estimate_correct() {
    let n = 50_000;
    let dim = 1;
    let payoff = |z: &Array1<f64>| z[0].max(0.0);
    let shift = Array1::from_vec(vec![1.0]);

    let is = estimate(n, dim, payoff, &shift);
    let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();

    assert!(
      (is.mean - expected).abs() < 0.05,
      "IS mean {:.4} far from expected {expected:.4}",
      is.mean
    );
  }
}
