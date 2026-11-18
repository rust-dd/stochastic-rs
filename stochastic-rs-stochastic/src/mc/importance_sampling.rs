//! # Importance Sampling
//!
//! $$
//! \mathbb{E}_P[f(Z)] = \mathbb{E}_Q\!\left[f(Z)\,\frac{dP}{dQ}(Z)\right],\quad
//! \frac{dP}{dQ}(z)=\exp\!\bigl(-z\!\cdot\!\theta+\tfrac{|\theta|^2}{2}\bigr)
//! $$
//!
//! Gaussian mean-shift importance sampling.
//!
//! Reference: Glasserman (2003), Springer, §4.6, DOI 10.1007/978-0-387-21617-1

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use super::McEstimate;
use super::estimate_from_samples;
use crate::traits::FloatExt;

/// `E[payoff(Z)]` for `Z ~ N(0, I)`, from `seed`'s normals moved by `shift` and reweighted by
/// `dP/dQ`; `payoff` receives the moved draw; panics unless `shift.len() == dim`.
pub fn estimate<T, F, S>(
  n_paths: usize,
  dim: usize,
  payoff: F,
  shift: &Array1<T>,
  seed: &S,
) -> McEstimate<T>
where
  T: FloatExt,
  F: Fn(&Array1<T>) -> T,
  S: SeedExt,
{
  assert_eq!(shift.len(), dim, "shift dimension must match dim");

  let two = T::from_f64_fast(2.0);
  let shift_norm_sq: T = shift.iter().map(|&s| s * s).sum();

  let mut normal = SimdNormal::<T>::new(T::zero(), T::one()).seeded(seed);
  let mut z = Array1::<T>::zeros(dim);
  estimate_from_samples((0..n_paths).map(|_| {
    normal.fill_slice(z.as_slice_mut().expect("contiguous"));
    z += shift;
    let dot: T = z.iter().zip(shift.iter()).map(|(&zi, &si)| zi * si).sum();
    let log_weight = -dot + shift_norm_sq / two;
    let weight = log_weight.exp();
    payoff(&z) * weight
  }))
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// Zero shift should recover plain MC for E[max(Z,0)] = 1/√(2π).
  #[test]
  fn zero_shift_matches_plain_mc() {
    let n = 50_000;
    let dim = 1;
    let payoff = |z: &Array1<f64>| z[0].max(0.0);
    let shift = Array1::zeros(dim);

    let is = estimate(n, dim, payoff, &shift, &Deterministic::new(4));
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

    let is = estimate(n, dim, payoff, &shift, &Deterministic::new(5));
    let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();

    assert!(
      (is.mean - expected).abs() < 0.05,
      "IS mean {:.4} far from expected {expected:.4}",
      is.mean
    );
  }
}
