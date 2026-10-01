//! # Antithetic Variates
//!
//! $$
//! \hat{\mu}_{\mathrm{AV}} = \frac{1}{N}\sum_{i=1}^{N}
//! \frac{f(Z_i)+f(-Z_i)}{2}
//! $$
//!
//! Reference: Glasserman (2003), *Monte Carlo Methods in Financial Engineering*, §4.2.
//! DOI: 10.1007/978-0-387-21617-1

use ndarray::Array1;
use ndarray::parallel::prelude::*;

use super::McEstimate;
use super::estimate_from_samples;
use crate::traits::FloatExt;

/// Antithetic variates MC estimate (sequential).
///
/// Generates `n_paths` pairs `(Z, −Z)` of `dim`-dimensional standard normals
/// and returns the averaged payoff.
pub fn estimate<T, F>(n_paths: usize, dim: usize, payoff: F) -> McEstimate<T>
where
  T: FloatExt,
  F: Fn(&Array1<T>) -> T,
{
  let two = T::from_f64_fast(2.0);
  estimate_from_samples((0..n_paths).map(|_| {
    let z = T::normal_array(dim, T::zero(), T::one());
    let neg_z = z.mapv(|v| -v);
    (payoff(&z) + payoff(&neg_z)) / two
  }))
}

/// Antithetic variates MC estimate (parallel via rayon).
pub fn estimate_par<T, F>(n_paths: usize, dim: usize, payoff: F) -> McEstimate<T>
where
  T: FloatExt,
  F: Fn(&Array1<T>) -> T + Sync,
{
  let two = T::from_f64_fast(2.0);
  let results: Vec<T> = (0..n_paths)
    .into_par_iter()
    .map(|_| {
      let z = T::normal_array(dim, T::zero(), T::one());
      let neg_z = z.mapv(|v| -v);
      (payoff(&z) + payoff(&neg_z)) / two
    })
    .collect();

  estimate_from_samples(results)
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Antithetic should give the correct mean for E[max(Z,0)] = 1/√(2π)
  /// and lower variance than plain MC for this monotone payoff.
  #[test]
  fn antithetic_reduces_variance_for_monotone_payoff() {
    let n = 50_000;
    let dim = 1;
    let payoff = |z: &Array1<f64>| z[0].max(0.0);

    let av = estimate(n, dim, payoff);

    // Plain MC for comparison
    let plain_se =
      estimate_from_samples((0..n).map(|_| payoff(&f64::normal_array(dim, 0.0, 1.0)))).std_err;

    let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    assert!(
      (av.mean - expected).abs() < 3.0 * av.std_err + 0.01,
      "AV mean {:.4} far from expected {expected:.4}",
      av.mean
    );
    assert!(
      av.std_err < plain_se * 1.1,
      "AV std_err {:.6} should be <= plain {plain_se:.6}",
      av.std_err
    );
  }
}
