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

use super::McEstimate;
use super::estimate_from_samples;
use super::par_welford;
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
///
/// The paths are cut into a fixed number of chunks that depends on `n_paths`
/// alone. Each chunk folds its payoffs into its own [`Welford`](super::Welford)
/// accumulator on a worker and the chunk accumulators are merged in chunk
/// order, so no per-path storage is kept and the serial tail is one merge per
/// chunk.
pub fn estimate_par<T, F>(n_paths: usize, dim: usize, payoff: F) -> McEstimate<T>
where
  T: FloatExt,
  F: Fn(&Array1<T>) -> T + Sync,
{
  let two = T::from_f64_fast(2.0);
  let acc = par_welford(n_paths, |_| {
    let z = T::normal_array(dim, T::zero(), T::one());
    let neg_z = z.mapv(|v| -v);
    (payoff(&z) + payoff(&neg_z)) / two
  });
  McEstimate::from(&acc)
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

  /// The pair average of `max(Z, 0)` is `|Z| / 2`: mean `1/√(2π)`, variance
  /// `(1 − 2/π) / 4`. A wrong chunk weighting in the parallel merge would move
  /// the standard error, which a constant payoff cannot show.
  #[test]
  fn antithetic_par_matches_the_analytic_moments() {
    let n = 200_000;
    let est = estimate_par(n, 1, |z: &Array1<f64>| z[0].max(0.0));
    let mean = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    let std_err = ((1.0 - 2.0 / std::f64::consts::PI) / 4.0 / n as f64).sqrt();
    assert!(
      (est.mean - mean).abs() < 5.0 * std_err,
      "mean {} against {mean}",
      est.mean
    );
    assert!(
      (est.std_err / std_err - 1.0).abs() < 0.05,
      "std_err {} against {std_err}",
      est.std_err
    );
    assert_eq!(est.n_samples, n);
  }
}
