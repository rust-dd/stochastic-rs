//! # Antithetic Variates
//!
//! $$
//! \hat{\mu}_{\mathrm{AV}} = \frac{1}{N}\sum_{i=1}^{N}
//! \frac{f(Z_i)+f(-Z_i)}{2}
//! $$
//!
//! Reference: Glasserman (2003), Springer, §4.2, DOI 10.1007/978-0-387-21617-1

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use super::McEstimate;
use super::estimate_from_samples;
use super::welford::par_welford_seeded;
use crate::traits::FloatExt;

/// The mean of `(payoff(Z) + payoff(−Z)) / 2` over `n_paths` draws of a `dim`-dimensional standard
/// normal `Z` from `seed`.
pub fn estimate<T, F, S>(n_paths: usize, dim: usize, payoff: F, seed: &S) -> McEstimate<T>
where
  T: FloatExt,
  F: Fn(&Array1<T>) -> T,
  S: SeedExt,
{
  let two = T::from_f64_fast(2.0);
  let mut normal = SimdNormal::<T>::new(T::zero(), T::one()).seeded(seed);
  let mut z = Array1::<T>::zeros(dim);
  estimate_from_samples((0..n_paths).map(|_| {
    normal.fill_slice(z.as_slice_mut().expect("contiguous"));
    let up = payoff(&z);
    z.mapv_inplace(|v| -v);
    (up + payoff(&z)) / two
  }))
}

/// [`estimate`] on the pool, one stream per chunk of `n_paths` forked from `seed`: the same bits
/// under any pool size.
pub fn estimate_par<T, F, S>(n_paths: usize, dim: usize, payoff: F, seed: &S) -> McEstimate<T>
where
  T: FloatExt,
  F: Fn(&Array1<T>) -> T + Sync,
  S: SeedExt,
{
  let two = T::from_f64_fast(2.0);
  let acc = par_welford_seeded(n_paths, seed, |_, normal| {
    let mut z = normal.sample_n(dim);
    let up = payoff(&z);
    z.mapv_inplace(|v: T| -v);
    (up + payoff(&z)) / two
  });
  McEstimate::from(&acc)
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// Antithetic should give the correct mean for E[max(Z,0)] = 1/√(2π)
  /// and lower variance than plain MC for this monotone payoff.
  #[test]
  fn antithetic_reduces_variance_for_monotone_payoff() {
    let n = 50_000;
    let dim = 1;
    let payoff = |z: &Array1<f64>| z[0].max(0.0);

    let av = estimate(n, dim, payoff, &Deterministic::new(1));

    // Plain MC for comparison
    let mut normal = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(2));
    let plain_se = estimate_from_samples((0..n).map(|_| payoff(&normal.sample_n(dim)))).std_err;

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

  /// The pair average of `max(Z, 0)` is `|Z| / 2` (mean `1/√(2π)`, variance `(1 − 2/π) / 4`): a
  /// wrong chunk weighting in the merge moves the standard error, which a constant payoff hides.
  #[test]
  fn antithetic_par_matches_the_analytic_moments() {
    let n = 200_000;
    let est = estimate_par(
      n,
      1,
      |z: &Array1<f64>| z[0].max(0.0),
      &Deterministic::new(5),
    );
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

  #[test]
  fn the_estimate_is_a_function_of_the_seed() {
    let payoff = |z: &Array1<f64>| z[0].max(0.0);
    let a = estimate(10_000, 1, payoff, &Deterministic::new(7));
    let b = estimate(10_000, 1, payoff, &Deterministic::new(7));
    let c = estimate(10_000, 1, payoff, &Deterministic::new(8));
    assert_eq!(a.mean, b.mean);
    assert_ne!(a.mean, c.mean);
  }

  #[test]
  fn the_parallel_estimate_does_not_depend_on_the_pool() {
    let payoff = |z: &Array1<f64>| z[0].max(0.0);
    let run = |threads: usize| {
      rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(|| estimate_par(50_000, 1, payoff, &Deterministic::new(3)))
    };
    let (one, four) = (run(1), run(4));
    assert_eq!(one.mean.to_bits(), four.mean.to_bits());
    assert_eq!(one.std_err.to_bits(), four.std_err.to_bits());
  }
}
