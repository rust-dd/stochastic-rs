//! Stratified standard normals, one uniform per stratum, each column of a sample on its own.
//!
//! $$
//! Z_k = \Phi^{-1}\!\left(\frac{k-1+U_k}{K}\right),\quad
//! U_k\sim\mathrm{Unif}(0,1),\;k=1,\dots,K
//! $$
//!
//! Equal strata of `[0, 1]` through the inverse normal CDF give variance-reduced normal samples.
//!
//! Reference: Glasserman (2003), Springer, §4.3, DOI 10.1007/978-0-387-21617-1

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::uniform::SimdUniform;

use super::McEstimate;
use super::estimate_from_samples;
use crate::traits::FloatExt;

/// `n` standard normals, `Φ⁻¹` of one uniform from `seed` in each of `n` equal strata of `[0, 1]`.
pub fn stratified_normals_1d<T: FloatExt, S: SeedExt>(n: usize, seed: &S) -> Array1<T> {
  strata(&mut SimdUniform::<f64>::new(0.0, 1.0).seeded(seed), n)
}

/// Draws in `f64` whatever `T` is: in `f32`, `k + u` would keep only 7 bits of `u` at `k = 10⁵`.
fn strata<T: FloatExt>(uniform: &mut Seeded<SimdUniform<f64>>, n: usize) -> Array1<T> {
  let n_f = n as f64;
  let u = uniform.sample_n(n);
  Array1::from_iter(u.iter().enumerate().map(|(k, &u)| {
    let u_strat = (k as f64 + u) / n_f;
    T::from_f64_fast(ndtri(u_strat.clamp(1e-10, 1.0 - 1e-10)))
  }))
}

/// `(n_samples, dim)` stratified standard normals, each column stratified on its own, all from one
/// stream of `seed`.
pub fn stratified_normals<T: FloatExt, S: SeedExt>(
  n_samples: usize,
  dim: usize,
  seed: &S,
) -> Array2<T> {
  let mut uniform = SimdUniform::<f64>::new(0.0, 1.0).seeded(seed);
  let mut out = Array2::<T>::zeros((n_samples, dim));
  for mut col in out.columns_mut() {
    col.assign(&strata::<T>(&mut uniform, n_samples));
  }
  out
}

/// The mean of `payoff` over [`stratified_normals`] from `seed`.
pub fn estimate<T, F, S>(n_paths: usize, dim: usize, payoff: F, seed: &S) -> McEstimate<T>
where
  T: FloatExt,
  F: Fn(&Array1<T>) -> T,
  S: SeedExt,
{
  let samples = stratified_normals::<T, S>(n_paths, dim, seed);

  estimate_from_samples((0..n_paths).map(|i| payoff(&samples.row(i).to_owned())))
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// Stratified normals should have mean ≈ 0 and variance ≈ 1.
  #[test]
  fn stratified_normals_have_correct_moments() {
    let n = 10_000;
    let z = stratified_normals_1d::<f64, _>(n, &Deterministic::new(6));
    let mean = z.sum() / n as f64;
    let var = z.mapv(|x| (x - mean) * (x - mean)).sum() / n as f64;
    assert!(mean.abs() < 0.05, "mean = {mean:.4}");
    assert!((var - 1.0).abs() < 0.1, "var = {var:.4}");
  }

  /// Stratified sampling should give the correct mean for E[exp(Z)].
  #[test]
  fn stratified_estimate_correct() {
    let n = 10_000;
    let dim = 1;
    let payoff = |z: &Array1<f64>| z[0].exp();

    let strat = estimate(n, dim, payoff, &Deterministic::new(7));
    let expected = (0.5_f64).exp();

    assert!(
      (strat.mean - expected).abs() < 3.0 * strat.std_err + 0.02,
      "Stratified mean {:.4} far from expected {expected:.4}",
      strat.mean
    );
  }
}
