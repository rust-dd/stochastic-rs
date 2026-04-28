//! # Stratified Sampling
//!
//! $$
//! Z_k = \Phi^{-1}\!\left(\frac{k-1+U_k}{K}\right),\quad
//! U_k\sim\mathrm{Unif}(0,1),\;k=1,\dots,K
//! $$
//!
//! Partitions the uniform `[0,1]` into equal strata and applies the inverse
//! normal CDF to obtain variance-reduced normal samples.
//!
//! Reference: Glasserman (2003), *Monte Carlo Methods in Financial Engineering*, §4.3.
//! DOI: 10.1007/978-0-387-21617-1

use ndarray::Array1;
use ndarray::Array2;
use rand::Rng;
use stochastic_rs_distributions::special::ndtri;

use super::McEstimate;
use crate::traits::FloatExt;

/// Generate `n` stratified standard normal samples in one dimension.
///
/// Divides `[0,1]` into `n` equal strata, draws one uniform per stratum,
/// and applies `Φ⁻¹`.
pub fn stratified_normals_1d<T: FloatExt>(n: usize) -> Array1<T> {
  let mut rng = rand::rng();
  let n_f = n as f64;
  let mut out = Array1::<T>::zeros(n);

  for k in 0..n {
    let u: f64 = rng.random();
    let u_strat = (k as f64 + u) / n_f;
    let z = ndtri(u_strat.clamp(1e-10, 1.0 - 1e-10));
    out[k] = T::from_f64_fast(z);
  }

  out
}

/// Generate `(n_samples, dim)` stratified standard normals.
///
/// Each dimension is independently stratified.
pub fn stratified_normals<T: FloatExt>(n_samples: usize, dim: usize) -> Array2<T> {
  let mut out = Array2::<T>::zeros((n_samples, dim));
  for j in 0..dim {
    let col = stratified_normals_1d::<T>(n_samples);
    for i in 0..n_samples {
      out[[i, j]] = col[i];
    }
  }
  out
}

/// Stratified sampling MC estimate.
///
/// Uses stratified normals instead of plain i.i.d. normals.
pub fn estimate<T, F>(n_paths: usize, dim: usize, payoff: F) -> McEstimate<T>
where
  T: FloatExt,
  F: Fn(&Array1<T>) -> T,
{
  let samples = stratified_normals::<T>(n_paths, dim);
  let mut sum = T::zero();
  let mut sum_sq = T::zero();

  for i in 0..n_paths {
    let z = samples.row(i).to_owned();
    let y = payoff(&z);
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

  /// Stratified normals should have mean ≈ 0 and variance ≈ 1.
  #[test]
  fn stratified_normals_have_correct_moments() {
    let n = 10_000;
    let z: Array1<f64> = stratified_normals_1d(n);
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

    let strat = estimate(n, dim, payoff);
    let expected = (0.5_f64).exp();

    assert!(
      (strat.mean - expected).abs() < 3.0 * strat.std_err + 0.02,
      "Stratified mean {:.4} far from expected {expected:.4}",
      strat.mean
    );
  }
}
