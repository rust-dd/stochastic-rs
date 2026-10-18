//! # Coupling stochastic processes via copulas
//!
//! Maps copula draws back through each series' empirical quantile, so the coupled sample keeps both
//! marginals; free helpers over `ArrayView1<f64>` keep this crate off `stochastic-rs-stochastic`.
//!
//! ## Example
//!
//! ```
//! use ndarray::Array1;
//! use stochastic_rs_copulas::bivariate::clayton::Clayton;
//! use stochastic_rs_copulas::process_coupling::couple_marginals;
//! use stochastic_rs_core::simd_rng::Deterministic;
//!
//! // Marginals from any source work the same (e.g. two `ProcessExt::sample()` paths); fixed sequences keep the example
//! // free of a stochastic dependency.
//! let s1 = Array1::linspace(0.0, 1.0, 32);
//! let s2 = Array1::linspace(1.0, 0.0, 32);
//! let c = Clayton::new(2.0);
//! let coupled = couple_marginals(s1.view(), s2.view(), &c, &Deterministic::new(1)).unwrap();
//! assert_eq!(coupled.shape(), &[32, 2]);
//! ```

use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::SeedExt;

use crate::error::CopulaError;
use crate::traits::BivariateExt;

/// Couples two marginal samples through a bivariate copula drawn from `seed`: an `(n, 2)` sample that keeps both empirical marginals.
/// Panics unless both series have the same length of at least 2.
pub fn couple_marginals<C: BivariateExt, S: SeedExt>(
  s1: ArrayView1<f64>,
  s2: ArrayView1<f64>,
  copula: &C,
  seed: &S,
) -> Result<ndarray::Array2<f64>, CopulaError> {
  let n = s1.len();
  assert!(
    s2.len() == n,
    "s2 must satisfy `s2.len() == s1.len()`, got s2.len() = {}",
    s2.len()
  );
  assert!(
    n >= 2,
    "s1 must satisfy `s1.len() >= 2`, got s1.len() = {n}"
  );

  let cop_sample = copula.sample(n, seed)?;
  let mut out = ndarray::Array2::<f64>::zeros((n, 2));

  let mut sorted1 = s1.to_vec();
  sorted1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
  let mut sorted2 = s2.to_vec();
  sorted2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

  for i in 0..n {
    let u = cop_sample[[i, 0]].clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    let v = cop_sample[[i, 1]].clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    let idx_u = ((u * n as f64) as usize).min(n - 1);
    let idx_v = ((v * n as f64) as usize).min(n - 1);
    out[[i, 0]] = sorted1[idx_u];
    out[[i, 1]] = sorted2[idx_v];
  }

  Ok(out)
}
