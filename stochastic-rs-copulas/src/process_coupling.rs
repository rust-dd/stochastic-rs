//! # Coupling stochastic processes via copulas
//!
//! Bridges [`crate::bivariate`] (Clayton, Frank, Gumbel, …) with the process
//! samplers in `stochastic-rs-stochastic`: given two univariate marginal
//! samples, transforms them through a fitted bivariate copula to produce a
//! joint sample with the desired tail dependence.
//!
//! The module deliberately does **not** depend on `stochastic-rs-stochastic`
//! (to keep the dependency graph acyclic and the copula crate lean). The
//! pattern is therefore expressed via free helpers operating on
//! `Array1<f64>` marginals.
//!
//! ## Pattern
//!
//! 1. Sample two marginal paths independently from any `ProcessExt` (e.g. two
//!    `Bm`, two `Heston` variance paths, two `Ou`).
//! 2. Transform each marginal to its empirical CDF on $[0, 1]$.
//! 3. Apply the bivariate copula's percent-point function to couple them.
//! 4. Invert through each marginal's empirical quantile to map back to the
//!    original scale.
//!
//! ## Example
//!
//! ```ignore
//! use ndarray::Array1;
//! use stochastic_rs_copulas::bivariate::clayton::Clayton;
//! use stochastic_rs_copulas::process_coupling::couple_marginals;
//! use stochastic_rs_copulas::traits::BivariateExt;
//!
//! let s1 = sample_bm(1024);
//! let s2 = sample_bm(1024);
//! let mut c = Clayton::new(Some(2.0), None);
//! let coupled = couple_marginals(&s1, &s2, &mut c).unwrap();
//! ```

use std::error::Error;

use ndarray::Array1;

use crate::traits::BivariateExt;

/// Couple two independent marginal samples through a bivariate copula.
///
/// Returns a `(n, 2)`-shaped sample with copula-induced dependence and
/// preserved empirical marginal distributions.
pub fn couple_marginals<C: BivariateExt>(
  s1: &Array1<f64>,
  s2: &Array1<f64>,
  copula: &mut C,
) -> Result<ndarray::Array2<f64>, Box<dyn Error>> {
  let n = s1.len();
  assert_eq!(n, s2.len(), "marginals must have equal length");
  assert!(n >= 2, "need at least two observations");

  let cop_sample = copula.sample(n)?;
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
