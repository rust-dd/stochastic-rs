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

/// Couples two marginal samples through a bivariate copula drawn from `seed`: an `(n, 2)` sample that keeps both empirical
/// marginals; `DimensionMismatch` for unequal lengths, `InsufficientData` below two values, `MarginalOutOfRange` for a NaN.
pub fn couple_marginals<C: BivariateExt, S: SeedExt>(
  s1: ArrayView1<f64>,
  s2: ArrayView1<f64>,
  copula: &C,
  seed: &S,
) -> Result<ndarray::Array2<f64>, CopulaError> {
  let n = s1.len();
  if s2.len() != n {
    return Err(CopulaError::DimensionMismatch {
      expected: n,
      got: s2.len(),
    });
  }
  if n < 2 {
    return Err(CopulaError::InsufficientData { needed: 2, got: n });
  }
  if s1.iter().chain(s2).any(|x| x.is_nan()) {
    return Err(CopulaError::MarginalOutOfRange);
  }

  let cop_sample = copula.sample(n, seed)?;
  let mut out = ndarray::Array2::<f64>::zeros((n, 2));

  let mut sorted1 = s1.to_vec();
  sorted1.sort_by(f64::total_cmp);
  let mut sorted2 = s2.to_vec();
  sorted2.sort_by(f64::total_cmp);

  for i in 0..n {
    let (u, v) = (cop_sample[[i, 0]], cop_sample[[i, 1]]);
    let idx_u = ((u * n as f64) as usize).min(n - 1);
    let idx_v = ((v * n as f64) as usize).min(n - 1);
    out[[i, 0]] = sorted1[idx_u];
    out[[i, 1]] = sorted2[idx_v];
  }

  Ok(out)
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::bivariate::clayton::Clayton;

  /// Data errors are answered, not panicked on: unequal lengths, a single value, a NaN in either series.
  #[test]
  fn data_errors_are_answered() {
    let (c, seed) = (Clayton::new(2.0), Deterministic::new(1));
    let couple = |a: ndarray::Array1<f64>, b: ndarray::Array1<f64>| {
      couple_marginals(a.view(), b.view(), &c, &seed)
    };
    let mismatch = CopulaError::DimensionMismatch {
      expected: 3,
      got: 2,
    };
    assert_eq!(
      couple(array![1.0, 2.0, 3.0], array![1.0, 2.0]),
      Err(mismatch)
    );
    let short = CopulaError::InsufficientData { needed: 2, got: 1 };
    assert_eq!(couple(array![1.0], array![1.0]), Err(short));
    for (a, b) in [
      (array![1.0, f64::NAN], array![1.0, 2.0]),
      (array![1.0, 2.0], array![f64::NAN, 2.0]),
    ] {
      assert_eq!(couple(a, b), Err(CopulaError::MarginalOutOfRange));
    }
  }
}
