//! # Monte Carlo Methods
//!
//! $$
//! \hat{\mu}_N = \frac{1}{N}\sum_{i=1}^{N} f(X_i),\quad
//! \operatorname{Var}[\hat{\mu}_N] = \frac{\sigma^2}{N}
//! $$
//!
//! Variance reduction, quasi-Monte Carlo sequences, multi-level MC, and
//! American option pricing via Longstaff-Schwartz.
//!
//! Reference: Glasserman (2003), *Monte Carlo Methods in Financial Engineering*,
//! DOI: 10.1007/978-0-387-21617-1

pub mod antithetic;
pub mod control_variates;
pub mod halton;
pub mod importance_sampling;
#[cfg(feature = "openblas")]
pub mod lsm;
pub mod mlmc;
pub mod sobol;
pub mod stratified;

use crate::traits::FloatExt;

/// Result of a Monte Carlo estimation.
#[derive(Debug, Clone)]
pub struct McEstimate<T: FloatExt> {
  /// Estimated mean.
  pub mean: T,
  /// Standard error of the estimate.
  pub std_err: T,
  /// Number of samples used.
  pub n_samples: usize,
}

impl<T: FloatExt> McEstimate<T> {
  /// Symmetric confidence interval `[mean ± z · std_err]`.
  pub fn confidence_interval(&self, z: T) -> (T, T) {
    (self.mean - z * self.std_err, self.mean + z * self.std_err)
  }

  /// 95% confidence interval (z = 1.96).
  pub fn ci_95(&self) -> (T, T) {
    self.confidence_interval(T::from_f64_fast(1.96))
  }
}
