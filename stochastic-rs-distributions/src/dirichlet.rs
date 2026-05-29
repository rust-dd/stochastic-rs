//! # Dirichlet distribution
//!
//! $$
//! f(x_1, \dots, x_K; \alpha_1, \dots, \alpha_K) = \frac{1}{\mathrm{B}(\alpha)}
//! \prod_{k=1}^{K} x_k^{\alpha_k - 1},
//! \qquad x_k \ge 0,\ \sum_k x_k = 1,
//! $$
//!
//! where the normaliser $\mathrm{B}(\alpha) = \prod_k \Gamma(\alpha_k) / \Gamma(\sum_k \alpha_k)$
//! is the multivariate Beta function.
//!
//! Used as the conjugate prior on the parameter vector of a Categorical /
//! Multinomial distribution, and as a flexible non-negative-weights prior
//! in Bayesian portfolio construction.
//!
//! ## Sampling
//!
//! Closed-form via the gamma trick: $Y_k \sim \mathrm{Gamma}(\alpha_k, 1)$,
//! then $X_k = Y_k / \sum_j Y_j$. The output is `Vec<f64>` so the
//! `DistributionExt` `pdf(f64) -> f64` signature does not apply; we
//! expose `log_pdf(&[f64]) -> f64` as a free method on the struct.

use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::gamma::SimdGamma;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;
use crate::traits::SimdFloatExt;

pub struct SimdDirichlet<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  alpha: Vec<T>,
  gammas: Vec<SimdGamma<T, R>>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdDirichlet<T, R> {
  pub fn new<S: SeedExt>(alpha: Vec<T>, seed: &S) -> Self {
    assert!(alpha.len() >= 2, "Dirichlet dim must be ≥ 2");
    for a in &alpha {
      assert!(*a > T::zero(), "α_k must be > 0");
    }
    let gammas = alpha
      .iter()
      .map(|&a| SimdGamma::<T, R>::new(a, T::one(), seed))
      .collect();
    Self { alpha, gammas }
  }

  pub fn dim(&self) -> usize {
    self.alpha.len()
  }

  /// Closed-form sample via independent Gamma marginals + simplex normalisation.
  pub fn sample_fast(&self) -> Vec<T> {
    let ys: Vec<T> = self.gammas.iter().map(|g| g.sample_fast()).collect();
    let sum: T = ys.iter().fold(T::zero(), |a, &b| a + b);
    let sum_safe = if sum > T::zero() {
      sum
    } else {
      T::from_f64_fast(1e-300)
    };
    ys.into_iter().map(|y| y / sum_safe).collect()
  }

  /// Log-density at point `x` (must lie on the open $K-1$-simplex).
  pub fn log_pdf(&self, x: &[T]) -> f64 {
    assert_eq!(x.len(), self.alpha.len(), "x and α must have the same dim");
    let alpha_f64: Vec<f64> = self.alpha.iter().map(|&a| a.to_f64().unwrap()).collect();
    let alpha_sum: f64 = alpha_f64.iter().sum();
    let log_norm = crate::special::ln_gamma(alpha_sum)
      - alpha_f64
        .iter()
        .map(|&a| crate::special::ln_gamma(a))
        .sum::<f64>();
    let log_kernel: f64 = x
      .iter()
      .zip(alpha_f64.iter())
      .map(|(&xi, &ak)| (ak - 1.0) * xi.to_f64().unwrap().max(1e-300).ln())
      .sum();
    log_norm + log_kernel
  }

  /// Density at point `x`. Returns 0 if `x` is outside the open simplex
  /// (negative component) but does NOT enforce $\sum x = 1$ — caller is
  /// responsible for projecting onto the simplex when needed.
  pub fn pdf(&self, x: &[T]) -> f64 {
    if x.iter().any(|&xi| xi < T::zero()) {
      return 0.0;
    }
    self.log_pdf(x).exp()
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdDirichlet<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.alpha.clone(), &Unseeded)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Samples lie on the simplex (sum to 1, all components non-negative).
  #[test]
  fn dirichlet_samples_on_simplex() {
    let d = SimdDirichlet::<f64>::new(vec![1.0, 2.0, 3.0], &Unseeded);
    for _ in 0..5_000 {
      let x = d.sample_fast();
      assert_eq!(x.len(), 3);
      let s: f64 = x.iter().sum();
      assert!((s - 1.0).abs() < 1e-10, "sum = {s}");
      for v in &x {
        assert!(*v >= 0.0);
      }
    }
  }

  /// Marginal expectations match $E[X_k] = \alpha_k / \sum_j \alpha_j$.
  #[test]
  fn dirichlet_marginal_means() {
    let alpha = vec![1.0, 2.0, 3.0];
    let alpha_sum: f64 = alpha.iter().sum();
    let expected: Vec<f64> = alpha.iter().map(|a| a / alpha_sum).collect();
    let d = SimdDirichlet::<f64>::new(alpha, &Unseeded);
    let n = 20_000;
    let mut sums = [0.0; 3];
    for _ in 0..n {
      let x = d.sample_fast();
      for k in 0..3 {
        sums[k] += x[k];
      }
    }
    let means: Vec<f64> = sums.iter().map(|s| s / n as f64).collect();
    for k in 0..3 {
      assert!(
        (means[k] - expected[k]).abs() < 0.02,
        "marginal {k}: mean = {}, expected ≈ {}",
        means[k],
        expected[k]
      );
    }
  }

  /// Symmetric Dirichlet (all α equal) gives uniform PDF on the simplex.
  #[test]
  fn dirichlet_symmetric_uniform_pdf() {
    // α = (1, 1, 1) → uniform on the 2-simplex; PDF = 2! = 2 everywhere.
    let d = SimdDirichlet::<f64>::new(vec![1.0, 1.0, 1.0], &Unseeded);
    let p1 = d.pdf(&[0.3, 0.4, 0.3]);
    let p2 = d.pdf(&[0.1, 0.1, 0.8]);
    let p3 = d.pdf(&[0.5, 0.25, 0.25]);
    assert!((p1 - 2.0).abs() < 1e-12);
    assert!((p2 - 2.0).abs() < 1e-12);
    assert!((p3 - 2.0).abs() < 1e-12);
  }
}
