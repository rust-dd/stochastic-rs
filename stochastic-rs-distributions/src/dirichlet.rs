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
  /// Creates a Dirichlet distribution over the `K = alpha.len()`-simplex.
  ///
  /// - `alpha` — concentration vector α₁..α_K (matches the module
  ///   header's α), each entry > 0; K must be ≥ 2.
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

  /// Closed-form sample via independent Gamma marginals + simplex
  /// normalisation, written into `out` (length must equal `dim`) without
  /// any per-draw allocation.
  pub fn sample_into(&self, out: &mut [T]) {
    assert_eq!(
      out.len(),
      self.alpha.len(),
      "out and α must have the same dim"
    );
    let mut sum = T::zero();
    for (x, g) in out.iter_mut().zip(self.gammas.iter()) {
      *x = g.sample_fast();
      sum += *x;
    }
    if sum > T::zero() {
      for x in out.iter_mut() {
        *x = *x / sum;
      }
      return;
    }
    self.simplex_from_logs(out);
  }

  /// The simplex point recovered from log draws, for the case where every
  /// Gamma marginal underflowed to exactly zero and the normalisation is
  /// `0/0`.
  ///
  /// Small concentrations make that ordinary: at `α = 0.001` two thirds of
  /// a single-precision Dirichlet's draws lose every coordinate this way.
  /// The old guard substituted `1e-300` for the vanished sum, which is
  /// itself zero once `T` is `f32` — the division stayed `0/0` and the
  /// whole vector came back NaN.
  #[cold]
  #[inline(never)]
  fn simplex_from_logs(&self, out: &mut [T]) {
    for (x, g) in out.iter_mut().zip(self.gammas.iter()) {
      *x = g.sample_log_fast();
    }
    let peak = out
      .iter()
      .copied()
      .fold(T::neg_infinity(), |m: T, x| if x > m { x } else { m });
    let mut sum = T::zero();
    for x in out.iter_mut() {
      *x = (*x - peak).exp();
      sum += *x;
    }
    for x in out.iter_mut() {
      *x = *x / sum;
    }
  }

  /// Allocating convenience wrapper around [`sample_into`](Self::sample_into).
  pub fn sample_fast(&self) -> Vec<T> {
    let mut out = vec![T::zero(); self.alpha.len()];
    self.sample_into(&mut out);
    out
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
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// Small concentrations must not lose the whole vector to `0 / 0`.
  ///
  /// Every Gamma marginal underflows to exactly zero two thirds of the time
  /// at `α = 0.001` in single precision, and the old guard substituted
  /// `1e-300` for the vanished sum — itself zero once `T` is `f32`, so the
  /// division stayed `0 / 0` and the vector came back all NaN. The mean
  /// check is what separates a real repair from one that merely returns a
  /// fixed point of the simplex.
  #[test]
  fn small_concentrations_stay_on_the_simplex() {
    let alpha = [0.001_f32, 0.002, 0.003, 0.004];
    let total: f32 = alpha.iter().sum();
    let d = SimdDirichlet::<f32>::new(alpha.to_vec(), &Deterministic::new(151));
    let mut out = [0.0_f32; 4];
    let mut acc = [0.0_f64; 4];
    let n = 100_000;
    for _ in 0..n {
      d.sample_into(&mut out);
      let s: f32 = out.iter().sum();
      assert!(
        out.iter().all(|x| x.is_finite() && *x >= 0.0) && (s - 1.0).abs() < 1e-3,
        "draw {out:?} is not on the simplex"
      );
      for k in 0..4 {
        acc[k] += out[k] as f64;
      }
    }
    for k in 0..4 {
      let mean = acc[k] / n as f64;
      let want = (alpha[k] / total) as f64;
      // Each coordinate is all but Bernoulli(want) at this concentration,
      // so five standard errors is 2.5/√n.
      assert!(
        (mean - want).abs() < 2.5 / (n as f64).sqrt(),
        "coordinate {k}: mean = {mean}, expected {want}"
      );
    }
  }

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
