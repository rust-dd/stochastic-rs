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
//!
//! Reference: Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §XI.4, Theorem 4.1, DOI 10.1007/978-1-4613-8643-8.

use ndarray::Array2;
use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use crate::gamma::GammaState;
use crate::gamma::SimdGamma;
use crate::seeded::Seeded;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// Dirichlet law with concentrations `alpha`: parameters only; a [`Seeded`] stream draws its simplex points.
#[derive(Clone, Debug, PartialEq)]
pub struct SimdDirichlet<T> {
  alpha: Vec<T>,
  gammas: Vec<SimdGamma<T>>,
}

impl<T: SimdFloatExt> SimdDirichlet<T> {
  /// Creates a Dirichlet distribution over the `K = alpha.len()`-simplex.
  ///
  /// - `alpha` — concentration vector α₁..α_K (matches the module
  ///   header's α), each entry > 0; K must be ≥ 2.
  pub fn new(alpha: Vec<T>) -> Self {
    assert!(
      alpha.len() >= 2,
      "alpha must satisfy `alpha.len() >= 2`, got alpha.len() = {}",
      alpha.len()
    );
    for (k, a) in alpha.iter().enumerate() {
      assert!(
        *a > T::zero(),
        "alpha must satisfy `alpha[k] > T::zero()`, got alpha[{k}] = {a:?}"
      );
    }
    let gammas = alpha
      .iter()
      .map(|&a| SimdGamma::<T>::new(a, T::one()))
      .collect::<Vec<_>>();
    Self { alpha, gammas }
  }

  /// The concentrations `α₁..α_K`.
  pub fn alpha(&self) -> &[T] {
    &self.alpha
  }

  pub fn dim(&self) -> usize {
    self.alpha.len()
  }

  /// Divides `out` by its sum; `false`, leaving `out` as it is, when every coordinate underflowed to zero.
  #[inline]
  fn normalize(out: &mut [T]) -> bool {
    let mut sum = T::zero();
    for x in out.iter() {
      sum += *x;
    }
    if sum > T::zero() {
      for x in out.iter_mut() {
        *x = *x / sum;
      }
      return true;
    }
    false
  }

  /// The simplex point of one log draw per coordinate in `out`, for the `0/0` sum of all-underflowed gammas: at
  /// `α = 0.001` two thirds of single-precision draws, where a `1e-300` guard is itself zero.
  fn normalize_logs(out: &mut [T]) {
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

  #[cold]
  #[inline(never)]
  fn simplex_from_logs<R: SimdRngExt>(&self, states: &mut [GammaState<T, R>], out: &mut [T]) {
    for ((x, g), st) in out.iter_mut().zip(&self.gammas).zip(states.iter_mut()) {
      *x = g.next_log(st);
    }
    Self::normalize_logs(out);
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

impl<T: SimdFloatExt> Sealed for SimdDirichlet<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdDirichlet<T> {
  type State<R: SimdRngExt> = Vec<GammaState<T, R>>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (Vec<GammaState<T, R>>, u64) {
    let mut states = Vec::with_capacity(self.gammas.len());
    let mut basis = 0;
    for g in &self.gammas {
      let (state, stream_seed) = g.init::<R, S>(seed);
      if states.is_empty() {
        basis = stream_seed;
      }
      states.push(state);
    }
    (states, basis)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Seeded<SimdDirichlet<T>, R> {
  /// Writes one simplex point into `out`, whose length must be the law's `dim()`; nothing is allocated.
  pub fn sample_into(&mut self, out: &mut [T]) {
    let (law, states) = self.parts_mut();
    assert_eq!(
      out.len(),
      law.alpha.len(),
      "out and α must have the same dim"
    );
    for ((x, g), st) in out.iter_mut().zip(&law.gammas).zip(states.iter_mut()) {
      *x = g.next(st);
    }
    if !SimdDirichlet::normalize(out) {
      law.simplex_from_logs(states, out);
    }
  }

  /// One simplex point.
  pub fn sample(&mut self) -> Vec<T> {
    let mut out = vec![T::zero(); self.dist().dim()];
    self.sample_into(&mut out);
    out
  }

  /// `m` simplex points, one per row, in draw order.
  pub fn sample_matrix(&mut self, m: usize) -> Array2<T> {
    let mut out = Array2::<T>::zeros((m, self.dist().dim()));
    for mut row in out.rows_mut() {
      self.sample_into(
        row
          .as_slice_mut()
          .expect("a standard-layout row is contiguous"),
      );
    }
    out
  }
}

impl<T: SimdFloatExt> Distribution<Vec<T>> for SimdDirichlet<T> {
  /// Normalised scalar gamma draws on the caller's rng, with the same log fallback as the stream.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> Vec<T> {
    let mut out = self
      .gammas
      .iter()
      .map(|g| g.draw_with(rng))
      .collect::<Vec<_>>();
    if !Self::normalize(&mut out) {
      for (x, g) in out.iter_mut().zip(&self.gammas) {
        *x = g.draw_log_with(rng);
      }
      Self::normalize_logs(&mut out);
    }
    out
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_core::simd_rng::SimdRng;
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::*;
  use crate::tests::assert_mean_within;

  fn assert_small_concentrations_on_the_simplex(mut draw: impl FnMut(&mut [f32])) {
    let alpha = [0.001_f32, 0.002, 0.003, 0.004];
    let total: f32 = alpha.iter().sum();
    let mut out = [0.0_f32; 4];
    let mut acc = [0.0_f64; 4];
    let n = 100_000;
    for _ in 0..n {
      draw(&mut out);
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

  /// Small concentrations must not lose the vector to `0 / 0` (two thirds of the `f32` draws at `α = 0.001`); the mean
  /// check tells a real repair from one that returns a fixed point of the simplex.
  #[test]
  fn small_concentrations_stay_on_the_simplex() {
    let mut s =
      SimdDirichlet::<f32>::new(vec![0.001, 0.002, 0.003, 0.004]).seeded(&Deterministic::new(151));
    assert_small_concentrations_on_the_simplex(|out| s.sample_into(out));
  }

  /// The honest draw takes the same log fallback on the caller's rng.
  #[test]
  fn scalar_small_concentrations_stay_on_the_simplex() {
    let d = SimdDirichlet::<f32>::new(vec![0.001, 0.002, 0.003, 0.004]);
    let mut rng = SimdRng::from_seed(151);
    assert_small_concentrations_on_the_simplex(|out| out.copy_from_slice(&d.sample(&mut rng)));
  }

  /// Samples lie on the simplex (sum to 1, all components non-negative).
  #[test]
  fn dirichlet_samples_on_simplex() {
    let mut d = SimdDirichlet::<f64>::new(vec![1.0, 2.0, 3.0]).seeded(&Unseeded);
    for _ in 0..5_000 {
      let x = d.sample();
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
    let mut d = SimdDirichlet::<f64>::new(alpha).seeded(&Unseeded);
    let n = 20_000;
    let mut sums = [0.0; 3];
    for _ in 0..n {
      let x = d.sample();
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

  /// The honest draw's marginal means are `α_k / Σα` within five standard errors.
  #[test]
  fn scalar_sample_marginal_means() {
    let d = SimdDirichlet::<f64>::new(vec![1.0, 2.0, 3.0]);
    let mut rng = SimdRng::from_seed(2718);
    let draws = (0..100_000).map(|_| d.sample(&mut rng)).collect::<Vec<_>>();
    for (k, want) in [1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0].into_iter().enumerate() {
      let coordinate = draws.iter().map(|x| x[k]).collect::<Vec<_>>();
      assert_mean_within(&coordinate, want, 5.0, &format!("coordinate {k}"));
    }
  }

  /// Symmetric Dirichlet (all α equal) gives uniform PDF on the simplex.
  #[test]
  fn dirichlet_symmetric_uniform_pdf() {
    // α = (1, 1, 1) → uniform on the 2-simplex; PDF = 2! = 2 everywhere.
    let d = SimdDirichlet::<f64>::new(vec![1.0, 1.0, 1.0]);
    let p1 = d.pdf(&[0.3, 0.4, 0.3]);
    let p2 = d.pdf(&[0.1, 0.1, 0.8]);
    let p3 = d.pdf(&[0.5, 0.25, 0.25]);
    assert!((p1 - 2.0).abs() < 1e-12);
    assert!((p2 - 2.0).abs() < 1e-12);
    assert!((p3 - 2.0).abs() < 1e-12);
  }
}
