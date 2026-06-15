//! # Lfsm
//!
//! $$
//! X_t=\int_{\mathbb{R}}\left(\max(t-u,0)^{H-1/\alpha}-\max(-u,0)^{H-1/\alpha}\right)\,dL_u^{\alpha}
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::alpha_stable::SimdAlphaStable;

use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Linear fractional stable motion (Lfsm), also commonly referred to as
/// Levy fractional stable motion in discretized form.
///
/// The implementation uses a one-sided moving-average discretization with
/// alpha-stable innovations:
///
/// `X_i = X_{i-1} + sum_{k=0}^{i-1} w_k * xi_{i-1-k}`,
/// where `w_k = dt^d * ((k+1)^d - k^d)` and `d = H - 1/alpha`.
pub struct Lfsm<T: FloatExt, S: SeedExt = Unseeded> {
  /// Stability index of the Levy-stable driver (`0 < alpha <= 2`).
  /// Smaller values produce heavier tails and larger jumps.
  pub alpha: T,
  /// Skewness of the stable innovations (`-1 <= beta <= 1`).
  /// Controls left/right jump asymmetry.
  pub beta: T,
  /// Self-similarity / roughness parameter (`1/alpha < H < 1` here).
  /// Governs long-memory strength in the fractional kernel.
  pub hurst: T,
  /// Scale of the stable noise term.
  /// Larger values increase path variability.
  pub scale: T,
  /// Number of grid points along the path.
  pub n: usize,
  /// Initial process value.
  pub x0: Option<T>,
  /// Total simulated time horizon (defaults to `1` if `None`).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Lfsm<T, S> {
  pub fn new(
    alpha: T,
    beta: T,
    hurst: T,
    scale: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(alpha > T::zero() && alpha <= T::from(2.0).unwrap());
    assert!((-T::one()..=T::one()).contains(&beta));
    assert!(scale > T::zero());
    assert!(
      hurst > T::one() / alpha && hurst < T::one(),
      "Lfsm requires 1/alpha < hurst < 1 for this discretization"
    );
    Self {
      alpha,
      beta,
      hurst,
      scale,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> Lfsm<T, S> {
  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Lfsm<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = LfsmSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> LfsmSampler<T> {
    let dt = self.dt();
    let d = self.hurst - T::one() / self.alpha;
    let kernel_scale = dt.powf(d);
    let innovation_scale = self.scale * dt.powf(T::one() / self.alpha);

    let stable = SimdAlphaStable::<T>::new(
      self.alpha,
      self.beta,
      innovation_scale,
      T::zero(),
      &self.seed,
    );

    // Moving-average weights `w_k = dt^d ((k+1)^d - k^d)` are deterministic,
    // so they are computed once and reused across samples.
    let weight_len = self.n.saturating_sub(1);
    let mut weights = Array1::<T>::zeros(weight_len);
    for k in 0..weight_len {
      let kf = T::from_usize_(k);
      weights[k] = kernel_scale * ((kf + T::one()).powf(d) - kf.powf(d));
    }

    LfsmSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      stable,
      weights,
    }
  }
}

/// Reusable [`Lfsm`] sampling state: the owned alpha-stable innovation source
/// and the precomputed fractional moving-average weights.
#[doc(hidden)]
pub struct LfsmSampler<T: FloatExt> {
  n: usize,
  x0: T,
  stable: SimdAlphaStable<T>,
  weights: Array1<T>,
}

impl<T: FloatExt> LfsmSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    // Legacy behaviour: the degenerate `n <= 1` path returns zeros without
    // writing `x0` (the `if n <= 1` guard preceded the `x[0] = x0` line).
    if out.len() == 1 {
      out[0] = T::zero();
      return;
    }
    out[0] = self.x0;

    let mut innovations = Array1::<T>::zeros(out.len() - 1);
    self
      .stable
      .fill_slice_fast(innovations.as_slice_mut().unwrap());

    let mut prev = self.x0;
    for i in 1..out.len() {
      let mut inc = T::zero();
      for k in 0..i {
        inc += self.weights[k] * innovations[i - 1 - k];
      }
      prev += inc;
      out[i] = prev;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for LfsmSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Lfsm output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyLfsm, Lfsm,
  sig: (alpha, beta, hurst, scale, n, x0=None, t=None, seed=None, dtype=None),
  params: (alpha: f64, beta: f64, hurst: f64, scale: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn lfsm_path_is_finite() {
    let p = Lfsm::new(1.7_f64, 0.0, 0.8, 1.0, 256, Some(0.0), Some(1.0), Unseeded);
    let x = p.sample();
    assert_eq!(x.len(), 256);
    assert!(x.iter().all(|v| v.is_finite()));
  }
}
