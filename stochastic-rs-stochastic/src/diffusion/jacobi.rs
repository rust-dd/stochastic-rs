//! # Jacobi
//!
//! $$
//! dX_t=\kappa(\theta-X_t)dt+\sigma\sqrt{X_t(1-X_t)}\,dW_t
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct Jacobi<T: FloatExt, S: SeedExt = Unseeded> {
  /// Linear-drift intercept (κθ combined) in the reparametrized drift
  /// `alpha - beta·X`, equivalent to `κ(θ-X)` with `alpha = κθ`. Must be
  /// less than `beta` so the implied θ = alpha/beta stays in (0, 1), the
  /// Jacobi boundary requirement.
  pub alpha: T,
  /// Linear-drift slope (mean-reversion speed κ) in `alpha - beta·X`.
  pub beta: T,
  /// Diffusion scale σ multiplying `√(X_t(1-X_t)) dW_t`.
  pub sigma: T,
  /// Number of points sampled along the Jacobi path.
  pub n: usize,
  /// Initial value X₀ of the Jacobi path (clamped into [0, 1]).
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Jacobi<T, S> {
  pub fn new(alpha: T, beta: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    assert!(alpha > T::zero(), "alpha must be positive");
    assert!(beta > T::zero(), "beta must be positive");
    assert!(sigma > T::zero(), "sigma must be positive");
    assert!(alpha < beta, "alpha must be less than beta");

    Self {
      alpha,
      beta,
      sigma,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Jacobi<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = JacobiSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> JacobiSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    JacobiSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      alpha: self.alpha,
      beta: self.beta,
      diff_scale: self.sigma,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`Jacobi`] sampling state.
#[doc(hidden)]
pub struct JacobiSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  alpha: T,
  beta: T,
  diff_scale: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> JacobiSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.normal.fill_slice(tail);
    let mut prev = self.x0;
    for z in tail.iter_mut() {
      let next = match prev {
        _ if prev <= T::zero() => T::zero(),
        _ if prev >= T::one() => T::one(),
        _ => {
          prev
            + (self.alpha - self.beta * prev) * self.dt
            + self.diff_scale * (prev * (T::one() - prev)).sqrt() * *z
        }
      };
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for JacobiSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Jacobi output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyJacobi, Jacobi,
  sig: (alpha, beta, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (alpha: f64, beta: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
