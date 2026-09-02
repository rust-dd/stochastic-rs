//! # RadialOU
//!
//! $$
//! dX_t=\left(\frac{\kappa}{X_t}-X_t\right)dt+\sigma\,dW_t
//! $$
//!
//! Reference: Göing-Jaeschke A., Yor M. (2003) — *A Survey and Some
//! Generalizations of Bessel Processes*, Bernoulli 9(2), 313–349,
//! DOI: 10.3150/bj/1068128980 — surveys the radial Ornstein-Uhlenbeck
//! process (the Euclidean norm of a multivariate OU process), of which
//! this is the scalar SDE with `κ` absorbing the dimension-dependent
//! constant.
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct RadialOU<T: FloatExt, S: SeedExt = Unseeded> {
  /// Radial restoring-force coefficient κ in the drift `κ/X_t − X_t`,
  /// which repels `X` away from 0 while an ordinary OU-style `-X_t` term
  /// still pulls it back at large `|X|`.
  pub kappa: T,
  /// Diffusion scale σ multiplying `dW_t`.
  pub sigma: T,
  /// Number of points sampled along the radial-OU path.
  pub n: usize,
  /// Initial value X₀ of the radial-OU path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> RadialOU<T, S> {
  pub fn new(kappa: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    Self {
      kappa,
      sigma,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for RadialOU<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = RadialOuSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> RadialOuSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    RadialOuSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      kappa: self.kappa,
      sigma: self.sigma,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`RadialOU`] sampling state: precomputed Euler step and the owned
/// Gaussian source.
#[doc(hidden)]
pub struct RadialOuSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  kappa: T,
  sigma: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> RadialOuSampler<T> {
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
      let safe_prev = if prev.abs() < T::from_f64_fast(1e-12) {
        T::from_f64_fast(1e-12)
      } else {
        prev
      };
      let next = prev + (self.kappa / safe_prev - prev) * self.dt + self.sigma * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for RadialOuSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("RadialOU output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyRadialOU, RadialOU,
  sig: (kappa, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (kappa: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
