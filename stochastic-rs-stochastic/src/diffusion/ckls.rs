//! # Ckls
//!
//! $$
//! dX_t=(\theta_1+\theta_2 X_t)\,dt+\theta_3 X_t^{\theta_4}\,dW_t
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

pub struct Ckls<T: FloatExt, S: SeedExt = Unseeded> {
  /// Drift intercept parameter.
  pub theta1: T,
  /// Drift slope parameter.
  pub theta2: T,
  /// Diffusion scale parameter.
  pub theta3: T,
  /// Diffusion elasticity parameter.
  pub theta4: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Ckls<T, S> {
  pub fn new(
    theta1: T,
    theta2: T,
    theta3: T,
    theta4: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self {
      theta1,
      theta2,
      theta3,
      theta4,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Ckls<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = CklsSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> CklsSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    CklsSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      theta1: self.theta1,
      theta2: self.theta2,
      theta3: self.theta3,
      theta4: self.theta4,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`Ckls`] sampling state.
#[doc(hidden)]
pub struct CklsSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  theta1: T,
  theta2: T,
  theta3: T,
  theta4: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> CklsSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.normal.fill_slice_fast(tail);
    let mut prev = self.x0;
    for z in tail.iter_mut() {
      let next = prev
        + (self.theta1 + self.theta2 * prev) * self.dt
        + self.theta3 * prev.abs().powf(self.theta4) * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for CklsSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Ckls output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyCkls, Ckls,
  sig: (theta1, theta2, theta3, theta4, n, x0=None, t=None, seed=None, dtype=None),
  params: (theta1: f64, theta2: f64, theta3: f64, theta4: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
