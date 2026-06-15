//! # FellerRoot
//!
//! $$
//! dX_t=X_t(\theta_1 - X_t(\theta_3^3 - \theta_1\theta_2))\,dt+\theta_3 X_t^{3/2}\,dW_t
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

#[derive(Clone, Copy)]
pub struct FellerRoot<T: FloatExt, S: SeedExt = Unseeded> {
  pub theta1: T,
  pub theta2: T,
  pub theta3: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> FellerRoot<T, S> {
  pub fn new(
    theta1: T,
    theta2: T,
    theta3: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self {
      theta1,
      theta2,
      theta3,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for FellerRoot<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = FellerRootSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> FellerRootSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    FellerRootSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      theta1: self.theta1,
      theta2: self.theta2,
      theta3: self.theta3,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`FellerRoot`] sampling state.
#[doc(hidden)]
pub struct FellerRootSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  theta1: T,
  theta2: T,
  theta3: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> FellerRootSampler<T> {
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
      let drift = prev * (self.theta1 - prev * (self.theta3.powi(3) - self.theta1 * self.theta2));
      let next = prev + drift * self.dt + self.theta3 * prev.abs().powf(T::from_f64_fast(1.5)) * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for FellerRootSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("FellerRoot output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyFellerRoot, FellerRoot,
  sig: (theta1, theta2, theta3, n, x0=None, t=None, seed=None, dtype=None),
  params: (theta1: f64, theta2: f64, theta3: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
