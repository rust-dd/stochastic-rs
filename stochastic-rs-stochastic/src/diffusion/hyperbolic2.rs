//! # Hyperbolic2
//!
//! $$
//! dX_t=\frac{\sigma^2}{2}\left(\beta-\frac{\gamma X_t}{\sqrt{\delta^2+(X_t-\mu)^2}}\right)dt+\sigma\,dW_t
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
pub struct Hyperbolic2<T: FloatExt, S: SeedExt = Unseeded> {
  pub beta: T,
  pub gamma: T,
  pub delta: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Hyperbolic2<T, S> {
  pub fn new(
    beta: T,
    gamma: T,
    delta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self {
      beta,
      gamma,
      delta,
      mu,
      sigma,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Hyperbolic2<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = Hyperbolic2Sampler<T>
  where
    Self: 's;

  fn sampler(&self) -> Hyperbolic2Sampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    Hyperbolic2Sampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      beta: self.beta,
      gamma: self.gamma,
      delta: self.delta,
      mu: self.mu,
      sigma: self.sigma,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`Hyperbolic2`] sampling state.
#[doc(hidden)]
pub struct Hyperbolic2Sampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  beta: T,
  gamma: T,
  delta: T,
  mu: T,
  sigma: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> Hyperbolic2Sampler<T> {
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
      let r = (self.delta * self.delta + (prev - self.mu) * (prev - self.mu)).sqrt();
      let half = T::from_f64_fast(0.5);
      let drift = half * self.sigma * self.sigma * (self.beta - self.gamma * prev / r);
      let next = prev + drift * self.dt + self.sigma * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for Hyperbolic2Sampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Hyperbolic2 output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyHyperbolic2, Hyperbolic2,
  sig: (beta, gamma, delta, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (beta: f64, gamma: f64, delta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
