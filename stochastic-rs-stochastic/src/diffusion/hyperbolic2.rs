//! # Hyperbolic2
//!
//! $$
//! dX_t=\frac{\sigma^2}{2}\left(\beta-\frac{\gamma X_t}{\sqrt{\delta^2+(X_t-\mu)^2}}\right)dt+\sigma\,dW_t
//! $$
//!
use ndarray::Array1;
use ndarray::s;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
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

  fn sample(&self) -> Self::Output {
    let mut x = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return x;
    }

    x[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return x;
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    let mut prev = x[0];
    let mut tail_view = x.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("Hyperbolic2 output tail must be contiguous");
    let normal = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &self.seed);
    normal.fill_slice_fast(tail);

    for z in tail.iter_mut() {
      let r = (self.delta * self.delta + (prev - self.mu) * (prev - self.mu)).sqrt();
      let half = T::from_f64_fast(0.5);
      let drift = half * self.sigma * self.sigma * (self.beta - self.gamma * prev / r);
      let next = prev + drift * dt + self.sigma * *z;
      *z = next;
      prev = next;
    }

    x
  }
}

py_process_1d!(PyHyperbolic2, Hyperbolic2,
  sig: (beta, gamma, delta, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (beta: f64, gamma: f64, delta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
