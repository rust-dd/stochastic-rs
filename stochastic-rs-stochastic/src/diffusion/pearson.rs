//! # Pearson
//!
//! $$
//! dX_t=\kappa(\mu-X_t)\,dt+\sqrt{2\kappa(aX_t^2+bX_t+c)}\,dW_t
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
pub struct Pearson<T: FloatExt, S: SeedExt = Unseeded> {
  pub kappa: T,
  pub mu: T,
  pub a: T,
  pub b: T,
  pub c: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Pearson<T, S> {
  pub fn new(
    kappa: T,
    mu: T,
    a: T,
    b: T,
    c: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self {
      kappa,
      mu,
      a,
      b,
      c,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Pearson<T, S> {
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
      .expect("Pearson output tail must be contiguous");
    let normal = SimdNormal::<T>::new(T::zero(), sqrt_dt, &self.seed);
    normal.fill_slice_fast(tail);

    for z in tail.iter_mut() {
      let diff_inner =
        T::from_f64_fast(2.0) * self.kappa * (self.a * prev * prev + self.b * prev + self.c);
      let next = prev + self.kappa * (self.mu - prev) * dt + diff_inner.abs().sqrt() * *z;
      *z = next;
      prev = next;
    }

    x
  }
}

py_process_1d!(PyPearson, Pearson,
  sig: (kappa, mu, a, b, c, n, x0=None, t=None, seed=None, dtype=None),
  params: (kappa: f64, mu: f64, a: f64, b: f64, c: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
