//! # NonLinearSDE
//!
//! $$
//! dX_t=\left(\frac{a_{-1}}{X_t}+a_0+a_1 X_t+a_2 X_t^2\right)dt+(b_0+b_1 X_t+b_2 X_t^{b_3})\,dW_t
//! $$
//!
use ndarray::Array1;
use ndarray::s;

use crate::distributions::normal::SimdNormal;
use crate::simd_rng::Deterministic;
use crate::simd_rng::SeedExt;
use crate::simd_rng::Unseeded;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

#[derive(Clone, Copy)]
pub struct NonLinearSDE<T: FloatExt, S: SeedExt = Unseeded> {
  pub am1: T,
  pub a0: T,
  pub a1: T,
  pub a2: T,
  pub b0: T,
  pub b1: T,
  pub b2: T,
  pub b3: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub seed: S,
}

impl<T: FloatExt> NonLinearSDE<T> {
  pub fn new(
    am1: T,
    a0: T,
    a1: T,
    a2: T,
    b0: T,
    b1: T,
    b2: T,
    b3: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
  ) -> Self {
    Self {
      am1,
      a0,
      a1,
      a2,
      b0,
      b1,
      b2,
      b3,
      n,
      x0,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> NonLinearSDE<T, Deterministic> {
  pub fn seeded(
    am1: T,
    a0: T,
    a1: T,
    a2: T,
    b0: T,
    b1: T,
    b2: T,
    b3: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: u64,
  ) -> Self {
    Self {
      am1,
      a0,
      a1,
      a2,
      b0,
      b1,
      b2,
      b3,
      n,
      x0,
      t,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for NonLinearSDE<T, S> {
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
      .expect("NonLinearSDE output tail must be contiguous");
    let mut seed = self.seed;
    let normal = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &mut seed);
    normal.fill_slice_fast(tail);

    for z in tail.iter_mut() {
      let safe_prev = if prev.abs() < T::from_f64_fast(1e-12) {
        T::from_f64_fast(1e-12)
      } else {
        prev
      };
      let drift = self.am1 / safe_prev + self.a0 + self.a1 * prev + self.a2 * prev * prev;
      let diff = self.b0 + self.b1 * prev + self.b2 * prev.abs().powf(self.b3);
      let next = prev + drift * dt + diff * *z;
      *z = next;
      prev = next;
    }

    x
  }
}

py_process_1d!(PyNonLinearSDE, NonLinearSDE,
  sig: (am1, a0, a1, a2, b0, b1, b2, b3, n, x0=None, t=None, seed=None, dtype=None),
  params: (am1: f64, a0: f64, a1: f64, a2: f64, b0: f64, b1: f64, b2: f64, b3: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
