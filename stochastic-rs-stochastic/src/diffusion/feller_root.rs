//! # FellerRoot
//!
//! $$
//! dX_t=X_t(\theta_1 - X_t(\theta_3^3 - \theta_1\theta_2))\,dt+\theta_3 X_t^{3/2}\,dW_t
//! $$
//!
use ndarray::Array1;
use ndarray::s;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
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

impl<T: FloatExt> FellerRoot<T> {
  pub fn new(theta1: T, theta2: T, theta3: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      theta1,
      theta2,
      theta3,
      n,
      x0,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> FellerRoot<T, Deterministic> {
  pub fn seeded(
    theta1: T,
    theta2: T,
    theta3: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: u64,
  ) -> Self {
    Self {
      theta1,
      theta2,
      theta3,
      n,
      x0,
      t,
      seed: Deterministic::new(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for FellerRoot<T, S> {
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
      .expect("FellerRoot output tail must be contiguous");
    let normal = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &self.seed);
    normal.fill_slice_fast(tail);

    for z in tail.iter_mut() {
      let drift = prev * (self.theta1 - prev * (self.theta3.powi(3) - self.theta1 * self.theta2));
      let next = prev + drift * dt + self.theta3 * prev.abs().powf(T::from_f64_fast(1.5)) * *z;
      *z = next;
      prev = next;
    }

    x
  }
}

py_process_1d!(PyFellerRoot, FellerRoot,
  sig: (theta1, theta2, theta3, n, x0=None, t=None, seed=None, dtype=None),
  params: (theta1: f64, theta2: f64, theta3: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
