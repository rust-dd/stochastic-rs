//! # Ckls
//!
//! $$
//! dX_t=(\theta_1+\theta_2 X_t)\,dt+\theta_3 X_t^{\theta_4}\,dW_t
//! $$
//!
use ndarray::Array1;
use ndarray::s;

use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use crate::traits::FloatExt;
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

impl<T: FloatExt> Ckls<T> {
  pub fn new(
    theta1: T,
    theta2: T,
    theta3: T,
    theta4: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
  ) -> Self {
    Self {
      theta1,
      theta2,
      theta3,
      theta4,
      n,
      x0,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> Ckls<T, Deterministic> {
  pub fn seeded(
    theta1: T,
    theta2: T,
    theta3: T,
    theta4: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: u64,
  ) -> Self {
    Self {
      theta1,
      theta2,
      theta3,
      theta4,
      n,
      x0,
      t,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Ckls<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut ckls = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return ckls;
    }

    ckls[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return ckls;
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    let mut prev = ckls[0];
    let mut tail_view = ckls.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("Ckls output tail must be contiguous");
    let mut seed = self.seed;
    let normal = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &mut seed);
    normal.fill_slice_fast(tail);

    for z in tail.iter_mut() {
      let next = prev
        + (self.theta1 + self.theta2 * prev) * dt
        + self.theta3 * prev.abs().powf(self.theta4) * *z;
      *z = next;
      prev = next;
    }

    ckls
  }
}

py_process_1d!(PyCkls, Ckls,
  sig: (theta1, theta2, theta3, theta4, n, x0=None, t=None, seed=None, dtype=None),
  params: (theta1: f64, theta2: f64, theta3: f64, theta4: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
