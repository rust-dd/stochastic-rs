use std::marker::PhantomData;

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::uniform::SimdUniform;

use super::sample_positive_stable;
use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Inverse alpha-stable subordinator:
/// `E_alpha(t) = inf { u >= 0 : D_alpha(u) > t }`.
pub struct InverseAlphaStableSubordinator<T: FloatExt, S: SeedExt = Unseeded> {
  /// Stability index in `(0, 1)`.
  pub alpha: T,
  /// Laplace scale of the direct stable subordinator.
  pub c: T,
  /// Number of target time-grid points.
  pub n: usize,
  /// Horizon for target time-grid.
  pub t: Option<T>,
  /// Internal grid size for the direct process `D_alpha(u)`.
  pub u_steps: usize,
  /// Optional upper bound for inverse-domain `u`.
  pub u_max: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> InverseAlphaStableSubordinator<T, S> {
  pub fn new(
    alpha: T,
    c: T,
    n: usize,
    t: Option<T>,
    u_steps: usize,
    u_max: Option<T>,
    seed: S,
  ) -> Self {
    assert!(
      alpha > T::zero() && alpha < T::one(),
      "alpha must be in (0,1)"
    );
    assert!(c > T::zero(), "c must be positive");
    assert!(u_steps >= 2, "u_steps must be >= 2");
    Self {
      alpha,
      c,
      n,
      t,
      u_steps,
      u_max,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for InverseAlphaStableSubordinator<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = InverseAlphaStableSubordinatorSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> InverseAlphaStableSubordinatorSampler<T> {
    let t_max = self.t.unwrap_or(T::one()).to_f64().unwrap();
    let u_max0 = self
      .u_max
      .unwrap_or(T::from_f64_fast(t_max))
      .to_f64()
      .unwrap();
    InverseAlphaStableSubordinatorSampler {
      n: self.n,
      alpha: self.alpha.to_f64().unwrap(),
      c: self.c.to_f64().unwrap(),
      t_max,
      u_steps: self.u_steps,
      u_max0,
      uniform: SimdUniform::<f64>::new(0.0, 1.0, &self.seed),
      _marker: PhantomData,
    }
  }
}

/// Reusable [`InverseAlphaStableSubordinator`] sampling state: the owned
/// uniform source driving the direct stable path plus the f64 process scalars.
#[doc(hidden)]
pub struct InverseAlphaStableSubordinatorSampler<T: FloatExt> {
  n: usize,
  alpha: f64,
  c: f64,
  t_max: f64,
  u_steps: usize,
  u_max0: f64,
  uniform: SimdUniform<f64>,
  _marker: PhantomData<T>,
}

impl<T: FloatExt> InverseAlphaStableSubordinatorSampler<T> {
  fn simulate_direct_path(&self, u_max: f64) -> (Vec<f64>, Vec<f64>) {
    let m = self.u_steps;
    let du = u_max / (m - 1) as f64;
    let scale = (self.c * du).powf(1.0 / self.alpha);
    let mut u = vec![0.0; m];
    let mut d = vec![0.0; m];
    for i in 1..m {
      u[i] = i as f64 * du;
      d[i] = d[i - 1] + scale * sample_positive_stable(self.alpha, &self.uniform);
    }
    (u, d)
  }

  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    if out.len() == 1 {
      out[0] = T::zero();
      return;
    }

    let mut u_max = self.u_max0;
    if u_max <= 0.0 {
      u_max = self.t_max.max(1.0);
    }

    let mut u = Vec::new();
    let mut d = Vec::new();
    let mut reached = false;

    for _ in 0..10 {
      let (u_try, d_try) = self.simulate_direct_path(u_max);
      if *d_try.last().unwrap_or(&0.0) >= self.t_max {
        u = u_try;
        d = d_try;
        reached = true;
        break;
      }
      u_max *= 2.0;
      u = u_try;
      d = d_try;
    }

    let dt = self.t_max / (out.len() - 1) as f64;
    let mut j = 1usize;
    for (i, x) in out.iter_mut().enumerate() {
      let t_i = i as f64 * dt;
      while j < d.len() && d[j] < t_i {
        j += 1;
      }
      let e_i = if j >= d.len() {
        *u.last().unwrap_or(&u_max)
      } else if d[j] <= d[j - 1] {
        u[j]
      } else {
        let w = (t_i - d[j - 1]) / (d[j] - d[j - 1]);
        u[j - 1] + w * (u[j] - u[j - 1])
      };
      *x = T::from_f64_fast(e_i);
    }

    if !reached {
      for x in out.iter_mut() {
        if !x.is_finite() {
          *x = T::from_f64_fast(u_max);
        }
      }
    }
  }
}

impl<T: FloatExt> PathSampler<T> for InverseAlphaStableSubordinatorSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("InverseAlphaStableSubordinator output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyInverseAlphaStableSubordinator, InverseAlphaStableSubordinator,
  sig: (alpha, c, n, t=None, u_steps=2048, u_max=None, seed=None, dtype=None),
  params: (alpha: f64, c: f64, n: usize, t: Option<f64>, u_steps: usize, u_max: Option<f64>)
);
