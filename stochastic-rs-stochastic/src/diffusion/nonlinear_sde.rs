//! # NonLinearSDE
//!
//! $$
//! dX_t=\left(\frac{a_{-1}}{X_t}+a_0+a_1 X_t+a_2 X_t^2\right)dt+(b_0+b_1 X_t+b_2 X_t^{b_3})\,dW_t
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
pub struct NonLinearSDE<T: FloatExt, S: SeedExt = Unseeded> {
  /// Inverse-state drift coefficient a₋₁ in `a₋₁/X_t + a_0 + a_1 X_t + a_2 X_t²`.
  pub am1: T,
  /// Constant drift coefficient a₀.
  pub a0: T,
  /// Linear drift coefficient a₁.
  pub a1: T,
  /// Quadratic drift coefficient a₂.
  pub a2: T,
  /// Constant diffusion coefficient b₀ in `b_0 + b_1 X_t + b_2 X_t^{b_3}`
  /// (unlike [`AitSahalia`](super::ait_sahalia::AitSahalia), this bracket is
  /// **not** square-rooted — it is the diffusion coefficient directly).
  pub b0: T,
  /// Linear diffusion coefficient b₁.
  pub b1: T,
  /// Power-law diffusion coefficient b₂ scaling `X_t^{b_3}`.
  pub b2: T,
  /// Diffusion exponent b₃ applied to `X_t`.
  pub b3: T,
  /// Number of points sampled along the nonlinear-SDE path.
  pub n: usize,
  /// Initial value X₀ of the nonlinear-SDE path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> NonLinearSDE<T, S> {
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
    seed: S,
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
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for NonLinearSDE<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = NonLinearSdeSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> NonLinearSdeSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    NonLinearSdeSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      am1: self.am1,
      a0: self.a0,
      a1: self.a1,
      a2: self.a2,
      b0: self.b0,
      b1: self.b1,
      b2: self.b2,
      b3: self.b3,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`NonLinearSDE`] sampling state: precomputed Euler step and the
/// owned Gaussian source.
#[doc(hidden)]
pub struct NonLinearSdeSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  am1: T,
  a0: T,
  a1: T,
  a2: T,
  b0: T,
  b1: T,
  b2: T,
  b3: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> NonLinearSdeSampler<T> {
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
      let drift = self.am1 / safe_prev + self.a0 + self.a1 * prev + self.a2 * prev * prev;
      let diff = self.b0 + self.b1 * prev + self.b2 * prev.abs().powf(self.b3);
      let next = prev + drift * self.dt + diff * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for NonLinearSdeSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("NonLinearSDE output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyNonLinearSDE, NonLinearSDE,
  sig: (am1, a0, a1, a2, b0, b1, b2, b3, n, x0=None, t=None, seed=None, dtype=None),
  params: (am1: f64, a0: f64, a1: f64, a2: f64, b0: f64, b1: f64, b2: f64, b3: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
