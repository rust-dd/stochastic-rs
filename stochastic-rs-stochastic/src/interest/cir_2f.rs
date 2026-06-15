//! # Cir 2f
//!
//! $$
//! r_t=x_t+y_t,\ dx_t=\kappa_1(\theta_1-x_t)dt+\sigma_1\sqrt{x_t}dW_t^1,\ dy_t=\kappa_2(\theta_2-y_t)dt+\sigma_2\sqrt{y_t}dW_t^2
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::cir::Cir;
use crate::buffer::array1_from_fill;
use crate::diffusion::cir::CirSampler;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct Cir2F<T: FloatExt, S: SeedExt = Unseeded> {
  /// Model parameter controlling process dynamics.
  pub x: Cir<T, S>,
  /// Model parameter controlling process dynamics.
  pub y: Cir<T, S>,
  /// Autoregressive coefficient vector.
  pub phi: Fn1D<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Cir2F<T, S> {
  pub fn new(x: Cir<T, S>, y: Cir<T, S>, phi: impl Into<Fn1D<T>>, seed: S) -> Self {
    assert_eq!(x.n, y.n, "x and y Cir factors must use the same n");
    if let (Some(tx), Some(ty)) = (x.t, y.t) {
      assert!(
        (tx - ty).abs() <= T::from_f64_fast(1e-12),
        "x and y Cir factors must use the same time horizon"
      );
    }
    Self {
      x,
      y,
      phi: phi.into(),
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Cir2F<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = Cir2FSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> Cir2FSampler<T> {
    let n = self.x.n;
    let dt = self.x.t.unwrap_or(T::one()) / T::from_usize_(n - 1);
    let phi = Array1::<T>::from_shape_fn(n, |i| self.phi.call(T::from_usize_(i) * dt));
    Cir2FSampler {
      n,
      x: self.x.sampler(),
      y: self.y.sampler(),
      phi,
    }
  }
}

/// Reusable [`Cir2F`] sampling state — owns the two inner [`CirSampler`]s and
/// the precomputed deterministic `φ(t)` curve, so each call resamples both
/// factors and sums `x + y + φ`.
#[doc(hidden)]
pub struct Cir2FSampler<T: FloatExt> {
  n: usize,
  x: CirSampler<T>,
  y: CirSampler<T>,
  phi: Array1<T>,
}

impl<T: FloatExt> Cir2FSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let x = self.x.sample();
    let y = self.y.sample();
    for ((dst, (&xi, &yi)), &p) in out
      .iter_mut()
      .zip(x.iter().zip(y.iter()))
      .zip(self.phi.iter())
    {
      *dst = xi + yi + p;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for Cir2FSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Cir2F output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn phi_fn(t: f64) -> f64 {
    t
  }

  #[test]
  fn default_time_horizon_is_one() {
    let x = Cir::new(0.0_f64, 0.0, 0.0, 3, Some(0.0), None, Some(false), Unseeded);
    let y = Cir::new(0.0_f64, 0.0, 0.0, 3, Some(0.0), None, Some(false), Unseeded);
    let model = Cir2F::new(x, y, phi_fn as fn(f64) -> f64, Unseeded);

    let out = model.sample();
    assert!((out[0] - 0.0).abs() < 1e-12);
    assert!((out[1] - 0.5).abs() < 1e-12);
    assert!((out[2] - 1.0).abs() < 1e-12);
  }

  #[test]
  #[should_panic(expected = "x and y Cir factors must use the same n")]
  fn mismatched_lengths_panic() {
    let x = Cir::new(
      0.0_f64,
      0.0,
      0.0,
      3,
      Some(0.0),
      Some(1.0),
      Some(false),
      Unseeded,
    );
    let y = Cir::new(
      0.0_f64,
      0.0,
      0.0,
      4,
      Some(0.0),
      Some(1.0),
      Some(false),
      Unseeded,
    );
    let _ = Cir2F::new(x, y, phi_fn as fn(f64) -> f64, Unseeded);
  }
}
