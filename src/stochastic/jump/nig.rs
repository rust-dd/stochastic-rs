//! # NIG
//!
//! $$
//! X_t=\mu t+\beta I_t+W_{I_t},\quad I_t\sim\mathrm{IG}(\delta t,\gamma)
//! $$
//!
use ndarray::Array1;
use ndarray_rand::RandomExt;

use crate::distributions::inverse_gauss::SimdInverseGauss;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct NIG<T: FloatExt> {
  /// Long-run target level / model location parameter.
  pub theta: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Mean-reversion speed parameter.
  pub kappa: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
}

impl<T: FloatExt> NIG<T> {
  pub fn new(theta: T, sigma: T, kappa: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    assert!(kappa > T::zero(), "kappa must be positive");
    Self {
      theta,
      sigma,
      kappa,
      n,
      x0,
      t,
    }
  }

  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }
}

impl<T: FloatExt> ProcessExt<T> for NIG<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut nig = Array1::zeros(self.n);
    if self.n == 0 {
      return nig;
    }
    nig[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return nig;
    }

    let dt = self.dt();
    // For NIG: G_dt ~ IG(mean=dt, shape=dt^2/kappa).
    let shape = dt * dt / self.kappa;
    let ig_dist = SimdInverseGauss::new(dt, shape);
    let ig = Array1::random(self.n - 1, &ig_dist);
    let mut z = Array1::<T>::zeros(self.n - 1);
    let z_slice = z.as_slice_mut().expect("NIG normals must be contiguous");
    T::fill_standard_normal_slice(z_slice);

    for i in 1..self.n {
      nig[i] = nig[i - 1] + self.theta * ig[i - 1] + self.sigma * ig[i - 1].sqrt() * z[i - 1]
    }

    nig
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn n_eq_1_keeps_initial_value() {
    let p = NIG::new(0.1_f64, 0.2, 0.3, 1, Some(4.0), Some(1.0));
    let x = p.sample();
    assert_eq!(x.len(), 1);
    assert_eq!(x[0], 4.0);
  }
}

py_process_1d!(PyNIG, NIG,
  sig: (theta, sigma, kappa, n, x0=None, t=None, dtype=None),
  params: (theta: f64, sigma: f64, kappa: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);