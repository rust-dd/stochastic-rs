//! # Sabr
//!
//! $$
//! dF_t=\alpha_t F_t^\beta dW_t^1,\quad d\alpha_t=\nu\alpha_t dW_t^2,\ d\langle W^1,W^2\rangle_t=\rho dt
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct Sabr<T: FloatExt, S: SeedExt = Unseeded> {
  /// Model shape / loading parameter.
  pub alpha: T,
  /// Model slope / loading parameter.
  pub beta: T,
  /// Instantaneous correlation parameter.
  pub rho: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial forward-rate level.
  pub f0: Option<T>,
  /// Initial variance/volatility level.
  pub v0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
  cgns: Cgns<T>,
}

impl<T: FloatExt, S: SeedExt> Sabr<T, S> {
  pub fn new(
    alpha: T,
    beta: T,
    rho: T,
    n: usize,
    f0: Option<T>,
    v0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(
      beta >= T::zero() && beta <= T::one(),
      "beta must be in [0, 1] for Sabr"
    );
    assert!(alpha >= T::zero(), "alpha must be non-negative");
    if let Some(v0) = v0 {
      assert!(v0 >= T::zero(), "v0 must be non-negative");
    }

    Self {
      alpha,
      beta,
      rho,
      n,
      f0,
      v0,
      t,
      seed,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Sabr<T, S> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.cgns.dt();
    let [cgn1, cgn2] = &self.cgns.sample_impl(&self.seed.derive());

    let mut f_ = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);

    f_[0] = self.f0.unwrap_or(T::zero());
    v[0] = self.v0.unwrap_or(T::zero()).max(T::zero());

    for i in 1..self.n {
      let f_prev = f_[i - 1].max(T::zero());
      let v_prev = v[i - 1].max(T::zero());
      f_[i] = f_[i - 1] + v_prev * f_prev.powf(self.beta) * cgn1[i - 1];
      // Exact step for dV = alpha * V * dW preserves non-negativity.
      v[i] =
        v_prev * (self.alpha * cgn2[i - 1] - T::from_f64_fast(0.5) * self.alpha.powi(2) * dt).exp();
    }

    [f_, v]
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn volatility_stays_non_negative() {
    let p = Sabr::new(
      0.4_f64,
      0.5,
      -0.3,
      256,
      Some(1.0),
      Some(0.2),
      Some(1.0),
      Unseeded,
    );
    let [_f, v] = p.sample();
    assert!(v.iter().all(|x| *x >= 0.0));
  }
}

impl<T: FloatExt, S: SeedExt> Sabr<T, S> {
  /// Calculate the Malliavin derivative of the Sabr model
  ///
  /// The Malliavin derivative of the volaility process in the Sabr model is given by:
  /// D_r \sigma_t = \alpha \sigma_t 1_{[0, T]}(r)
  pub fn malliavin_of_vol(&self) -> [Array1<T>; 3] {
    let [f, v] = self.sample();

    let mut malliavin = Array1::<T>::zeros(self.n);

    for i in 0..self.n {
      malliavin[i] = self.alpha * *v.last().unwrap();
    }

    [f, v, malliavin]
  }
}

py_process_2x1d!(PySabr, Sabr,
  sig: (alpha, beta, rho, n, f0=None, v0=None, t=None, seed=None, dtype=None),
  params: (alpha: f64, beta: f64, rho: f64, n: usize, f0: Option<f64>, v0: Option<f64>, t: Option<f64>)
);
