//! A multi-dimensional Wu–Zhang-style model, combining a forward rate and a stochastic
//! volatility process for each dimension.
//!
//! Produces a 2D array (`(2*xn, n)`) of forward rates and volatilities. The first `xn` rows
//! are the forward rates; the next `xn` rows are the volatilities, each evolving over `n` steps.
//!
//! # Parameters
//! - `alpha`: Mean reversion level for each dimension's volatility.
//! - `beta`: Mean reversion speed for each dimension's volatility.
//! - `nu`: Volatility-of-volatility parameter for each dimension.
//! - `lambda`: Parameter controlling how volatility impacts the forward rate.
//! - `x0`: Initial forward rates for each dimension.
//! - `v0`: Initial volatilities for each dimension.
//! - `xn`: Number of `(rate, vol)` pairs.
//! - `t`: Total time horizon.
//! - `n`: Number of time steps in the simulation.
//! - `m`: Batch size for parallel sampling (if used).

use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray::Array2;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

#[derive(ImplNew)]
pub struct WuZhangD<T: Float> {
  /// Mean reversion level for each dimension's volatility.
  pub alpha: Array1<T>,
  /// Mean reversion speed for each dimension's volatility.
  pub beta: Array1<T>,
  /// Volatility of volatility for each dimension.
  pub nu: Array1<T>,
  /// Parameter controlling the impact of volatility on the forward rate.
  pub lambda: Array1<T>,
  /// Initial forward rates for each dimension.
  pub x0: Array1<T>,
  /// Initial volatilities for each dimension.
  pub v0: Array1<T>,
  /// Number of (rate, vol) pairs.
  pub xn: usize,
  /// Total time horizon.
  pub t: Option<T>,
  /// Number of time steps in the simulation.
  pub n: usize,
}

impl<T: Float> Process<T> for WuZhangD<T> {
  type Output = Array2<T>;
  type Noise = Gn<T>;

  fn sample(&self) -> Self::Output {
    self.euler_maruyama(|gn| gn.sample())
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    self.euler_maruyama(|gn| gn.sample_simd())
  }

  fn euler_maruyama(
    &self,
    noise_fn: impl Fn(&Self::Noise) -> <Self::Noise as Process<T>>::Output,
  ) -> Self::Output {
    let gn = Gn::new(self.n - 1, self.t);
    let dt = gn.dt();
    let mut fv = Array2::<T>::zeros((2 * self.xn, self.n));

    for i in 0..self.xn {
      fv[(i, 0)] = self.x0[i];
      fv[(i + self.xn, 0)] = self.v0[i];
    }

    for i in 0..self.xn {
      let gn_f = noise_fn(&gn);
      let gn_v = noise_fn(&gn);

      for j in 1..self.n {
        let v_old = fv[(i + self.xn, j - 1)].max(T::zero());
        let f_old = fv[(i, j - 1)].max(T::zero());

        let dv =
          (self.alpha[i] - self.beta[i] * v_old) * dt + self.nu[i] * v_old.sqrt() * gn_v[j - 1];

        let v_new = (v_old + dv).max(T::zero());
        fv[(i + self.xn, j)] = v_new;

        let df = f_old * self.lambda[i] * v_new.sqrt() * gn_f[j - 1];
        fv[(i, j)] = f_old + df;
      }
    }

    fv
  }
}
