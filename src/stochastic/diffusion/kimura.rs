//! # Kimura
//!
//! $$
//! dX_t=(a+bX_t)dt+\sigma\sqrt{X_t}\,dW_t
//! $$
//!
use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Kimura / Wright–Fisher diffusion
/// dX_t = a X_t (1 - X_t) dt + sigma sqrt(X_t (1 - X_t)) dW_t
pub struct Kimura<T: FloatExt> {
  /// Model coefficient / user-supplied drift term.
  pub a: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Gaussian increment generator used internally.
  pub gn: Gn<T>,
}

impl<T: FloatExt> Kimura<T> {
  pub fn new(a: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Kimura {
      a,
      sigma,
      n,
      x0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for Kimura<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = self.gn.sample();

    let mut x = Array1::<T>::zeros(self.n);
    x[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      // enforce [0,1] domain when computing coefficients
      let xi = x[i - 1].clamp(T::zero(), T::one());
      let sqrt_term = (xi * (T::one() - xi)).sqrt();
      let drift = self.a * xi * (T::one() - xi) * dt;
      let diff = self.sigma * sqrt_term * gn[i - 1];
      let mut next = xi + drift + diff;
      next = next.clamp(T::zero(), T::one());
      x[i] = next;
    }

    x
  }
}

py_process_1d!(PyKimura, Kimura,
  sig: (a, sigma, n, x0=None, t=None, dtype=None),
  params: (a: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
