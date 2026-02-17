//! # Feller
//!
//! $$
//! dX_t=a(t,X_t)dt+b(t,X_t)dW_t
//! $$
//!
use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Feller–logistic diffusion
/// dX_t = kappa (theta - X_t) X_t dt + sigma sqrt(X_t) dW_t
pub struct FellerLogistic<T: FloatExt> {
  /// Mean-reversion speed parameter.
  pub kappa: T,
  /// Long-run target level / model location parameter.
  pub theta: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// If true, reflect at 0; otherwise clamp at 0
  pub use_sym: Option<bool>,
  gn: Gn<T>,
}

impl<T: FloatExt> FellerLogistic<T> {
  pub fn new(
    kappa: T,
    theta: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
  ) -> Self {
    Self {
      kappa,
      theta,
      sigma,
      n,
      x0,
      t,
      use_sym,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for FellerLogistic<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = self.gn.sample();

    let mut x = Array1::<T>::zeros(self.n);
    x[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      let xi = x[i - 1].max(T::zero());
      let drift = self.kappa * (self.theta - xi) * xi * dt;
      let diff = self.sigma * xi.sqrt() * gn[i - 1];
      let next = xi + drift + diff;
      x[i] = match self.use_sym.unwrap_or(false) {
        true => next.abs(),
        false => next.max(T::zero()),
      };
    }

    x
  }
}

py_process_1d!(PyFellerLogistic, FellerLogistic,
  sig: (kappa, theta, sigma, n, x0=None, t=None, use_sym=None, dtype=None),
  params: (kappa: f64, theta: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, use_sym: Option<bool>)
);