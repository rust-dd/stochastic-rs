//! Maximum likelihood for one-dimensional diffusions through a transition-density approximation.
//!
//! $$
//! \hat\theta = \arg\max_\theta \sum_{i=1}^{N} \ln p(X_{t_i} \mid X_{t_{i-1}};\theta,\Delta t),
//! \qquad dX_t = \mu(X_t, t;\theta)\,dt + \sigma(X_t, t;\theta)\,dW_t
//! $$
//!
//! The densities are Euler, Ozaki, Shoji–Ozaki, Elerian, Kessler, the exact one and Aït-Sahalia's.

mod density;
mod fit;
mod process_impls;

pub use density::DensityApprox;
pub use fit::MleFit;
pub use fit::fit_mle;

/// A one-dimensional stochastic differential equation
///
/// $$
/// dX_t = \mu(X_t,t)\,dt + \sigma(X_t,t)\,dW_t
/// $$
pub trait DiffusionModel: Send + Sync {
  /// Number of parameters.
  fn num_params(&self) -> usize;

  /// Current parameter vector.
  fn params(&self) -> ndarray::Array1<f64>;

  /// Set parameters from a slice (optimizer calls this).
  fn set_params(&mut self, params: &[f64]);

  /// Human-readable parameter names (for display).
  fn param_names(&self) -> Vec<&str>;

  /// Default parameter bounds for bounded optimisation.
  fn param_bounds(&self) -> Vec<(f64, f64)>;

  /// Drift coefficient mu(x, t).
  fn drift(&self, x: f64, t: f64) -> f64;

  /// Diffusion coefficient sigma(x, t).
  fn diffusion(&self, x: f64, t: f64) -> f64;

  /// Whether the process is non-negative (enables clamping in simulation).
  fn is_positive(&self) -> bool {
    false
  }

  /// d mu / d x (central difference).
  fn drift_x(&self, x: f64, t: f64) -> f64 {
    let h = 1e-5;
    (self.drift(x + h, t) - self.drift(x - h, t)) / (2.0 * h)
  }

  /// d^2 mu / d x^2 (central difference).
  fn drift_xx(&self, x: f64, t: f64) -> f64 {
    let h = 1e-5;
    (self.drift(x + h, t) - 2.0 * self.drift(x, t) + self.drift(x - h, t)) / (h * h)
  }

  /// d mu / d t (central difference).
  fn drift_t(&self, x: f64, t: f64) -> f64 {
    let h = 1e-5;
    (self.drift(x, t + h) - self.drift(x, t - h)) / (2.0 * h)
  }

  /// d sigma / d x (central difference).
  fn diffusion_x(&self, x: f64, t: f64) -> f64 {
    let h = 1e-5;
    (self.diffusion(x + h, t) - self.diffusion(x - h, t)) / (2.0 * h)
  }

  /// d^2 sigma / d x^2 (central difference).
  fn diffusion_xx(&self, x: f64, t: f64) -> f64 {
    let h = 1e-5;
    (self.diffusion(x + h, t) - 2.0 * self.diffusion(x, t) + self.diffusion(x - h, t)) / (h * h)
  }

  /// The exact density of `xt` from `x0` over `dt`, `None` if the model has none: `0.0` outside the
  /// support and NaN where there is no density, never a positive stand-in.
  fn exact_density(&self, _x0: f64, _xt: f64, _t0: f64, _dt: f64) -> Option<f64> {
    None
  }

  /// The log of [`exact_density`](Self::exact_density), overridden to keep the tails where it
  /// underflows: `None` where there is no exact density, `−∞` outside the support.
  fn exact_ln_density(&self, x0: f64, xt: f64, t0: f64, dt: f64) -> Option<f64> {
    self.exact_density(x0, xt, t0, dt).map(f64::ln)
  }

  /// Aït-Sahalia Hermite density expansion. Return `None` if not available.
  fn ait_sahalia_density(&self, _x0: f64, _xt: f64, _t0: f64, _dt: f64) -> Option<f64> {
    None
  }

  /// Exact simulation step (return `None` to fall back to Euler).
  fn exact_step(&self, _t: f64, _dt: f64, _x: f64, _dz: f64) -> Option<f64> {
    None
  }
}

#[cfg(test)]
mod tests;
