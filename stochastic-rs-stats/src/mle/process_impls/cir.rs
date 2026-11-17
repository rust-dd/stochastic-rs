use ndarray::Array1;
use ndarray::array;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_stochastic::diffusion::cev::Cev;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::ckls::Ckls;
use stochastic_rs_stochastic::diffusion::modified_cir::ModifiedCir;
use stochastic_rs_stochastic::diffusion::radial_ou::RadialOu;

use crate::mle::DiffusionModel;

impl<S: SeedExt> DiffusionModel for Cir<f64, S> {
  fn num_params(&self) -> usize {
    3
  }

  fn params(&self) -> Array1<f64> {
    array![self.theta, self.mu, self.sigma]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.theta = p[0];
    self.mu = p[1];
    self.sigma = p[2];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["theta", "mu", "sigma"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![(1e-4, 50.0), (1e-6, 10.0), (1e-6, 10.0)]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    self.theta * (self.mu - x)
  }

  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    self.sigma * x.abs().sqrt()
  }

  /// `σ/(2√x)`: unbounded at zero, so the trait's fixed difference step misses it below `1e-5`.
  fn diffusion_x(&self, x: f64, _t: f64) -> f64 {
    0.5 * self.sigma * x.signum() / x.abs().sqrt()
  }

  fn is_positive(&self) -> bool {
    2.0 * self.theta * self.mu >= self.sigma * self.sigma
  }

  fn exact_density(&self, x0: f64, xt: f64, _t0: f64, dt: f64) -> Option<f64> {
    Some(self.transition_pdf(dt, x0, xt))
  }

  fn exact_ln_density(&self, x0: f64, xt: f64, _t0: f64, dt: f64) -> Option<f64> {
    Some(self.transition_ln_pdf(dt, x0, xt))
  }
}

impl<S: SeedExt> DiffusionModel for Cev<f64, S> {
  fn num_params(&self) -> usize {
    3
  }

  fn params(&self) -> Array1<f64> {
    array![self.mu, self.sigma, self.gamma]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.mu = p[0];
    self.sigma = p[1];
    self.gamma = p[2];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["mu", "sigma", "gamma"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![(-5.0, 5.0), (1e-6, 5.0), (0.01, 2.0)]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    self.mu * x
  }

  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    self.sigma * x.abs().powf(self.gamma)
  }

  fn is_positive(&self) -> bool {
    self.gamma >= 0.5
  }
}

impl<S: SeedExt> DiffusionModel for Ckls<f64, S> {
  fn num_params(&self) -> usize {
    4
  }

  fn params(&self) -> Array1<f64> {
    array![self.theta1, self.theta2, self.theta3, self.theta4]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.theta1 = p[0];
    self.theta2 = p[1];
    self.theta3 = p[2];
    self.theta4 = p[3];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["theta1", "theta2", "theta3", "theta4"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![(-10.0, 10.0), (-10.0, 10.0), (1e-6, 10.0), (0.01, 3.0)]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    self.theta1 + self.theta2 * x
  }

  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    self.theta3 * x.abs().powf(self.theta4)
  }
}

impl<S: SeedExt> DiffusionModel for ModifiedCir<f64, S> {
  fn num_params(&self) -> usize {
    2
  }

  fn params(&self) -> Array1<f64> {
    array![self.kappa, self.sigma]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.kappa = p[0];
    self.sigma = p[1];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["kappa", "sigma"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![(1e-4, 50.0), (1e-6, 10.0)]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    -self.kappa * x
  }

  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    self.sigma * (1.0 + x * x).sqrt()
  }
}

impl<S: SeedExt> DiffusionModel for RadialOu<f64, S> {
  fn num_params(&self) -> usize {
    2
  }

  fn params(&self) -> Array1<f64> {
    array![self.kappa, self.sigma]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.kappa = p[0];
    self.sigma = p[1];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["kappa", "sigma"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![(1e-4, 50.0), (1e-6, 10.0)]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    let safe_x = if x.abs() < 1e-12 { 1e-12 } else { x };
    self.kappa / safe_x - safe_x
  }

  fn diffusion(&self, _x: f64, _t: f64) -> f64 {
    self.sigma
  }

  fn is_positive(&self) -> bool {
    true
  }
}
