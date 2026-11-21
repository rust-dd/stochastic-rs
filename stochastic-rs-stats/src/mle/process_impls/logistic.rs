use ndarray::Array1;
use ndarray::array;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_stochastic::diffusion::feller_logistic::FellerLogistic;
use stochastic_rs_stochastic::diffusion::feller_root::FellerRoot;
use stochastic_rs_stochastic::diffusion::gompertz::Gompertz;
use stochastic_rs_stochastic::diffusion::jacobi::Jacobi;
use stochastic_rs_stochastic::diffusion::kimura::Kimura;
use stochastic_rs_stochastic::diffusion::logistic::Logistic;
use stochastic_rs_stochastic::diffusion::three_half::ThreeHalf;
use stochastic_rs_stochastic::diffusion::verhulst::Verhulst;

use super::off_zero;
use super::power_x;
use super::power_xx;
use super::root_x;
use super::root_xx;
use crate::mle::DiffusionModel;

impl<S: SeedExt> DiffusionModel for Jacobi<f64, S> {
  fn num_params(&self) -> usize {
    3
  }

  fn params(&self) -> Array1<f64> {
    array![self.alpha, self.beta, self.sigma]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.alpha = p[0];
    self.beta = p[1];
    self.sigma = p[2];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["alpha", "beta", "sigma"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![(-10.0, 10.0), (1e-4, 50.0), (1e-6, 10.0)]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    self.alpha - self.beta * x
  }

  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    self.sigma * (x.abs() * (1.0 - x).abs()).sqrt()
  }

  fn drift_x(&self, _x: f64, _t: f64) -> f64 {
    -self.beta
  }

  fn drift_xx(&self, _x: f64, _t: f64) -> f64 {
    0.0
  }

  fn diffusion_x(&self, x: f64, _t: f64) -> f64 {
    self.sigma * root_x(x * (1.0 - x), 1.0 - 2.0 * x)
  }

  fn diffusion_xx(&self, x: f64, _t: f64) -> f64 {
    self.sigma * root_xx(x * (1.0 - x), 1.0 - 2.0 * x, -2.0)
  }
}

impl<S: SeedExt> DiffusionModel for Verhulst<f64, S> {
  fn num_params(&self) -> usize {
    3
  }

  fn params(&self) -> Array1<f64> {
    array![self.r, self.k, self.sigma]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.r = p[0];
    self.k = p[1];
    self.sigma = p[2];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["r", "k", "sigma"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![(1e-4, 50.0), (1e-6, 50.0), (1e-6, 10.0)]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    self.r * x * (1.0 - x / self.k)
  }

  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    self.sigma * x
  }

  fn drift_x(&self, x: f64, _t: f64) -> f64 {
    self.r * (1.0 - 2.0 * x / self.k)
  }

  fn drift_xx(&self, _x: f64, _t: f64) -> f64 {
    -2.0 * self.r / self.k
  }

  fn diffusion_x(&self, _x: f64, _t: f64) -> f64 {
    self.sigma
  }

  fn diffusion_xx(&self, _x: f64, _t: f64) -> f64 {
    0.0
  }

  fn is_positive(&self) -> bool {
    true
  }
}

impl<S: SeedExt> DiffusionModel for FellerLogistic<f64, S> {
  fn num_params(&self) -> usize {
    3
  }

  fn params(&self) -> Array1<f64> {
    array![self.kappa, self.theta, self.sigma]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.kappa = p[0];
    self.theta = p[1];
    self.sigma = p[2];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["kappa", "theta", "sigma"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![(1e-4, 50.0), (1e-6, 50.0), (1e-6, 10.0)]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    self.kappa * (self.theta - x) * x
  }

  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    self.sigma * x.abs().sqrt()
  }

  fn drift_x(&self, x: f64, _t: f64) -> f64 {
    self.kappa * (self.theta - 2.0 * x)
  }

  fn drift_xx(&self, _x: f64, _t: f64) -> f64 {
    -2.0 * self.kappa
  }

  fn diffusion_x(&self, x: f64, _t: f64) -> f64 {
    power_x(self.sigma, 0.5, x)
  }

  fn diffusion_xx(&self, x: f64, _t: f64) -> f64 {
    power_xx(self.sigma, 0.5, x)
  }

  fn is_positive(&self) -> bool {
    true
  }
}

impl<S: SeedExt> DiffusionModel for Logistic<f64, S> {
  fn num_params(&self) -> usize {
    2
  }

  fn params(&self) -> Array1<f64> {
    array![self.a, self.b]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.a = p[0];
    self.b = p[1];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["a", "b"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![(1e-6, 50.0), (1e-6, 10.0)]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    x * (1.0 - self.a * x)
  }

  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    self.b * x
  }

  fn drift_x(&self, x: f64, _t: f64) -> f64 {
    1.0 - 2.0 * self.a * x
  }

  fn drift_xx(&self, _x: f64, _t: f64) -> f64 {
    -2.0 * self.a
  }

  fn diffusion_x(&self, _x: f64, _t: f64) -> f64 {
    self.b
  }

  fn diffusion_xx(&self, _x: f64, _t: f64) -> f64 {
    0.0
  }

  fn is_positive(&self) -> bool {
    true
  }
}

impl<S: SeedExt> DiffusionModel for ThreeHalf<f64, S> {
  fn num_params(&self) -> usize {
    3
  }

  fn params(&self) -> Array1<f64> {
    array![self.kappa, self.mu, self.sigma]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.kappa = p[0];
    self.mu = p[1];
    self.sigma = p[2];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["kappa", "mu", "sigma"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![(1e-4, 50.0), (1e-6, 50.0), (1e-6, 10.0)]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    self.kappa * x * (self.mu - x)
  }

  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    self.sigma * x.abs().powf(1.5)
  }

  fn drift_x(&self, x: f64, _t: f64) -> f64 {
    self.kappa * (self.mu - 2.0 * x)
  }

  fn drift_xx(&self, _x: f64, _t: f64) -> f64 {
    -2.0 * self.kappa
  }

  fn diffusion_x(&self, x: f64, _t: f64) -> f64 {
    power_x(self.sigma, 1.5, x)
  }

  fn diffusion_xx(&self, x: f64, _t: f64) -> f64 {
    power_xx(self.sigma, 1.5, x)
  }

  fn is_positive(&self) -> bool {
    true
  }
}

impl<S: SeedExt> DiffusionModel for FellerRoot<f64, S> {
  fn num_params(&self) -> usize {
    3
  }

  fn params(&self) -> Array1<f64> {
    array![self.theta1, self.theta2, self.theta3]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.theta1 = p[0];
    self.theta2 = p[1];
    self.theta3 = p[2];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["theta1", "theta2", "theta3"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![(-10.0, 10.0), (-10.0, 10.0), (1e-6, 10.0)]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    x * (self.theta1 - x * (self.theta3.powi(3) - self.theta1 * self.theta2))
  }

  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    self.theta3 * x.abs().powf(1.5)
  }

  fn drift_x(&self, x: f64, _t: f64) -> f64 {
    self.theta1 - 2.0 * x * (self.theta3.powi(3) - self.theta1 * self.theta2)
  }

  fn drift_xx(&self, _x: f64, _t: f64) -> f64 {
    -2.0 * (self.theta3.powi(3) - self.theta1 * self.theta2)
  }

  fn diffusion_x(&self, x: f64, _t: f64) -> f64 {
    power_x(self.theta3, 1.5, x)
  }

  fn diffusion_xx(&self, x: f64, _t: f64) -> f64 {
    power_xx(self.theta3, 1.5, x)
  }

  fn is_positive(&self) -> bool {
    true
  }
}

impl<S: SeedExt> DiffusionModel for Kimura<f64, S> {
  fn num_params(&self) -> usize {
    2
  }

  fn params(&self) -> Array1<f64> {
    array![self.a, self.sigma]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.a = p[0];
    self.sigma = p[1];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["a", "sigma"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![(-10.0, 10.0), (1e-6, 10.0)]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    self.a * x * (1.0 - x)
  }

  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    self.sigma * (x.abs() * (1.0 - x).abs()).sqrt()
  }

  fn drift_x(&self, x: f64, _t: f64) -> f64 {
    self.a * (1.0 - 2.0 * x)
  }

  fn drift_xx(&self, _x: f64, _t: f64) -> f64 {
    -2.0 * self.a
  }

  fn diffusion_x(&self, x: f64, _t: f64) -> f64 {
    self.sigma * root_x(x * (1.0 - x), 1.0 - 2.0 * x)
  }

  fn diffusion_xx(&self, x: f64, _t: f64) -> f64 {
    self.sigma * root_xx(x * (1.0 - x), 1.0 - 2.0 * x, -2.0)
  }
}

impl<S: SeedExt> DiffusionModel for Gompertz<f64, S> {
  fn num_params(&self) -> usize {
    3
  }

  fn params(&self) -> Array1<f64> {
    array![self.a, self.b, self.sigma]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.a = p[0];
    self.b = p[1];
    self.sigma = p[2];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["a", "b", "sigma"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![(-10.0, 10.0), (-10.0, 10.0), (1e-6, 10.0)]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    let safe_x = off_zero(x);
    (self.a - self.b * safe_x.abs().ln()) * safe_x
  }

  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    self.sigma * x
  }

  fn drift_x(&self, x: f64, _t: f64) -> f64 {
    self.a - self.b * (off_zero(x).abs().ln() + 1.0)
  }

  fn drift_xx(&self, x: f64, _t: f64) -> f64 {
    -self.b / off_zero(x)
  }

  fn diffusion_x(&self, _x: f64, _t: f64) -> f64 {
    self.sigma
  }

  fn diffusion_xx(&self, _x: f64, _t: f64) -> f64 {
    0.0
  }

  fn is_positive(&self) -> bool {
    true
  }
}
