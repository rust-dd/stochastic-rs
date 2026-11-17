use ndarray::Array1;
use ndarray::array;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_stochastic::diffusion::ait_sahalia::AitSahalia;
use stochastic_rs_stochastic::diffusion::hyperbolic::Hyperbolic;
use stochastic_rs_stochastic::diffusion::hyperbolic2::Hyperbolic2;
use stochastic_rs_stochastic::diffusion::linear_sde::LinearSde;
use stochastic_rs_stochastic::diffusion::nonlinear_sde::NonlinearSde;
use stochastic_rs_stochastic::diffusion::pearson::Pearson;
use stochastic_rs_stochastic::diffusion::quadratic::Quadratic;

use super::off_zero;
use super::power_x;
use super::power_xx;
use super::root_x;
use super::root_xx;
use crate::mle::DiffusionModel;

impl<S: SeedExt> DiffusionModel for Quadratic<f64, S> {
  fn num_params(&self) -> usize {
    4
  }

  fn params(&self) -> Array1<f64> {
    array![self.alpha, self.beta, self.gamma, self.sigma]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.alpha = p[0];
    self.beta = p[1];
    self.gamma = p[2];
    self.sigma = p[3];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["alpha", "beta", "gamma", "sigma"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (1e-6, 10.0)]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    self.alpha + self.beta * x + self.gamma * x * x
  }

  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    self.sigma * x
  }

  fn drift_x(&self, x: f64, _t: f64) -> f64 {
    self.beta + 2.0 * self.gamma * x
  }

  fn drift_xx(&self, _x: f64, _t: f64) -> f64 {
    2.0 * self.gamma
  }

  fn diffusion_x(&self, _x: f64, _t: f64) -> f64 {
    self.sigma
  }

  fn diffusion_xx(&self, _x: f64, _t: f64) -> f64 {
    0.0
  }
}

impl<S: SeedExt> DiffusionModel for Pearson<f64, S> {
  fn num_params(&self) -> usize {
    5
  }

  fn params(&self) -> Array1<f64> {
    array![self.kappa, self.mu, self.a, self.b, self.c]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.kappa = p[0];
    self.mu = p[1];
    self.a = p[2];
    self.b = p[3];
    self.c = p[4];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["kappa", "mu", "a", "b", "c"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![
      (1e-4, 50.0),
      (-10.0, 10.0),
      (-10.0, 10.0),
      (-10.0, 10.0),
      (1e-6, 10.0),
    ]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    self.kappa * (self.mu - x)
  }

  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    let inner = 2.0 * self.kappa * (self.a * x * x + self.b * x + self.c);
    inner.abs().sqrt()
  }

  fn drift_x(&self, _x: f64, _t: f64) -> f64 {
    -self.kappa
  }

  fn drift_xx(&self, _x: f64, _t: f64) -> f64 {
    0.0
  }

  fn diffusion_x(&self, x: f64, _t: f64) -> f64 {
    let g = 2.0 * self.kappa * (self.a * x * x + self.b * x + self.c);
    root_x(g, 2.0 * self.kappa * (2.0 * self.a * x + self.b))
  }

  fn diffusion_xx(&self, x: f64, _t: f64) -> f64 {
    let g = 2.0 * self.kappa * (self.a * x * x + self.b * x + self.c);
    root_xx(
      g,
      2.0 * self.kappa * (2.0 * self.a * x + self.b),
      4.0 * self.kappa * self.a,
    )
  }
}

impl<S: SeedExt> DiffusionModel for Hyperbolic<f64, S> {
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
    -self.kappa * x / (1.0 + x * x).sqrt()
  }

  fn diffusion(&self, _x: f64, _t: f64) -> f64 {
    self.sigma
  }

  fn drift_x(&self, x: f64, _t: f64) -> f64 {
    -self.kappa / (1.0 + x * x).powf(1.5)
  }

  fn drift_xx(&self, x: f64, _t: f64) -> f64 {
    3.0 * self.kappa * x / (1.0 + x * x).powf(2.5)
  }
}

impl<S: SeedExt> DiffusionModel for Hyperbolic2<f64, S> {
  fn num_params(&self) -> usize {
    5
  }

  fn params(&self) -> Array1<f64> {
    array![self.beta, self.gamma, self.delta, self.mu, self.sigma]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.beta = p[0];
    self.gamma = p[1];
    self.delta = p[2];
    self.mu = p[3];
    self.sigma = p[4];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["beta", "gamma", "delta", "mu", "sigma"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![
      (-10.0, 10.0),
      (1e-6, 10.0),
      (1e-6, 10.0),
      (-10.0, 10.0),
      (1e-6, 10.0),
    ]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    let r = (self.delta * self.delta + (x - self.mu) * (x - self.mu)).sqrt();
    0.5 * self.sigma * self.sigma * (self.beta - self.gamma * x / r)
  }

  fn diffusion(&self, _x: f64, _t: f64) -> f64 {
    self.sigma
  }

  fn drift_x(&self, x: f64, _t: f64) -> f64 {
    let r = (self.delta * self.delta + (x - self.mu) * (x - self.mu)).sqrt();
    -0.5 * self.sigma * self.sigma * self.gamma * (1.0 / r - x * (x - self.mu) / r.powi(3))
  }

  fn drift_xx(&self, x: f64, _t: f64) -> f64 {
    let r = (self.delta * self.delta + (x - self.mu) * (x - self.mu)).sqrt();
    let u = x - self.mu;
    -0.5
      * self.sigma
      * self.sigma
      * self.gamma
      * (3.0 * x * u * u / r.powi(5) - (u + x + u) / r.powi(3))
  }
}

impl<S: SeedExt> DiffusionModel for LinearSde<f64, S> {
  fn num_params(&self) -> usize {
    3
  }

  fn params(&self) -> Array1<f64> {
    array![self.a, self.b, self.c]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.a = p[0];
    self.b = p[1];
    self.c = p[2];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["a", "b", "c"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![(-10.0, 10.0), (-10.0, 10.0), (1e-6, 10.0)]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    self.a + self.b * x
  }

  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    self.c * x
  }

  fn drift_x(&self, _x: f64, _t: f64) -> f64 {
    self.b
  }

  fn drift_xx(&self, _x: f64, _t: f64) -> f64 {
    0.0
  }

  fn diffusion_x(&self, _x: f64, _t: f64) -> f64 {
    self.c
  }

  fn diffusion_xx(&self, _x: f64, _t: f64) -> f64 {
    0.0
  }
}

impl<S: SeedExt> DiffusionModel for NonlinearSde<f64, S> {
  fn num_params(&self) -> usize {
    8
  }

  fn params(&self) -> Array1<f64> {
    array![
      self.am1, self.a0, self.a1, self.a2, self.b0, self.b1, self.b2, self.b3,
    ]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.am1 = p[0];
    self.a0 = p[1];
    self.a1 = p[2];
    self.a2 = p[3];
    self.b0 = p[4];
    self.b1 = p[5];
    self.b2 = p[6];
    self.b3 = p[7];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["am1", "a0", "a1", "a2", "b0", "b1", "b2", "b3"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![
      (-10.0, 10.0),
      (-10.0, 10.0),
      (-10.0, 10.0),
      (-10.0, 10.0),
      (1e-6, 10.0),
      (-10.0, 10.0),
      (-10.0, 10.0),
      (0.01, 3.0),
    ]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    self.am1 / off_zero(x) + self.a0 + self.a1 * x + self.a2 * x * x
  }

  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    self.b0 + self.b1 * x + self.b2 * x.abs().powf(self.b3)
  }

  fn drift_x(&self, x: f64, _t: f64) -> f64 {
    let safe_x = off_zero(x);
    -self.am1 / (safe_x * safe_x) + self.a1 + 2.0 * self.a2 * x
  }

  fn drift_xx(&self, x: f64, _t: f64) -> f64 {
    let safe_x = off_zero(x);
    2.0 * self.am1 / (safe_x * safe_x * safe_x) + 2.0 * self.a2
  }

  fn diffusion_x(&self, x: f64, _t: f64) -> f64 {
    self.b1 + power_x(self.b2, self.b3, x)
  }

  fn diffusion_xx(&self, x: f64, _t: f64) -> f64 {
    power_xx(self.b2, self.b3, x)
  }
}

impl<S: SeedExt> DiffusionModel for AitSahalia<f64, S> {
  fn num_params(&self) -> usize {
    8
  }

  fn params(&self) -> Array1<f64> {
    array![
      self.am1, self.a0, self.a1, self.a2, self.b0, self.b1, self.b2, self.b3,
    ]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.am1 = p[0];
    self.a0 = p[1];
    self.a1 = p[2];
    self.a2 = p[3];
    self.b0 = p[4];
    self.b1 = p[5];
    self.b2 = p[6];
    self.b3 = p[7];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["am1", "a0", "a1", "a2", "b0", "b1", "b2", "b3"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![
      (-10.0, 10.0),
      (-10.0, 10.0),
      (-10.0, 10.0),
      (-10.0, 10.0),
      (1e-6, 10.0),
      (-10.0, 10.0),
      (-10.0, 10.0),
      (0.01, 3.0),
    ]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    self.am1 / off_zero(x) + self.a0 + self.a1 * x + self.a2 * x * x
  }

  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    let inner = self.b0 + self.b1 * x + self.b2 * x.abs().powf(self.b3);
    inner.abs().sqrt()
  }

  fn drift_x(&self, x: f64, _t: f64) -> f64 {
    let safe_x = off_zero(x);
    -self.am1 / (safe_x * safe_x) + self.a1 + 2.0 * self.a2 * x
  }

  fn drift_xx(&self, x: f64, _t: f64) -> f64 {
    let safe_x = off_zero(x);
    2.0 * self.am1 / (safe_x * safe_x * safe_x) + 2.0 * self.a2
  }

  fn diffusion_x(&self, x: f64, _t: f64) -> f64 {
    let g = self.b0 + self.b1 * x + self.b2 * x.abs().powf(self.b3);
    let g1 = self.b1 + power_x(self.b2, self.b3, x);
    root_x(g, g1)
  }

  fn diffusion_xx(&self, x: f64, _t: f64) -> f64 {
    let g = self.b0 + self.b1 * x + self.b2 * x.abs().powf(self.b3);
    let g1 = self.b1 + power_x(self.b2, self.b3, x);
    root_xx(g, g1, power_xx(self.b2, self.b3, x))
  }
}
