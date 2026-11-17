use ndarray::Array1;
use ndarray::array;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::diffusion::gbm_ih::GbmIh;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::process::bm::Bm;

use crate::mle::DiffusionModel;
use crate::mle::density::gaussian_ln_pdf;
use crate::mle::density::gaussian_pdf;

impl<S: SeedExt> DiffusionModel for Bm<f64, S> {
  fn num_params(&self) -> usize {
    0
  }

  fn params(&self) -> Array1<f64> {
    array![]
  }

  fn set_params(&mut self, _p: &[f64]) {}
  fn param_names(&self) -> Vec<&str> {
    vec![]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![]
  }

  fn drift(&self, _x: f64, _t: f64) -> f64 {
    0.0
  }

  fn diffusion(&self, _x: f64, _t: f64) -> f64 {
    1.0
  }

  fn exact_density(&self, x0: f64, xt: f64, _t0: f64, dt: f64) -> Option<f64> {
    Some(gaussian_pdf(xt, x0, dt))
  }

  fn exact_ln_density(&self, x0: f64, xt: f64, _t0: f64, dt: f64) -> Option<f64> {
    Some(gaussian_ln_pdf(xt, x0, dt))
  }
}

impl<S: SeedExt> DiffusionModel for GbmIh<f64, S> {
  fn num_params(&self) -> usize {
    2
  }

  fn params(&self) -> Array1<f64> {
    array![self.mu, self.sigma]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.mu = p[0];
    self.sigma = p[1];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["mu", "sigma"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![(-5.0, 5.0), (1e-6, 5.0)]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    self.mu * x
  }

  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    self.sigma * x
  }

  fn is_positive(&self) -> bool {
    true
  }
}

impl<S: SeedExt> DiffusionModel for Gbm<f64, S> {
  fn num_params(&self) -> usize {
    2
  }

  fn params(&self) -> Array1<f64> {
    array![self.mu, self.sigma]
  }

  fn set_params(&mut self, p: &[f64]) {
    self.mu = p[0];
    self.sigma = p[1];
  }

  fn param_names(&self) -> Vec<&str> {
    vec!["mu", "sigma"]
  }

  fn param_bounds(&self) -> Vec<(f64, f64)> {
    vec![(-5.0, 5.0), (1e-6, 5.0)]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    self.mu * x
  }

  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    self.sigma * x
  }

  fn is_positive(&self) -> bool {
    true
  }

  fn exact_density(&self, x0: f64, xt: f64, _t0: f64, dt: f64) -> Option<f64> {
    use std::f64::consts::PI;
    if x0.is_nan() || x0 <= 0.0 {
      return Some(f64::NAN);
    }
    if xt <= 0.0 {
      return Some(0.0);
    }
    let (log_mean, log_var) = gbm_log_law(self, x0, dt);
    let z = (xt.ln() - log_mean) / log_var.sqrt();
    Some((-0.5 * z * z).exp() / (xt * (2.0 * PI * log_var).sqrt()))
  }

  fn exact_ln_density(&self, x0: f64, xt: f64, _t0: f64, dt: f64) -> Option<f64> {
    if x0.is_nan() || x0 <= 0.0 {
      return Some(f64::NAN);
    }
    if xt <= 0.0 {
      return Some(f64::NEG_INFINITY);
    }
    let (log_mean, log_var) = gbm_log_law(self, x0, dt);
    Some(gaussian_ln_pdf(xt.ln(), log_mean, log_var) - xt.ln())
  }

  fn exact_step(&self, _t: f64, dt: f64, x: f64, dz: f64) -> Option<f64> {
    Some(x * ((self.mu - 0.5 * self.sigma * self.sigma) * dt + self.sigma * dt.sqrt() * dz).exp())
  }
}

impl<S: SeedExt> DiffusionModel for Ou<f64, S> {
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
    vec![(1e-4, 50.0), (-10.0, 10.0), (1e-6, 10.0)]
  }

  fn drift(&self, x: f64, _t: f64) -> f64 {
    self.theta * (self.mu - x)
  }

  fn diffusion(&self, _x: f64, _t: f64) -> f64 {
    self.sigma
  }

  fn exact_density(&self, x0: f64, xt: f64, _t0: f64, dt: f64) -> Option<f64> {
    let (mean, var) = ou_law(self, x0, dt);
    Some(gaussian_pdf(xt, mean, var))
  }

  fn exact_ln_density(&self, x0: f64, xt: f64, _t0: f64, dt: f64) -> Option<f64> {
    let (mean, var) = ou_law(self, x0, dt);
    Some(gaussian_ln_pdf(xt, mean, var))
  }

  fn exact_step(&self, _t: f64, dt: f64, x: f64, dz: f64) -> Option<f64> {
    let (mean, var) = ou_law(self, x, dt);
    Some(mean + var.sqrt() * dz)
  }
}

/// The mean and variance of `ln X_{t+Δ}` from `x0`, a normal.
fn gbm_log_law<S: SeedExt>(gbm: &Gbm<f64, S>, x0: f64, dt: f64) -> (f64, f64) {
  let log_mean = x0.ln() + (gbm.mu - 0.5 * gbm.sigma * gbm.sigma) * dt;
  (log_mean, gbm.sigma * gbm.sigma * dt)
}

/// The mean and variance of `X_{t+Δ}` from `x0`, a normal.
fn ou_law<S: SeedExt>(ou: &Ou<f64, S>, x0: f64, dt: f64) -> (f64, f64) {
  let e = (-ou.theta * dt).exp();
  let mean = ou.mu + (x0 - ou.mu) * e;
  (mean, ou.sigma * ou.sigma / (2.0 * ou.theta) * (1.0 - e * e))
}
