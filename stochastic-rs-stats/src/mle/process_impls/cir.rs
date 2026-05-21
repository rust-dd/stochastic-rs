use ndarray::Array1;
use ndarray::array;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_stochastic::diffusion::cev::Cev;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::ckls::Ckls;
use stochastic_rs_stochastic::diffusion::modified_cir::ModifiedCIR;
use stochastic_rs_stochastic::diffusion::radial_ou::RadialOU;

use crate::mle::DiffusionModel;

fn log_bessel_i(nu: f64, z: f64) -> f64 {
  use stochastic_rs_distributions::special::ln_gamma;

  if z.abs() < 1e-30 {
    return f64::NEG_INFINITY;
  }

  let half_z = 0.5 * z;
  let log_half_z = half_z.ln();
  let max_terms = 80;
  let mut log_terms = Vec::with_capacity(max_terms);
  for k in 0..max_terms {
    let kf = k as f64;
    let log_term = 2.0 * kf * log_half_z - ln_gamma(kf + 1.0) - ln_gamma(nu + kf + 1.0);
    log_terms.push(log_term);
  }

  let max_log = log_terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
  let sum: f64 = log_terms.iter().map(|&lt| (lt - max_log).exp()).sum();

  nu * log_half_z + max_log + sum.ln()
}

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
  fn is_positive(&self) -> bool {
    2.0 * self.theta * self.mu >= self.sigma * self.sigma
  }
  fn exact_density(&self, x0: f64, xt: f64, _t0: f64, dt: f64) -> Option<f64> {
    let sig2 = self.sigma * self.sigma;
    let e = (-self.theta * dt).exp();
    let c = 2.0 * self.theta / (sig2 * (1.0 - e));
    let q = 2.0 * self.theta * self.mu / sig2 - 1.0;
    let u = c * x0.abs() * e;
    let v = c * xt.abs();

    if u < 1e-30 || v < 1e-30 {
      return Some(1e-30);
    }

    let sqrt_uv = (u * v).sqrt();
    let log_bessel = log_bessel_i(q, 2.0 * sqrt_uv);
    let log_density = c.ln() - u - v + 0.5 * q * (v / u).ln() + log_bessel;
    let density = log_density.exp();
    Some(if density.is_finite() && density > 0.0 {
      density
    } else {
      1e-30
    })
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

impl<S: SeedExt> DiffusionModel for ModifiedCIR<f64, S> {
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

impl<S: SeedExt> DiffusionModel for RadialOU<f64, S> {
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
