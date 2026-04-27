use ndarray::Array1;
use ndarray::array;

use super::DiffusionModel;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_stochastic::diffusion::ait_sahalia::AitSahalia;
use stochastic_rs_stochastic::diffusion::cev::Cev;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::ckls::Ckls;
use stochastic_rs_stochastic::diffusion::feller::FellerLogistic;
use stochastic_rs_stochastic::diffusion::feller_root::FellerRoot;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::diffusion::gbm_ih::GbmIh;
use stochastic_rs_stochastic::diffusion::gompertz::Gompertz;
use stochastic_rs_stochastic::diffusion::hyperbolic::Hyperbolic;
use stochastic_rs_stochastic::diffusion::hyperbolic2::Hyperbolic2;
use stochastic_rs_stochastic::diffusion::jacobi::Jacobi;
use stochastic_rs_stochastic::diffusion::kimura::Kimura;
use stochastic_rs_stochastic::diffusion::linear_sde::LinearSDE;
use stochastic_rs_stochastic::diffusion::logistic::Logistic;
use stochastic_rs_stochastic::diffusion::modified_cir::ModifiedCIR;
use stochastic_rs_stochastic::diffusion::nonlinear_sde::NonLinearSDE;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::diffusion::pearson::Pearson;
use stochastic_rs_stochastic::diffusion::quadratic::Quadratic;
use stochastic_rs_stochastic::diffusion::radial_ou::RadialOU;
use stochastic_rs_stochastic::diffusion::three_half::ThreeHalf;
use stochastic_rs_stochastic::diffusion::verhulst::Verhulst;
use stochastic_rs_stochastic::process::bm::Bm;

fn log_bessel_i(nu: f64, z: f64) -> f64 {
  use statrs::function::gamma::ln_gamma;

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
    Some(super::density::gaussian_pdf(xt, x0, dt))
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
    if xt <= 0.0 || x0 <= 0.0 {
      return Some(1e-30);
    }
    let log_mean = x0.ln() + (self.mu - 0.5 * self.sigma * self.sigma) * dt;
    let log_var = self.sigma * self.sigma * dt;
    let z = (xt.ln() - log_mean) / log_var.sqrt();
    Some((-0.5 * z * z).exp() / (xt * (2.0 * PI * log_var).sqrt()))
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
    use std::f64::consts::PI;
    let e = (-self.theta * dt).exp();
    let mean = self.mu + (x0 - self.mu) * e;
    let var = self.sigma * self.sigma / (2.0 * self.theta) * (1.0 - e * e);
    if var <= 0.0 {
      return Some(1e-30);
    }
    let z = (xt - mean) / var.sqrt();
    Some((-0.5 * z * z).exp() / (2.0 * PI * var).sqrt())
  }
  fn exact_step(&self, _t: f64, dt: f64, x: f64, dz: f64) -> Option<f64> {
    let e = (-self.theta * dt).exp();
    let mean = self.mu + (x - self.mu) * e;
    let var = self.sigma * self.sigma / (2.0 * self.theta) * (1.0 - e * e);
    Some(mean + var.sqrt() * dz)
  }
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
  fn is_positive(&self) -> bool {
    true
  }
}

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
    let safe_x = if x.abs() < 1e-12 { 1e-12 } else { x };
    (self.a - self.b * safe_x.abs().ln()) * safe_x
  }
  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    self.sigma * x
  }
  fn is_positive(&self) -> bool {
    true
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

impl<S: SeedExt> DiffusionModel for LinearSDE<f64, S> {
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
}

impl<S: SeedExt> DiffusionModel for NonLinearSDE<f64, S> {
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
    let safe_x = if x.abs() < 1e-12 { 1e-12 } else { x };
    self.am1 / safe_x + self.a0 + self.a1 * x + self.a2 * x * x
  }
  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    self.b0 + self.b1 * x + self.b2 * x.abs().powf(self.b3)
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
    let safe_x = if x.abs() < 1e-12 { 1e-12 } else { x };
    self.am1 / safe_x + self.a0 + self.a1 * x + self.a2 * x * x
  }
  fn diffusion(&self, x: f64, _t: f64) -> f64 {
    let inner = self.b0 + self.b1 * x + self.b2 * x.abs().powf(self.b3);
    inner.abs().sqrt()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::mle::DensityApprox;
  use crate::mle::fit_mle;
  use crate::traits::ProcessExt;

  #[test]
  fn gbm_process_ext_with_mle() {
    let gbm = Gbm::seeded(0.05, 0.2, 2501, Some(100.0), Some(10.0), 42);
    let path = gbm.sample();
    assert_eq!(path.len(), 2501);

    let dt = 10.0 / 2500.0;
    let mut gbm_fit = Gbm::seeded(0.0, 0.5, 100, Some(100.0), Some(1.0), 0);
    let result = fit_mle(&mut gbm_fit, &path, dt, DensityApprox::Euler, None);
    assert!(
      (result.params[1] - 0.2).abs() < 0.15,
      "sigma estimate too far: {} vs 0.2",
      result.params[1]
    );
  }

  #[test]
  fn ou_process_ext_with_mle() {
    let ou = Ou::seeded(2.0, 1.0, 0.3, 2501, Some(1.0), Some(10.0), 123);
    let path = ou.sample();
    assert_eq!(path.len(), 2501);

    let dt = 10.0 / 2500.0;
    let mut ou_fit = Ou::seeded(1.0, 0.5, 0.5, 100, Some(1.0), Some(1.0), 0);
    let result = fit_mle(&mut ou_fit, &path, dt, DensityApprox::Exact, None);
    assert!(
      (result.params[1] - 1.0).abs() < 0.5,
      "mu estimate too far: {} vs 1.0",
      result.params[1]
    );
    assert!(
      (result.params[2] - 0.3).abs() < 0.15,
      "sigma estimate too far: {} vs 0.3",
      result.params[2]
    );
  }

  #[test]
  fn cir_process_ext_with_mle() {
    let cir = Cir::seeded(2.0, 0.04, 0.1, 5001, Some(0.04), Some(20.0), None, 55);
    let path = cir.sample();
    assert_eq!(path.len(), 5001);

    let dt = 20.0 / 5000.0;
    let mut cir_fit = Cir::seeded(1.0, 0.05, 0.2, 100, Some(0.04), Some(1.0), None, 0);
    let result = fit_mle(&mut cir_fit, &path, dt, DensityApprox::Euler, None);
    assert!(
      (result.params[1] - 0.04).abs() < 0.03,
      "mu estimate too far: {} vs 0.04",
      result.params[1]
    );
  }

  #[test]
  fn ou_sample_then_mle_roundtrip() {
    let ou = Ou::seeded(3.0, 2.0, 0.5, 10001, Some(2.0), Some(10.0), 77);
    let path: Array1<f64> = ProcessExt::<f64>::sample(&ou);

    let dt = 10.0 / 10000.0;
    let mut ou_fit = Ou::seeded(1.0, 1.0, 1.0, 100, Some(1.0), Some(1.0), 0);
    let result = fit_mle(&mut ou_fit, &path, dt, DensityApprox::Exact, None);

    assert!(
      (result.params[0] - 3.0).abs() < 2.0,
      "theta estimate: {} vs 3.0",
      result.params[0]
    );
    assert!(
      (result.params[1] - 2.0).abs() < 0.5,
      "mu estimate: {} vs 2.0",
      result.params[1]
    );
    assert!(
      (result.params[2] - 0.5).abs() < 0.2,
      "sigma estimate: {} vs 0.5",
      result.params[2]
    );
  }

  #[test]
  fn cev_process_ext_with_mle() {
    let cev = Cev::seeded(0.05, 0.3, 0.7, 2501, Some(100.0), Some(10.0), 42);
    let path = cev.sample();
    assert_eq!(path.len(), 2501);

    let dt = 10.0 / 2500.0;
    let mut cev_fit = Cev::seeded(0.0, 0.5, 0.5, 100, Some(100.0), Some(1.0), 0);
    let result = fit_mle(&mut cev_fit, &path, dt, DensityApprox::Euler, None);

    assert!(
      (result.params[1] - 0.3).abs() < 0.2,
      "sigma estimate too far: {} vs 0.3",
      result.params[1]
    );
    assert!(
      (result.params[2] - 0.7).abs() < 0.5,
      "gamma estimate too far: {} vs 0.7",
      result.params[2]
    );
  }
}
