//! # Maximum Likelihood Estimation for 1-D Diffusions
//!
//! $$
//! \hat\theta = \arg\max_\theta \sum_{i=1}^{N} \ln p(X_{t_i} \mid X_{t_{i-1}};\theta,\Delta t)
//! $$
//!
//! This module implements MLE-based parameter estimation for univariate SDEs of the form
//!
//! $$
//! dX_t = \mu(X_t, t;\theta)\,dt + \sigma(X_t, t;\theta)\,dW_t
//! $$
//!
//! using several transition density approximations (Euler, Ozaki, Shoji-Ozaki, Elerian,
//! Kessler, exact, and Aït-Sahalia).

mod density;
mod fit;
mod process_impls;

pub use density::DensityApprox;
pub use fit::MleResult;
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

  /// Exact transition density p(x0 -> xt | dt). Return `None` if not available.
  fn exact_density(&self, _x0: f64, _xt: f64, _t0: f64, _dt: f64) -> Option<f64> {
    None
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
mod tests {
  use super::*;
  use crate::stochastic::diffusion::cir::CIR;
  use crate::stochastic::diffusion::gbm::GBM;
  use crate::stochastic::diffusion::ou::OU;
  use crate::traits::ProcessExt;

  #[test]
  fn euler_density_positive() {
    let ou = OU::seeded(1.5, 0.5, 0.2, 100, None, Some(1.0), 0);
    let d = DensityApprox::Euler.density(&ou, 0.3, 0.35, 0.0, 0.01);
    assert!(d > 0.0);
    assert!(d.is_finite());
  }

  #[test]
  fn all_densities_positive_for_ou() {
    let ou = OU::seeded(1.5, 0.5, 0.2, 100, None, Some(1.0), 0);
    let densities = [
      DensityApprox::Exact,
      DensityApprox::Euler,
      DensityApprox::Ozaki,
      DensityApprox::ShojiOzaki,
      DensityApprox::Elerian,
      DensityApprox::Kessler,
    ];
    for dens in &densities {
      let d = dens.density(&ou, 0.5, 0.55, 0.0, 0.01);
      assert!(
        d > 0.0 && d.is_finite(),
        "{:?} density is not valid: {}",
        dens,
        d
      );
    }
  }

  #[test]
  fn mle_gbm_via_process_ext() {
    let gbm = GBM::seeded(0.05, 0.2, 2501, Some(100.0), Some(10.0), 99);
    let path = gbm.sample();
    let dt = 10.0 / 2500.0;

    let mut gbm_fit = GBM::seeded(0.0, 0.5, 100, Some(100.0), Some(1.0), 0);
    let result = fit_mle(&mut gbm_fit, &path, dt, DensityApprox::Euler, None);

    assert!(
      (result.params[1] - 0.2).abs() < 0.15,
      "sigma estimate too far: {} vs 0.2",
      result.params[1]
    );
  }

  #[test]
  fn mle_ou_euler_via_process_ext() {
    let ou = OU::seeded(2.0, 1.0, 0.3, 2501, Some(1.0), Some(10.0), 123);
    let path = ou.sample();
    let dt = 10.0 / 2500.0;

    let mut ou_fit = OU::seeded(1.0, 0.5, 0.5, 100, Some(1.0), Some(1.0), 0);
    let result = fit_mle(&mut ou_fit, &path, dt, DensityApprox::Euler, None);

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
    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
  }

  #[test]
  fn mle_ou_exact_via_process_ext() {
    let ou = OU::seeded(2.0, 1.0, 0.3, 2501, Some(1.0), Some(10.0), 77);
    let path = ou.sample();
    let dt = 10.0 / 2500.0;

    let mut ou_fit = OU::seeded(1.0, 0.5, 0.5, 100, Some(1.0), Some(1.0), 0);
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
  fn mle_cir_euler_via_process_ext() {
    let cir = CIR::seeded(2.0, 0.04, 0.1, 5001, Some(0.04), Some(20.0), None, 55);
    let path = cir.sample();
    let dt = 20.0 / 5000.0;

    let mut cir_fit = CIR::seeded(1.0, 0.05, 0.2, 100, Some(0.04), Some(1.0), None, 0);
    let result = fit_mle(&mut cir_fit, &path, dt, DensityApprox::Euler, None);

    assert!(
      (result.params[1] - 0.04).abs() < 0.03,
      "mu estimate too far: {} vs 0.04",
      result.params[1]
    );
    assert!(result.log_likelihood.is_finite());
  }

  #[test]
  fn mle_ou_shoji_ozaki_via_process_ext() {
    let ou = OU::seeded(2.0, 1.0, 0.3, 2501, Some(1.0), Some(10.0), 200);
    let path = ou.sample();
    let dt = 10.0 / 2500.0;

    let mut ou_fit = OU::seeded(1.0, 0.5, 0.5, 100, Some(1.0), Some(1.0), 0);
    let result = fit_mle(&mut ou_fit, &path, dt, DensityApprox::ShojiOzaki, None);

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
  fn mle_ou_kessler_via_process_ext() {
    let ou = OU::seeded(2.0, 1.0, 0.3, 2501, Some(1.0), Some(10.0), 300);
    let path = ou.sample();
    let dt = 10.0 / 2500.0;

    let mut ou_fit = OU::seeded(1.0, 0.5, 0.5, 100, Some(1.0), Some(1.0), 0);
    let result = fit_mle(&mut ou_fit, &path, dt, DensityApprox::Kessler, None);

    assert!(
      (result.params[2] - 0.3).abs() < 0.15,
      "sigma estimate too far: {} vs 0.3",
      result.params[2]
    );
  }

  #[test]
  fn density_cir_euler_reference() {
    let cir = CIR::seeded(3.0, 0.3, 0.2, 100, Some(0.4), Some(1.0), None, 0);
    let dt = 1.0 / 250.0;
    let d = DensityApprox::Euler.density(&cir, 0.4, 0.41, 0.0, dt);
    // reference value: 18.715933204468332
    assert!(
      (d - 18.715933204468332).abs() < 1e-8,
      "CIR Euler: got {d}, expected 18.715933204468332"
    );
  }

  #[test]
  fn density_cir_kessler_reference() {
    let cir = CIR::seeded(3.0, 0.3, 0.2, 100, Some(0.4), Some(1.0), None, 0);
    let dt = 1.0 / 250.0;
    let d = DensityApprox::Kessler.density(&cir, 0.4, 0.41, 0.0, dt);
    // reference value: 18.734374214427948
    assert!(
      (d - 18.734374214427948).abs() < 1e-6,
      "CIR Kessler: got {d}, expected 18.734374214427948"
    );
  }

  #[test]
  fn density_ou_exact_reference() {
    let ou = OU::seeded(2.0, 1.0, 0.3, 100, Some(1.0), Some(1.0), 0);
    let d = DensityApprox::Exact.density(&ou, 0.5, 0.55, 0.0, 0.01);
    // reference value: 5.399419276877125
    assert!(
      (d - 5.399419276877125).abs() < 1e-8,
      "OU Exact: got {d}, expected 5.399419276877125"
    );
  }

  #[test]
  fn density_ou_euler_reference() {
    let ou = OU::seeded(2.0, 1.0, 0.3, 100, Some(1.0), Some(1.0), 0);
    let d = DensityApprox::Euler.density(&ou, 0.5, 0.55, 0.0, 0.01);
    // reference value: 5.467002489199778
    assert!(
      (d - 5.467002489199778).abs() < 1e-8,
      "OU Euler: got {d}, expected 5.467002489199778"
    );
  }

  #[test]
  fn density_ou_shoji_ozaki_reference() {
    let ou = OU::seeded(2.0, 1.0, 0.3, 100, Some(1.0), Some(1.0), 0);
    let d = DensityApprox::ShojiOzaki.density(&ou, 0.5, 0.55, 0.0, 0.01);
    // reference value: 5.399419278094993
    assert!(
      (d - 5.399419278094993).abs() < 1e-6,
      "OU ShojiOzaki: got {d}, expected 5.399419278094993"
    );
  }

  #[test]
  fn density_ou_kessler_reference() {
    let ou = OU::seeded(2.0, 1.0, 0.3, 100, Some(1.0), Some(1.0), 0);
    let d = DensityApprox::Kessler.density(&ou, 0.5, 0.55, 0.0, 0.01);
    // reference value: 5.447446973872427
    assert!(
      (d - 5.447446973872427).abs() < 1e-6,
      "OU Kessler: got {d}, expected 5.447446973872427"
    );
  }

  #[test]
  fn cir_kessler_mle() {
    let cir = CIR::seeded(3.0, 0.3, 0.2, 1251, Some(0.4), Some(5.0), None, 42);
    let path = cir.sample();
    let dt = 5.0 / 1250.0;

    let mut cir_fit = CIR::seeded(1.0, 0.5, 0.3, 100, Some(0.4), Some(1.0), None, 0);
    let result = fit_mle(&mut cir_fit, &path, dt, DensityApprox::Kessler, None);

    assert!(
      (result.params[0] - 3.0).abs() < 3.0,
      "kappa estimate: {} vs 3.0",
      result.params[0]
    );
    assert!(
      (result.params[1] - 0.3).abs() < 0.15,
      "mu estimate: {} vs 0.3",
      result.params[1]
    );
    assert!(
      (result.params[2] - 0.2).abs() < 0.1,
      "sigma estimate: {} vs 0.2",
      result.params[2]
    );
    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
  }

  #[test]
  fn ou_all_densities_agree() {
    let ou = OU::seeded(2.0, 1.0, 0.3, 5001, Some(1.0), Some(10.0), 77);
    let path = ou.sample();
    let dt = 10.0 / 5000.0;

    let densities = [
      ("Exact", DensityApprox::Exact),
      ("Euler", DensityApprox::Euler),
      ("Ozaki", DensityApprox::Ozaki),
      ("ShojiOzaki", DensityApprox::ShojiOzaki),
      ("Kessler", DensityApprox::Kessler),
    ];

    let mut results = Vec::new();
    for (name, dens) in &densities {
      let mut ou_fit = OU::seeded(1.0, 0.5, 0.5, 100, Some(1.0), Some(1.0), 0);
      let result = fit_mle(&mut ou_fit, &path, dt, *dens, None);
      results.push((*name, result));
    }

    for (name, result) in &results {
      assert!(
        (result.params[1] - 1.0).abs() < 0.5,
        "{name}: mu estimate too far: {} vs 1.0",
        result.params[1]
      );
      assert!(
        (result.params[2] - 0.3).abs() < 0.1,
        "{name}: sigma estimate too far: {} vs 0.3",
        result.params[2]
      );
    }
  }

  #[test]
  fn mle_result_display() {
    use ndarray::array;
    let result = MleResult {
      params: array![1.0, 2.0, 0.3],
      param_names: vec!["kappa".into(), "mu".into(), "sigma".into()],
      log_likelihood: -100.0,
      sample_size: 1000,
      aic: 206.0,
      bic: 220.0,
    };
    let s = format!("{}", result);
    assert!(s.contains("kappa"));
    assert!(s.contains("mu"));
    assert!(s.contains("sigma"));
    assert!(s.contains("log-lik"));
  }
}
