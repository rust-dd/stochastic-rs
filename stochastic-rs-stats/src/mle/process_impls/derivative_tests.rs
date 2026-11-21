use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::cev::Cev;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::ckls::Ckls;
use stochastic_rs_stochastic::params::VarianceBoundary;

use crate::mle::DiffusionModel;

pub(super) fn boxed(model: impl DiffusionModel + 'static) -> Box<dyn DiffusionModel> {
  Box::new(model)
}

/// `σ|x|^γ` with `(σ, γ)`: CIR's root, a CEV and a CKLS above one, and CKLS at CIR's own root.
pub(super) fn power_diffusions() -> Vec<(&'static str, Box<dyn DiffusionModel>, f64, f64)> {
  let seed = || Deterministic::new(0);
  let floor = VarianceBoundary::Absorption;
  vec![
    (
      "Cir",
      boxed(Cir::new(2.0, 0.04, 0.3, 2, None, None, floor, seed())),
      0.3,
      0.5,
    ),
    (
      "Cev",
      boxed(Cev::new(0.05, 0.3, 0.7, 2, Some(1.0), None, seed())),
      0.3,
      0.7,
    ),
    (
      "Ckls",
      boxed(Ckls::new(0.1, -0.5, 0.2, 1.5, 2, Some(1.0), None, seed())),
      0.2,
      1.5,
    ),
    (
      "Ckls at a root",
      boxed(Ckls::new(0.1, -0.5, 0.3, 0.5, 2, Some(1.0), None, seed())),
      0.3,
      0.5,
    ),
  ]
}

/// The derivatives of `σ|x|^γ` are `σγ·sign(x)|x|^{γ−1}` and `σγ(γ−1)|x|^{γ−2}`, without bound at
/// zero below `γ = 1` and `γ = 2`: closed forms hold at `1e-12`, where a step of `1e-5` cannot.
#[test]
fn a_power_diffusions_derivatives_are_their_closed_forms() {
  for (name, model, sigma, gamma) in power_diffusions() {
    for x in [1e-12_f64, 1e-9, 1e-6, 1e-3, 0.05, 1.0, 7.5] {
      let first = sigma * gamma * x.powf(gamma - 1.0);
      let second = sigma * gamma * (gamma - 1.0) * x.powf(gamma - 2.0);
      for (at, sign) in [(x, 1.0), (-x, -1.0)] {
        let (got_x, got_xx) = (model.diffusion_x(at, 0.0), model.diffusion_xx(at, 0.0));
        assert!(
          (got_x / (sign * first) - 1.0).abs() < 1e-13,
          "{name} at {at}: σ′ = {got_x} for {}",
          sign * first
        );
        assert!(
          (got_xx / second - 1.0).abs() < 1e-13,
          "{name} at {at}: σ″ = {got_xx} for {second}"
        );
      }
    }
  }
}

/// They are the derivatives of `diffusion` itself: central differences with a step in proportion
/// to the state, which follow the power to any depth, agree at every state.
#[test]
fn a_power_diffusions_derivatives_are_those_of_its_diffusion() {
  for (name, model, _, _) in power_diffusions() {
    for x in [1e-12_f64, 1e-9, 1e-6, 1e-3, 0.05, 1.0, 7.5] {
      let at = |r: f64| model.diffusion(x * (1.0 + r), 0.0);
      let slope = (at(1e-5) - at(-1e-5)) / (2e-5 * x);
      let curve = (at(1e-3) - 2.0 * at(0.0) + at(-1e-3)) / (1e-3 * x).powi(2);
      let (got_x, got_xx) = (model.diffusion_x(x, 0.0), model.diffusion_xx(x, 0.0));
      assert!(
        (got_x / slope - 1.0).abs() < 1e-8,
        "{name} at {x}: σ′ = {got_x} for {slope}"
      );
      assert!(
        (got_xx / curve - 1.0).abs() < 1e-5,
        "{name} at {x}: σ″ = {got_xx} for {curve}"
      );
    }
  }
}
