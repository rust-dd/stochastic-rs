use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::ait_sahalia::AitSahalia;
use stochastic_rs_stochastic::diffusion::feller_logistic::FellerLogistic;
use stochastic_rs_stochastic::diffusion::feller_root::FellerRoot;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::diffusion::gbm_ih::GbmIh;
use stochastic_rs_stochastic::diffusion::gompertz::Gompertz;
use stochastic_rs_stochastic::diffusion::hyperbolic::Hyperbolic;
use stochastic_rs_stochastic::diffusion::hyperbolic2::Hyperbolic2;
use stochastic_rs_stochastic::diffusion::jacobi::Jacobi;
use stochastic_rs_stochastic::diffusion::kimura::Kimura;
use stochastic_rs_stochastic::diffusion::linear_sde::LinearSde;
use stochastic_rs_stochastic::diffusion::logistic::Logistic;
use stochastic_rs_stochastic::diffusion::modified_cir::ModifiedCir;
use stochastic_rs_stochastic::diffusion::nonlinear_sde::NonlinearSde;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::diffusion::pearson::Pearson;
use stochastic_rs_stochastic::diffusion::quadratic::Quadratic;
use stochastic_rs_stochastic::diffusion::radial_ou::RadialOu;
use stochastic_rs_stochastic::diffusion::three_half::ThreeHalf;
use stochastic_rs_stochastic::diffusion::verhulst::Verhulst;
use stochastic_rs_stochastic::params::VarianceBoundary;

use super::derivative_tests::boxed;
use super::derivative_tests::power_diffusions;
use crate::mle::DiffusionModel;

/// `[μ′, μ″, σ′, σ″]` at a state.
type Closed = fn(f64) -> [f64; 4];

/// The model of `derivative_tests`' row `name`, so both tables test one set of parameters.
fn power(name: &str) -> Box<dyn DiffusionModel> {
  let row = power_diffusions().into_iter().find(|row| row.0 == name);
  row.expect("a power diffusion's row").1
}

/// The derivatives of `c√|g|` from `g`, `g′` and `g″`.
fn root(c: f64, g: f64, g1: f64, g2: f64) -> [f64; 2] {
  let r = g.abs().sqrt();
  let first = 0.5 * c * g.signum() * g1 / r;
  [
    first,
    0.5 * c * g.signum() * g2 / r - 0.25 * c * g1 * g1 / (r * g.abs()),
  ]
}

/// Every implementor that is not flat in the state, with its derivatives written out by hand.
fn models() -> Vec<(&'static str, Box<dyn DiffusionModel>, Closed)> {
  let seed = || Deterministic::new(0);
  let absorption = VarianceBoundary::Absorption;
  let one = Some(1.0);
  vec![
    (
      "GbmIh",
      boxed(GbmIh::new(0.05, 0.2, 2, one, None, None, seed())),
      |_| [0.05, 0.0, 0.2, 0.0],
    ),
    (
      "Gbm",
      boxed(Gbm::new(0.05, 0.2, 2, 1.0, None, seed())),
      |_| [0.05, 0.0, 0.2, 0.0],
    ),
    (
      "Ou",
      boxed(Ou::new(2.0, 1.0, 0.3, 2, None, None, seed())),
      |_| [-2.0, 0.0, 0.0, 0.0],
    ),
    ("Cir", power("Cir"), |x| {
      [-2.0, 0.0, 0.15 / x.sqrt(), -0.075 / (x * x.sqrt())]
    }),
    ("Cev", power("Cev"), |x| {
      [0.05, 0.0, 0.21 * x.powf(-0.3), -0.063 * x.powf(-1.3)]
    }),
    ("Ckls", power("Ckls"), |x| {
      [-0.5, 0.0, 0.3 * x.sqrt(), 0.15 / x.sqrt()]
    }),
    (
      "ModifiedCir",
      boxed(ModifiedCir::new(1.5, 0.4, 2, one, None, seed())),
      |x| {
        let r = (1.0 + x * x).sqrt();
        [-1.5, 0.0, 0.4 * x / r, 0.4 / (r * r * r)]
      },
    ),
    (
      "RadialOu",
      boxed(RadialOu::new(0.7, 0.4, 2, one, None, seed())),
      |x| [-0.7 / (x * x) - 1.0, 1.4 / (x * x * x), 0.0, 0.0],
    ),
    (
      "Jacobi",
      boxed(Jacobi::new(0.5, 1.2, 0.4, 2, Some(0.5), None, seed())),
      |x| {
        let [s1, s2] = root(0.4, x * (1.0 - x), 1.0 - 2.0 * x, -2.0);
        [-1.2, 0.0, s1, s2]
      },
    ),
    (
      "Verhulst",
      boxed(Verhulst::new(1.5, 2.0, 0.3, 2, one, None, false, seed())),
      |x| [1.5 * (1.0 - x), -1.5, 0.3, 0.0],
    ),
    (
      "FellerLogistic",
      boxed(FellerLogistic::new(
        1.5,
        0.8,
        0.3,
        2,
        one,
        None,
        absorption,
        seed(),
      )),
      |x| {
        [
          1.5 * (0.8 - 2.0 * x),
          -3.0,
          0.15 / x.sqrt(),
          -0.075 / (x * x.sqrt()),
        ]
      },
    ),
    (
      "Logistic",
      boxed(Logistic::new(0.5, 0.3, 2, one, None, seed())),
      |x| [1.0 - x, -1.0, 0.3, 0.0],
    ),
    (
      "ThreeHalf",
      boxed(ThreeHalf::new(1.5, 0.8, 0.3, 2, one, None, seed())),
      |x| {
        [
          1.5 * (0.8 - 2.0 * x),
          -3.0,
          0.45 * x.sqrt(),
          0.225 / x.sqrt(),
        ]
      },
    ),
    (
      "FellerRoot",
      boxed(FellerRoot::new(0.5, 0.3, 0.4, 2, one, None, seed())),
      |x| [0.5 + 0.172 * x, 0.172, 0.6 * x.sqrt(), 0.3 / x.sqrt()],
    ),
    (
      "Kimura",
      boxed(Kimura::new(0.5, 0.4, 2, Some(0.5), None, seed())),
      |x| {
        let [s1, s2] = root(0.4, x * (1.0 - x), 1.0 - 2.0 * x, -2.0);
        [0.5 * (1.0 - 2.0 * x), -1.0, s1, s2]
      },
    ),
    (
      "Gompertz",
      boxed(Gompertz::new(0.5, 0.3, 0.2, 2, one, None, seed())),
      |x| [0.2 - 0.3 * x.ln(), -0.3 / x, 0.2, 0.0],
    ),
    (
      "Quadratic",
      boxed(Quadratic::new(0.1, -0.5, 0.2, 0.3, 2, one, None, seed())),
      |x| [-0.5 + 0.4 * x, 0.4, 0.3, 0.0],
    ),
    (
      "Pearson",
      boxed(Pearson::new(1.5, 0.5, 0.2, 0.1, 0.3, 2, one, None, seed())),
      |x| {
        let g = 3.0 * (0.2 * x * x + 0.1 * x + 0.3);
        let [s1, s2] = root(1.0, g, 3.0 * (0.4 * x + 0.1), 1.2);
        [-1.5, 0.0, s1, s2]
      },
    ),
    (
      "Hyperbolic",
      boxed(Hyperbolic::new(1.5, 0.3, 2, one, None, seed())),
      |x| {
        let q = 1.0 + x * x;
        [-1.5 / q.powf(1.5), 4.5 * x / q.powf(2.5), 0.0, 0.0]
      },
    ),
    (
      "Hyperbolic2",
      boxed(Hyperbolic2::new(
        0.2,
        1.0,
        0.5,
        0.3,
        0.4,
        2,
        one,
        None,
        seed(),
      )),
      |x| {
        let u = x - 0.3;
        let r = (0.25 + u * u).sqrt();
        let first = 0.25 / r.powi(3);
        let second = -0.75 * u / r.powi(5);
        [-0.08 * first, -0.08 * second, 0.0, 0.0]
      },
    ),
    (
      "LinearSde",
      boxed(LinearSde::new(0.1, -0.5, 0.3, 2, one, None, seed())),
      |_| [-0.5, 0.0, 0.3, 0.0],
    ),
    (
      "NonlinearSde",
      boxed(NonlinearSde::new(
        0.05,
        0.1,
        -0.5,
        0.2,
        0.1,
        0.05,
        0.2,
        1.5,
        2,
        one,
        None,
        seed(),
      )),
      |x| {
        [
          -0.05 / (x * x) - 0.5 + 0.4 * x,
          0.1 / (x * x * x) + 0.4,
          0.05 + 0.3 * x.sqrt(),
          0.15 / x.sqrt(),
        ]
      },
    ),
    (
      "AitSahalia",
      boxed(AitSahalia::new(
        0.05,
        0.1,
        -0.5,
        0.2,
        0.1,
        0.05,
        0.2,
        1.5,
        2,
        one,
        None,
        seed(),
      )),
      |x| {
        let g = 0.1 + 0.05 * x + 0.2 * x.powf(1.5);
        let [s1, s2] = root(1.0, g, 0.05 + 0.3 * x.sqrt(), 0.15 / x.sqrt());
        [
          -0.05 / (x * x) - 0.5 + 0.4 * x,
          0.1 / (x * x * x) + 0.4,
          s1,
          s2,
        ]
      },
    ),
  ]
}

fn derivatives(model: &dyn DiffusionModel, x: f64) -> [f64; 4] {
  [
    model.drift_x(x, 0.0),
    model.drift_xx(x, 0.0),
    model.diffusion_x(x, 0.0),
    model.diffusion_xx(x, 0.0),
  ]
}

/// Each derivative is its closed form, near zero and at ordinary states alike; a fixed difference
/// step of `1e-5` turns a linear drift's second into rounding noise and misses a root's below it.
#[test]
fn every_implementors_derivatives_are_their_closed_forms() {
  for (name, model, closed) in models() {
    for x in [1e-9, 1e-6, 1e-3, 0.05, 0.4, 2.5] {
      let (got, want) = (derivatives(&*model, x), closed(x));
      for (k, what) in ["μ′", "μ″", "σ′", "σ″"].into_iter().enumerate() {
        assert!(
          (got[k] - want[k]).abs() <= 1e-10 * want[k].abs(),
          "{name} at {x}: {what} = {} for {}",
          got[k],
          want[k]
        );
      }
    }
  }
}

/// They are the derivatives of `drift` and `diffusion` themselves: at states a difference
/// resolves, one with a step in proportion to the state or one with a fixed step agrees.
#[test]
fn every_implementors_derivatives_are_those_of_its_coefficients() {
  let slope = |f: &dyn Fn(f64) -> f64, x: f64, h: f64| (f(x + h) - f(x - h)) / (2.0 * h);
  let curve = |f: &dyn Fn(f64) -> f64, x: f64, h: f64| (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
  for (name, model, _) in models() {
    let drift = |x: f64| model.drift(x, 0.0);
    let diffusion = |x: f64| model.diffusion(x, 0.0);
    for x in [1e-3, 0.05, 0.4, 2.5] {
      let got = derivatives(&*model, x);
      let estimates = [
        [slope(&drift, x, 1e-5 * x), slope(&drift, x, 1e-6)],
        [curve(&drift, x, 1e-3 * x), curve(&drift, x, 1e-4)],
        [slope(&diffusion, x, 1e-5 * x), slope(&diffusion, x, 1e-6)],
        [curve(&diffusion, x, 1e-3 * x), curve(&diffusion, x, 1e-4)],
      ];
      for (k, what) in ["μ′", "μ″", "σ′", "σ″"].into_iter().enumerate() {
        let size = 1.0 + drift(x).abs() + diffusion(x).abs();
        let tol = [1e-7, 1e-3][k % 2] * got[k].abs() + 1e-6 * size;
        assert!(
          estimates[k].iter().any(|e| (got[k] - e).abs() <= tol),
          "{name} at {x}: {what} = {} for {:?}",
          got[k],
          estimates[k]
        );
      }
    }
  }
}
