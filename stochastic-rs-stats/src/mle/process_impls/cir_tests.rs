use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::params::VarianceBoundary;

use crate::mle::DensityApprox;
use crate::mle::DiffusionModel;
use crate::mle::MleFit;
use crate::mle::fit_mle;
use crate::qmle::test_support::simulate_cir;

fn cir(theta: f64, mu: f64, sigma: f64) -> Cir<f64, Deterministic> {
  let floor = VarianceBoundary::Floor;
  Cir::new(
    theta,
    mu,
    sigma,
    2,
    None,
    None,
    floor,
    Deterministic::new(0),
  )
}

fn exact_fit(start: [f64; 3], path: &Array1<f64>, dt: f64) -> MleFit {
  let mut model = cir(start[0], start[1], start[2]);
  fit_mle(&mut model, path.view(), dt, DensityApprox::Exact, None).unwrap()
}

/// From `σ = 0.5` the search probes a `σ` where the exact density underflows; its log, which the
/// law forms itself, leads it to the optimum a start at the truth finds.
#[test]
fn an_exact_cir_fit_through_underflowing_densities_reaches_the_optimum() {
  let (truth, dt) = ([8.0, 0.03, 0.02], 1.0 / 12.0);
  for seed in [2u64, 11, 39] {
    let path = simulate_cir(8.0, 0.03, 0.02, 0.03, dt, 1000, seed);
    let near = exact_fit(truth, &path, dt);
    let far = exact_fit([1.0, 0.5, 0.5], &path, dt);
    assert!(near.converged && far.converged, "seed {seed}");
    assert!(
      (far.log_likelihood - near.log_likelihood).abs() < 1e-6,
      "seed {seed}: {} for {}",
      far.log_likelihood,
      near.log_likelihood
    );
    assert!(
      (far.params[0] / near.params[0] - 1.0).abs() < 1e-3,
      "seed {seed}: θ̂ = {} for {}",
      far.params[0],
      near.params[0]
    );
  }
}

/// From a state of zero the law is a Gamma of shape `2θμ/σ²`, whose density overflows its powers
/// from a shape near 80 and its Γ past 171: in logs the path fits.
#[test]
fn an_exact_cir_fit_of_a_path_from_zero_converges_at_a_large_shape() {
  let dt = 1.0 / 252.0;
  for truth in [[2.0, 0.2, 0.1], [5.0, 1.0, 0.2]] {
    let path = simulate_cir(truth[0], truth[1], truth[2], 0.0, dt, 500, 1);
    assert_eq!(path[0], 0.0);
    for start in [truth, [1.0, 0.05, 0.2]] {
      let fit = exact_fit(start, &path, dt);
      assert!(fit.converged, "law {truth:?} from {start:?}");
      assert!(
        (fit.params[2] / truth[2] - 1.0).abs() < 0.1,
        "law {truth:?} from {start:?}: σ̂ = {}",
        fit.params[2]
      );
    }
  }
}

/// Far in a tail the exact density is zero in doubles and its log the law's own, here against
/// mpmath's.
#[test]
fn the_exact_cir_log_density_keeps_the_tail_the_density_loses() {
  let (law, dt) = (cir(2.0, 0.04, 0.3), 1.0 / 252.0);
  for (x0, xt, reference) in [
    (0.05, 0.5, -1314.3),
    (0.05, 1.05, -3611.0),
    (4.0, 0.01, -20_125.9),
  ] {
    assert_eq!(law.exact_density(x0, xt, 0.0, dt), Some(0.0));
    let ln = DensityApprox::Exact.ln_density(&law, x0, xt, 0.0, dt);
    assert!((ln - reference).abs() < 0.05, "{x0} to {xt}: {ln}");
  }
}

/// `σ√x` steepens without bound towards zero, which no fixed difference step follows: below the
/// step Elerian's density is the Milstein law's, of the exact density's order.
#[test]
fn elerian_from_a_start_near_zero_is_of_the_exact_densitys_order() {
  let (law, dt) = (cir(2.0, 0.04, 0.3), 1.0 / 252.0);
  for x0 in [1e-12, 1e-9, 1e-7, 1e-6] {
    let slope = law.diffusion_x(x0, 0.0);
    assert!(
      (slope * x0.sqrt() / 0.15 - 1.0).abs() < 1e-15,
      "{x0}: {slope}"
    );
    for xt in [0.01, 0.051] {
      let elerian = DensityApprox::Elerian.ln_density(&law, x0, xt, 0.0, dt);
      let exact = DensityApprox::Exact.ln_density(&law, x0, xt, 0.0, dt);
      assert!(
        (elerian / exact - 1.0).abs() < 0.2,
        "{x0} to {xt}: {elerian} for {exact}"
      );
    }
  }
  let milstein = DensityApprox::Elerian.ln_density(&law, 1e-9, 0.05, 0.0, dt);
  assert!((milstein + 273.476).abs() < 1e-3, "{milstein}");
}
