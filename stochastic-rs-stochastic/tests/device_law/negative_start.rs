//! A start below zero reads `f₃(x₀)` at the first point and leaves `f₁(x₀)`, on the device as on
//! the host: the first two points, which no draw reaches when the start reads as zero.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::correlation::heston_stoch_corr::HestonStochCorr;
use stochastic_rs_stochastic::diffusion::bessel::Bessel;
use stochastic_rs_stochastic::diffusion::bessel::SquaredBessel;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::fcir::Fcir;
use stochastic_rs_stochastic::diffusion::feller_logistic::FellerLogistic;
use stochastic_rs_stochastic::interest::cir_2f::Cir2F;
use stochastic_rs_stochastic::interest::cir_pp::CirPlusPlus;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::Device;

const KAPPA: f32 = 0.4;
const THETA: f32 = 0.2;
const SIGMA: f32 = 0.6;
const T: f32 = 5.0;
const N: usize = 61;
const DT: f32 = T / (N - 1) as f32;
const FT: VarianceBoundary = VarianceBoundary::FullTruncation;
const ABSORPTION: VarianceBoundary = VarianceBoundary::Absorption;
const REFLECT: VarianceBoundary = VarianceBoundary::Reflect;
/// A start below zero that one drift step of `KAPPA * THETA * DT` lifts above it.
const BELOW: f32 = -0.001;

/// The report of the start, then one drift step, of a host path and a device path.
fn first_points<H, D>(host: H, device: D) -> [[f32; 2]; 2]
where
  H: ProcessExt<f32, Output = Array1<f32>>,
  D: ProcessExt<f32, Output = Array1<f32>>,
{
  [host.sample(), device.sample()].map(|path| [path[0], path[1]])
}

fn close(got: f32, want: f32, what: &str) {
  assert!(
    (got - want).abs() <= 1e-6 * want.abs(),
    "{what}: {got} against {want}"
  );
}

#[test]
fn a_start_below_zero_reads_the_same_on_host_and_device() {
  let seed = || Deterministic::new(89);
  let phi = (|_| 0.0) as fn(f32) -> f32;
  let drift = KAPPA * THETA * DT;
  for (boundary, start, next) in [
    (FT, 0.0, Some(BELOW + drift)),
    (ABSORPTION, 0.0, Some(drift)),
    (REFLECT, -BELOW, None),
  ] {
    let check = |what: &str, [host, device]: [[f32; 2]; 2], next: Option<f32>| {
      assert_eq!(
        [host[0], device[0]],
        [start; 2],
        "{what} under {boundary:?}"
      );
      if let Some(next) = next {
        close(host[1], next, what);
        close(device[1], next, what);
      }
    };
    let cir = || {
      Cir::<f32, _>::new(
        KAPPA,
        THETA,
        SIGMA,
        N,
        Some(BELOW),
        Some(T),
        boundary,
        seed(),
      )
    };
    check("Cir", first_points(cir(), cir().on::<Device>()), next);
    let shifted = || {
      CirPlusPlus::<f32, _>::new(
        KAPPA,
        THETA,
        SIGMA,
        phi,
        N,
        Some(BELOW),
        Some(T),
        boundary,
        seed(),
      )
    };
    check(
      "CirPlusPlus",
      first_points(shifted(), shifted().on::<Device>()),
      next,
    );
    let fractional = || {
      Fcir::<f32, _>::new(
        0.7,
        KAPPA,
        THETA,
        SIGMA,
        N,
        Some(BELOW),
        Some(T),
        boundary,
        seed(),
      )
    };
    check(
      "Fcir",
      first_points(fractional(), fractional().on::<Device>()),
      next,
    );
    let delta = 0.5;
    let squared = || SquaredBessel::<f32, _>::new(delta, N, Some(BELOW), Some(T), boundary, seed());
    let lifted = next.map(|next| next - drift + delta * DT);
    check(
      "SquaredBessel",
      first_points(squared(), squared().on::<Device>()),
      lifted,
    );
    let logistic = || {
      FellerLogistic::<f32, _>::new(
        KAPPA,
        THETA,
        SIGMA,
        N,
        Some(BELOW),
        Some(T),
        boundary,
        seed(),
      )
    };
    let stuck = next.map(|_| 0.0);
    check(
      "FellerLogistic",
      first_points(logistic(), logistic().on::<Device>()),
      stuck,
    );
  }
}

#[test]
fn a_two_factor_start_below_zero_reads_the_same_on_host_and_device() {
  let factor = |boundary| {
    let seed = Deterministic::new(1);
    Cir::<f32, _>::new(KAPPA, THETA, SIGMA, N, Some(BELOW), Some(T), boundary, seed)
  };
  let phi = (|_| 0.0) as fn(f32) -> f32;
  let drift = KAPPA * THETA * DT;
  for (x, y, start, next) in [
    (FT, ABSORPTION, 0.0, Some(BELOW + 2.0 * drift)),
    (ABSORPTION, ABSORPTION, 0.0, Some(2.0 * drift)),
    (FT, REFLECT, -BELOW, None),
    (REFLECT, REFLECT, -2.0 * BELOW, None),
  ] {
    let build = || Cir2F::<f32, _>::new(factor(x), factor(y), phi, Deterministic::new(97));
    let [host, device] = first_points(build(), build().on::<Device>());
    assert_eq!([host[0], device[0]], [start; 2], "{x:?} and {y:?}");
    if let Some(next) = next {
      close(host[1], next, "Cir2F, host");
      close(device[1], next, "Cir2F, device");
    }
  }
}

#[test]
fn a_radius_and_a_boundary_less_variance_below_zero_read_the_same_on_host_and_device() {
  let seed = || Deterministic::new(101);
  let bessel = || Bessel::<f32, _>::new(0.5, N, Some(-0.3), Some(T), FT, seed());
  let [host, device] = first_points(bessel(), bessel().on::<Device>());
  assert_eq!([host[0], device[0]], [0.3; 2], "Bessel");

  let build = || {
    let (rho0, kappa_r, mu_r, sigma_r, rho2) = (-0.7, 1.0, -0.5, 0.2, 0.1);
    HestonStochCorr::<f32, _>::new(
      0.0,
      100.0,
      BELOW,
      KAPPA,
      THETA,
      SIGMA,
      rho0,
      kappa_r,
      mu_r,
      sigma_r,
      rho2,
      N,
      Some(T),
      seed(),
    )
  };
  let (host, device) = (build().sample().v, build().on::<Device>().sample().v);
  assert_eq!([host[0], device[0]], [0.0; 2], "HestonStochCorr");
  let next = BELOW + KAPPA * THETA * DT;
  close(host[1], next, "HestonStochCorr, host");
  close(device[1], next, "HestonStochCorr, device");
}
