//! Each boundary where the three differ: outside Feller's condition they have three means, on the
//! device as on the host.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::traits::Expr;
use stochastic_rs_stochastic::correlation::heston_stoch_corr::HestonStochCorr;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::heston_slv::HestonSlv;

use super::common::Device;

const KAPPA: f32 = 0.4;
const THETA: f32 = 0.2;
const SIGMA: f32 = 0.6;
const V0: f32 = 0.05;
const T: f32 = 5.0;
/// A monthly grid over `T`.
const N: usize = 61;
const PATHS: usize = 40_000;
const FT: VarianceBoundary = VarianceBoundary::FullTruncation;
const ABSORPTION: VarianceBoundary = VarianceBoundary::Absorption;
const REFLECT: VarianceBoundary = VarianceBoundary::Reflect;

/// The relative bias of the rows' terminal mean against the exact CIR mean, and their least entry.
fn bias_and_least<'a>(rows: impl Iterator<Item = &'a Array1<f32>>) -> (f64, f32) {
  let exact = (THETA + (V0 - THETA) * (-KAPPA * T).exp()) as f64;
  let (mut sum, mut count, mut least) = (0.0, 0usize, f32::INFINITY);
  for row in rows {
    sum += row[N - 1] as f64;
    count += 1;
    least = row.iter().fold(least, |a, &b| a.min(b));
  }
  (sum / count as f64 / exact - 1.0, least)
}

/// Bands five standard errors (0.8 % each) apart around the measured +1 %, +12 % and +23 %.
#[test]
fn cir_steps_its_own_boundary_on_the_device_outside_feller() {
  for (boundary, low, high) in [
    (FT, -0.04, 0.04),
    (ABSORPTION, 0.07, 0.17),
    (REFLECT, 0.18, 0.30),
  ] {
    let seed = || Deterministic::new(73);
    let build = || Cir::<f32, _>::new(KAPPA, THETA, SIGMA, N, Some(V0), Some(T), boundary, seed());
    let (device, least) = bias_and_least(build().on::<Device>().sample_par(PATHS).iter());
    let (host, _) = bias_and_least(build().sample_par(PATHS).iter());
    let touches_zero = boundary != REFLECT;
    assert!(
      least >= 0.0 && (least == 0.0) == touches_zero,
      "{boundary:?}: device least {least}"
    );
    for (side, bias) in [("device", device), ("host", host)] {
      assert!(
        low < bias && bias < high,
        "{boundary:?}: the {side} bias {bias} outside ({low}, {high})"
      );
    }
  }
}

/// Family 44 has no boundary and steps full truncation; absorption would sit 11 % above.
#[test]
fn stochastic_correlation_heston_truncates_in_full_on_the_device_outside_feller() {
  let build = || {
    let (rho0, kappa_r, mu_r, sigma_r, rho2) = (-0.7, 1.0, -0.5, 0.2, 0.1);
    HestonStochCorr::<f32, _>::new(
      0.0,
      100.0,
      V0,
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
      Deterministic::new(79),
    )
  };
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  let (device, least) = bias_and_least(device.iter().map(|p| &p.v));
  let (host, _) = bias_and_least(host.iter().map(|p| &p.v));
  assert_eq!(least, 0.0, "the device's smallest variance");
  assert!(
    device.abs() < 0.04 && host.abs() < 0.04,
    "device {device}, host {host}"
  );
}

/// Family 120 likewise, under a flat leverage so the variance is Heston's own.
#[test]
fn heston_slv_truncates_in_full_on_the_device_outside_feller() {
  let build = || {
    let (rho, mu, eta, leverage) = (-0.7, 0.0, 1.0, Expr::lit(1.0).compile().unwrap());
    let seed = Deterministic::new(83);
    HestonSlv::<f32, _>::new(
      100.0,
      V0,
      KAPPA,
      THETA,
      SIGMA,
      rho,
      mu,
      eta,
      leverage,
      N,
      Some(T),
      seed,
    )
  };
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  let (device, least) = bias_and_least(device.iter().map(|p| &p[1]));
  let (host, _) = bias_and_least(host.iter().map(|p| &p[1]));
  assert_eq!(least, 0.0, "the device's smallest variance");
  assert!(
    device.abs() < 0.04 && host.abs() < 0.04,
    "device {device}, host {host}"
  );
}
