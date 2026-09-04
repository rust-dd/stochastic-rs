//! The fGN-driven processes reach a device through the Euler engine: the
//! device runs the fractional-noise pipeline itself, keeps the increments in
//! its own memory and hands them to the same families that serve the Gaussian
//! processes. The two sides draw different streams, so what is pinned here is
//! the law and the boundary, not the path.

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::process::fbm::Fbm;
use stochastic_rs_stochastic::diffusion::fcir::Fcir;
use stochastic_rs_stochastic::diffusion::fgbm::Fgbm;
use stochastic_rs_stochastic::diffusion::fjacobi::FJacobi;
use stochastic_rs_stochastic::diffusion::fou::Fou;
use stochastic_rs_stochastic::interest::fractional_vasicek::FVasicek;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::Device;
use super::common::agrees;
use super::common::all_finite;
use super::common::terminal_mean;
use super::common::terminal_std;
use super::common::within;

/// Fewer paths than the Gaussian cases use: each one carries a 512-point fGN
/// draw, and the comparisons below are on statistics that settle early.
const M: usize = 2_000;
const N: usize = 512;

#[test]
fn fou_agrees_with_the_cpu_law() {
  let build = || Fou::<f32, _>::new(0.7, 2.0, 1.0, 0.3, N, Some(0.0), Some(1.0), Deterministic::new(9));
  let device = build().on::<Device>().sample_par(M);
  assert_eq!(device.len(), M);
  assert_eq!(device[0].len(), N);
  assert_eq!(device[0][0], 0.0, "every path starts at x0");
  all_finite(&device, "fOU");
  let (host, dev) = (terminal_mean(&build().sample_par(M)), terminal_mean(&device));
  assert!((host - dev).abs() < 0.05, "fOU terminal mean: host {host}, device {dev}");
}

#[test]
fn fgbm_agrees_with_the_cpu_law() {
  let build =
    || Fgbm::<f32, _>::new(0.7, 0.05, 0.2, N, Some(100.0), Some(1.0), Deterministic::new(9));
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "fGBM");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.02,
    "fGBM terminal mean",
  );
}

#[test]
fn fcir_stays_nonnegative_and_agrees() {
  let build = || {
    Fcir::<f32, _>::new(0.7, 2.0, 0.04, 0.1, N, Some(0.04), Some(1.0), None, Deterministic::new(9))
  };
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, f32::INFINITY, "fCIR");
  let (host, dev) = (terminal_mean(&build().sample_par(M)), terminal_mean(&device));
  assert!((host - dev).abs() < 0.01, "fCIR terminal mean: host {host}, device {dev}");
}

#[test]
fn fjacobi_stays_in_the_unit_interval() {
  let build = || {
    FJacobi::<f32, _>::new(0.7, 0.3, 0.6, 0.2, N, Some(0.5), Some(1.0), Deterministic::new(9))
  };
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, 1.0, "fJacobi");
  let (host, dev) = (terminal_mean(&build().sample_par(M)), terminal_mean(&device));
  assert!((host - dev).abs() < 0.02, "fJacobi terminal mean: host {host}, device {dev}");
}

/// Fractional Brownian motion is the additive family fed fractional
/// increments, and its mean is zero by construction, so the spread carries
/// the law.
#[test]
fn fbm_agrees_with_the_cpu_law() {
  let build = || Fbm::<f32, _>::new(0.7, N, Some(1.0), Deterministic::new(9));
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "fBM");
  agrees(
    terminal_std(&build().sample_par(M)),
    terminal_std(&device),
    0.06,
    "fBM terminal spread",
  );
}

/// The fractional Vasicek is the fOU under short-rate names, and it reaches
/// the device through the wrapped process's own increment pipeline rather
/// than one of its own.
#[test]
fn fractional_vasicek_agrees_with_the_cpu_law() {
  let build =
    || FVasicek::<f32, _>::new(0.7, 2.0, 0.04, 0.02, N, Some(0.03), Some(1.0), Deterministic::new(9));
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "fractional Vasicek");
  let (host, dev) = (terminal_mean(&build().sample_par(M)), terminal_mean(&device));
  assert!(
    (host - dev).abs() < 0.01,
    "fractional Vasicek terminal mean: host {host}, device {dev}"
  );
}
