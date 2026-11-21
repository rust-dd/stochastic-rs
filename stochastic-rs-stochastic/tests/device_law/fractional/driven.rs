use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::diffusion::fcir::Fcir;
use stochastic_rs_stochastic::diffusion::fgbm::Fgbm;
use stochastic_rs_stochastic::diffusion::fjacobi::Fjacobi;
use stochastic_rs_stochastic::diffusion::fou::Fou;
use stochastic_rs_stochastic::interest::fvasicek::Fvasicek;
use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
use stochastic_rs_stochastic::jump::jump_fou_custom::JumpFouCustom;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::process::fbm::Fbm;

use super::*;

#[test]
fn fou_agrees_with_the_cpu_law() {
  let build = || {
    Fou::<f32, _>::new(
      0.7,
      2.0,
      1.0,
      0.3,
      N,
      Some(0.0),
      Some(1.0),
      Deterministic::new(9),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  assert_eq!(device.len(), M);
  assert_eq!(device[0].len(), N);
  assert_eq!(device[0][0], 0.0, "every path starts at x0");
  all_finite(&device, "fOU");
  let (host, dev) = (
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
  );
  assert!(
    (host - dev).abs() < 0.05,
    "fOU terminal mean: host {host}, device {dev}"
  );
}

#[test]
fn fgbm_agrees_with_the_cpu_law() {
  let build = || {
    Fgbm::<f32, _>::new(
      0.7,
      0.05,
      0.2,
      N,
      Some(100.0),
      Some(1.0),
      Deterministic::new(9),
    )
  };
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
    Fcir::<f32, _>::new(
      0.7,
      2.0,
      0.04,
      0.1,
      N,
      Some(0.04),
      Some(1.0),
      VarianceBoundary::Absorption,
      Deterministic::new(9),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, f32::INFINITY, "fCIR");
  let (host, dev) = (
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
  );
  assert!(
    (host - dev).abs() < 0.01,
    "fCIR terminal mean: host {host}, device {dev}"
  );
}

#[test]
fn fjacobi_stays_in_the_unit_interval() {
  let build = || {
    Fjacobi::<f32, _>::new(
      0.7,
      0.3,
      0.6,
      0.2,
      N,
      Some(0.5),
      Some(1.0),
      Deterministic::new(9),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, 1.0, "fJacobi");
  let (host, dev) = (
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
  );
  assert!(
    (host - dev).abs() < 0.02,
    "fJacobi terminal mean: host {host}, device {dev}"
  );
}

/// Fractional Brownian motion is the additive family fed fractional increments; its mean is
/// zero by construction, so the spread carries the law.
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

/// The fractional Vasicek is the fOU under short-rate names and reaches the device through the
/// wrapped process's increment pipeline, not one of its own.
#[test]
fn fractional_vasicek_agrees_with_the_cpu_law() {
  let build = || {
    Fvasicek::<f32, _>::new(
      0.7,
      2.0,
      0.04,
      0.02,
      N,
      Some(0.03),
      Some(1.0),
      Deterministic::new(9),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "fractional Vasicek");
  let (host, dev) = (
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
  );
  assert!(
    (host - dev).abs() < 0.01,
    "fractional Vasicek terminal mean: host {host}, device {dev}"
  );
}

#[test]
fn jump_fou_agrees_with_the_cpu_law() {
  let build = || {
    JumpFou::<f32, _, _>::new(
      0.7,
      2.0,
      1.0,
      0.3,
      3.0,
      SimdNormal::<f32>::new(0.05, 0.1),
      N,
      Some(0.0),
      Some(1.0),
      Deterministic::new(9),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device[0][0], 0.0, "every path starts at x0");
  all_finite(&device, "jump fOU");
  agrees(
    terminal_mean(&host),
    terminal_mean(&device),
    0.03,
    "jump fOU terminal mean",
  );
  agrees(
    terminal_std(&host),
    terminal_std(&device),
    0.06,
    "jump fOU terminal spread",
  );
}

/// Exponential inter-arrivals are Poisson arrivals and exponential sizes a law the kernels
/// draw, so this pair of user-supplied laws runs on the device.
#[test]
fn jump_fou_custom_with_exponential_laws_agrees_with_the_cpu_law() {
  let build = || {
    JumpFouCustom::<f32, _, _, _>::new(
      0.7,
      2.0,
      1.0,
      0.3,
      N,
      Some(0.0),
      Some(1.0),
      SimdExp::<f32>::new(3.0),
      SimdExp::<f32>::new(10.0),
      Deterministic::new(41),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  all_finite(&device, "custom jump fOU");
  agrees(
    terminal_mean(&host),
    terminal_mean(&device),
    0.03,
    "custom jump fOU terminal mean",
  );
  agrees(
    terminal_std(&host),
    terminal_std(&device),
    0.06,
    "custom jump fOU terminal spread",
  );
}

/// Exponential times with normal sizes, the pair one law parameter could not hold.
#[test]
fn jump_fou_custom_with_normal_sizes_agrees_with_the_cpu_law() {
  let build = || {
    JumpFouCustom::<f32, _, _, _>::new(
      0.7,
      2.0,
      1.0,
      0.3,
      N,
      Some(0.0),
      Some(1.0),
      SimdExp::<f32>::new(3.0),
      SimdNormal::<f32>::new(0.05, 0.1),
      Deterministic::new(47),
    )
  };
  const PATHS: usize = 3 * M;
  assert!(build().device_ready());
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  all_finite(&device, "custom jump fOU, normal sizes");
  agrees(
    terminal_mean(&host),
    terminal_mean(&device),
    0.03,
    "custom jump fOU, normal sizes, terminal mean",
  );
  agrees(
    terminal_std(&host),
    terminal_std(&device),
    0.06,
    "custom jump fOU, normal sizes, terminal spread",
  );
}
