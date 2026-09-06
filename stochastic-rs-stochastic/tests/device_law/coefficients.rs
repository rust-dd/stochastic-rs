//! The processes whose coefficients are functions of time and state written as
//! expressions: the kernel interprets each coefficient's compiled program at
//! every step, so what the device has to reproduce is the host sampler's law
//! under the very same expression — and a coefficient written as a closure has
//! to stay on the host, bit-identically to the `Cpu` build.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::traits::Expr;
use stochastic_rs_stochastic::interest::cheyette::Cheyette;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volterra::kernel::ExponentialKernel;
use stochastic_rs_stochastic::volterra::sve::VolterraSde;

use super::common::Device;
use super::common::M;
use super::common::agrees;
use super::common::all_finite;
use super::common::terminal_mean;
use super::common::terminal_std;

const N: usize = 128;

fn forward_curve(t: f32) -> f32 {
  0.03 + 0.01 * t
}

fn native_sigma(_t: f32, x: f32) -> f32 {
  0.01 + 0.4 * x.abs()
}

fn native_drift(_t: f32, x: f32) -> f32 {
  -0.5 * x
}

fn native_diffusion(_t: f32, x: f32) -> f32 {
  0.3 + 0.1 * x.abs()
}

/// The Cheyette state under a displaced local volatility written as an
/// expression: the device's `x` plane is centred where the host's is and
/// spreads as the host's does, and its `y` plane — the accumulated variance
/// — agrees in mean and spread.
#[test]
fn cheyette_with_an_expression_volatility_agrees_with_the_cpu_law() {
  let build = || {
    Cheyette::<f32, _>::new(
      forward_curve as fn(f32) -> f32,
      0.5,
      Expr::lit(0.01) + Expr::x().abs() * 0.4,
      N,
      Some(1.0),
      Deterministic::new(211),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  let host = build().sample_par(M);
  let plane = |paths: &[[Array1<f32>; 2]], k: usize| -> Vec<Array1<f32>> {
    paths.iter().map(|p| p[k].clone()).collect()
  };
  let (dx, dy) = (plane(&device, 0), plane(&device, 1));
  let (hx, hy) = (plane(&host, 0), plane(&host, 1));
  all_finite(&dx, "Cheyette x");
  all_finite(&dy, "Cheyette y");
  let scale = terminal_std(&hx) / (M as f64).sqrt();
  assert!(
    (terminal_mean(&hx) - terminal_mean(&dx)).abs() < 5.0 * scale,
    "Cheyette x terminal mean: host {}, device {}",
    terminal_mean(&hx),
    terminal_mean(&dx)
  );
  agrees(terminal_std(&hx), terminal_std(&dx), 0.06, "Cheyette x terminal spread");
  agrees(terminal_mean(&hy), terminal_mean(&dy), 0.03, "Cheyette y terminal mean");
  agrees(terminal_std(&hy), terminal_std(&dy), 0.06, "Cheyette y terminal spread");
}

/// With the diffusion at zero the Volterra equation is deterministic, so the
/// device's lift has to reproduce the host's path point for point: the same
/// kernel tables, the same start added back, the drift program evaluated at
/// the same time and state — the wiring, checked without noise in the way.
#[test]
fn a_noiseless_volterra_equation_agrees_point_for_point() {
  let build = || {
    VolterraSde::<f32, _, _>::new(
      ExponentialKernel::new(2.0_f32, 1.0),
      Expr::lit(0.3) - Expr::x() * 0.5 + Expr::t() * 0.2,
      Expr::lit(0.0),
      N,
      Some(0.2),
      Some(1.0),
      Deterministic::new(233),
    )
  };
  let device = build().on::<Device>().sample();
  let host = build().sample();
  for (i, (h, d)) in host.iter().zip(device.iter()).enumerate() {
    assert!(
      (h - d).abs() < 1e-5,
      "noiseless Volterra equation at point {i}: host {h}, device {d}"
    );
  }
}

/// A Volterra equation under an exponential kernel with an expression drift
/// and diffusion, on the kernel's Markov lift: the terminal mean and spread
/// agree with the host's lift.
#[test]
fn volterra_sde_with_expression_coefficients_agrees_with_the_cpu_law() {
  let build = || {
    VolterraSde::<f32, _, _>::new(
      ExponentialKernel::new(2.0_f32, 1.0),
      Expr::x() * -0.5,
      Expr::lit(0.3) + Expr::x().abs() * 0.1,
      N,
      Some(0.2),
      Some(1.0),
      Deterministic::new(223),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  all_finite(&device, "Volterra SDE");
  let scale = terminal_std(&host) / (PATHS as f64).sqrt();
  assert!(
    (terminal_mean(&host) - terminal_mean(&device)).abs() < 5.0 * scale,
    "Volterra SDE terminal mean: host {}, device {}",
    terminal_mean(&host),
    terminal_mean(&device)
  );
  agrees(
    terminal_std(&host),
    terminal_std(&device),
    0.06,
    "Volterra SDE terminal spread",
  );
}

/// A coefficient written as a Rust closure has no program; the device build
/// samples on the host and is the host build to the bit.
#[test]
fn closure_coefficients_keep_cheyette_and_the_volterra_sde_on_the_host() {
  let cheyette = || {
    Cheyette::<f32, _>::new(
      forward_curve as fn(f32) -> f32,
      0.5,
      native_sigma as fn(f32, f32) -> f32,
      32,
      Some(1.0),
      Deterministic::new(227),
    )
  };
  assert_eq!(cheyette().on::<Device>().sample_par(4), cheyette().sample_par(4));
  assert_eq!(cheyette().on::<Device>().sample(), cheyette().sample());
  let volterra = || {
    VolterraSde::<f32, _, _>::new(
      ExponentialKernel::new(2.0_f32, 1.0),
      native_drift as fn(f32, f32) -> f32,
      native_diffusion as fn(f32, f32) -> f32,
      32,
      Some(0.2),
      Some(1.0),
      Deterministic::new(229),
    )
  };
  assert_eq!(volterra().on::<Device>().sample_par(4), volterra().sample_par(4));
  assert_eq!(volterra().on::<Device>().sample(), volterra().sample());
}
