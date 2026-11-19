use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::jump::bilateral_gamma::BilateralGamma;
use stochastic_rs_stochastic::jump::bilateral_gamma::BilateralGammaMotion;
use stochastic_rs_stochastic::jump::nig::Nig;
use stochastic_rs_stochastic::jump::vg::Vg;
use stochastic_rs_stochastic::process::subordinator::alpha_stable::AlphaStableSubordinator;
use stochastic_rs_stochastic::process::subordinator::gamma_subordinator::GammaSubordinator;
use stochastic_rs_stochastic::process::subordinator::ig_subordinator::IgSubordinator;
use stochastic_rs_stochastic::process::subordinator::inverse_alpha_stable::InverseAlphaStableSubordinator;
use stochastic_rs_stochastic::process::subordinator::poisson_subordinator::PoissonSubordinator;
use stochastic_rs_stochastic::process::subordinator::tempered_stable::TemperedStableSubordinator;

use super::*;

/// A counting path only increases, and its terminal mean is `λt`.
#[test]
fn poisson_subordinator_agrees_with_the_cpu_law() {
  let build =
    || PoissonSubordinator::<f32, _>::new(20.0, N, Some(0.0), Some(1.0), Deterministic::new(103));
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, f32::INFINITY, "Poisson subordinator");
  assert!(
    device
      .iter()
      .all(|p| p.windows(2).into_iter().all(|w| w[1] >= w[0])),
    "a counting path went backwards"
  );
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.03,
    "Poisson subordinator terminal mean",
  );
}

/// An inverse-Gaussian subordinator is non-decreasing and positive.
#[test]
fn inverse_gaussian_subordinator_agrees_with_the_cpu_law() {
  let build =
    || IgSubordinator::<f32, _>::new(1.0, 2.0, N, Some(0.0), Some(1.0), Deterministic::new(107));
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, f32::INFINITY, "IG subordinator");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.05,
    "IG subordinator terminal mean",
  );
}

/// Brownian motion under an inverse-Gaussian clock: the drift is `θ` times
/// the clock, so the terminal mean carries both draws.
#[test]
fn normal_inverse_gaussian_agrees_with_the_cpu_law() {
  let build = || {
    Nig::<f32, _>::new(
      -0.1,
      0.2,
      0.5,
      N,
      Some(0.0),
      Some(1.0),
      Deterministic::new(113),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "normal inverse Gaussian");
  let (host, dev) = (
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
  );
  assert!(
    (host - dev).abs() < 0.02,
    "normal inverse Gaussian terminal mean: host {host}, device {dev}"
  );
}

/// The inverse-Gaussian clock stays positive on a fine grid. Michael-Schucany-Haas's small root
/// as a difference cancels to zero or below at `n = 2048` in `f32`; a sum and a quotient do not.
#[test]
fn the_inverse_gaussian_clock_survives_a_fine_grid() {
  const FINE: usize = 2_048;
  let clock = || {
    IgSubordinator::<f32, _>::new(
      1.0,
      2.0,
      FINE,
      Some(0.0),
      Some(1.0),
      Deterministic::new(113),
    )
  };
  let nig = || {
    Nig::<f32, _>::new(
      -0.1,
      0.2,
      0.5,
      FINE,
      Some(0.0),
      Some(1.0),
      Deterministic::new(113),
    )
  };
  all_finite(
    &clock().sample_par(M),
    "inverse-Gaussian subordinator (host)",
  );
  all_finite(
    &clock().on::<Device>().sample_par(M),
    "inverse-Gaussian subordinator",
  );
  all_finite(&nig().sample_par(M), "normal inverse Gaussian (host)");
  all_finite(
    &nig().on::<Device>().sample_par(M),
    "normal inverse Gaussian",
  );
}

/// A positive-stable subordinator is non-decreasing and so heavy-tailed that rare jumps
/// dominate the terminal mean; the median, which the tail does not move, is compared instead.
#[test]
fn stable_subordinator_agrees_with_the_cpu_law() {
  let build = || {
    AlphaStableSubordinator::<f32, _>::new(
      0.7,
      1.0,
      N,
      Some(0.0),
      Some(1.0),
      Deterministic::new(127),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, f32::INFINITY, "stable subordinator");
  assert!(
    device
      .iter()
      .all(|p| p.windows(2).into_iter().all(|w| w[1] >= w[0])),
    "a subordinator path went backwards"
  );
  let median = |paths: &[ndarray::Array1<f32>]| {
    let last = paths[0].len() - 1;
    let mut v: Vec<f32> = paths.iter().map(|p| p[last]).collect();
    v.sort_by(f32::total_cmp);
    v[v.len() / 2] as f64
  };
  agrees(
    median(&build().sample_par(M)),
    median(&device),
    0.10,
    "stable subordinator terminal median",
  );
}

/// A gamma subordinator is non-decreasing and positive with terminal mean `ν t / λ`; this pins
/// that the kernel's bounded Marsaglia-Tsang rejection loop still produces the law.
#[test]
fn gamma_subordinator_agrees_with_the_cpu_law() {
  let build =
    || GammaSubordinator::<f32, _>::new(2.0, 1.5, N, Some(0.0), Some(1.0), Deterministic::new(197));
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, f32::INFINITY, "gamma subordinator");
  assert!(
    device
      .iter()
      .all(|p| p.windows(2).into_iter().all(|w| w[1] >= w[0])),
    "a subordinator path went backwards"
  );
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.05,
    "gamma subordinator terminal mean",
  );
}

/// Brownian motion under a gamma clock: the drift is `μ` times the clock, so
/// the terminal mean carries the gamma draw and the Brownian one together.
#[test]
fn variance_gamma_agrees_with_the_cpu_law() {
  let build = || {
    Vg::<f32, _>::new(
      -0.1,
      0.2,
      0.5,
      N,
      Some(0.0),
      Some(1.0),
      Deterministic::new(199),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "variance gamma");
  let (host, dev) = (
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
  );
  assert!(
    (host - dev).abs() < 0.02,
    "variance gamma terminal mean: host {host}, device {dev}"
  );
}

/// The difference of two gamma processes: both draws happen in the same step,
/// from streams of their own.
#[test]
fn bilateral_gamma_agrees_with_the_cpu_law() {
  let build = || {
    BilateralGamma::<f32, _>::new(
      1.5,
      10.0,
      1.2,
      12.0,
      N,
      Some(0.0),
      Some(1.0),
      Deterministic::new(211),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "bilateral gamma");
  let (host, dev) = (
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
  );
  assert!(
    (host - dev).abs() < 0.02,
    "bilateral gamma terminal mean: host {host}, device {dev}"
  );
}

/// The same, with a Brownian part added.
#[test]
fn bilateral_gamma_motion_agrees_with_the_cpu_law() {
  let build = || {
    BilateralGammaMotion::<f32, _>::new(
      0.1,
      1.5,
      10.0,
      1.2,
      12.0,
      N,
      Some(0.0),
      Some(1.0),
      Deterministic::new(223),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "bilateral gamma motion");
  let (host, dev) = (
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
  );
  assert!(
    (host - dev).abs() < 0.03,
    "bilateral gamma motion terminal mean: host {host}, device {dev}"
  );
}

/// A tempered-stable subordinator: the kernel keeps each candidate above the truncation with
/// the tempering probability, the host's own test, and every kept jump is positive.
#[test]
fn tempered_stable_subordinator_agrees_with_the_cpu_law() {
  let build = || {
    TemperedStableSubordinator::<f32, _>::new(
      0.6,
      1.0,
      2.0,
      0.05,
      N,
      Some(0.0),
      Some(1.0),
      Deterministic::new(227),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, f32::INFINITY, "tempered stable subordinator");
  assert!(
    device
      .iter()
      .all(|p| p.windows(2).into_iter().all(|w| w[1] >= w[0])),
    "a subordinator path went backwards"
  );
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.06,
    "tempered stable subordinator terminal mean",
  );
}

/// A small scale keeps the kernel's stable subordinator on the host's law: `(c·dt)^{1/α}`
/// underflows in `f32` (`8.8e-46` at `α = 0.2`, `c·dt = 9.8e-10`), so the kernel folds it in logs.
#[test]
fn a_small_scale_keeps_the_stable_subordinator_on_the_cpu_law() {
  let build = || {
    AlphaStableSubordinator::<f32, _>::new(
      0.2,
      1e-6,
      1024,
      Some(0.0),
      Some(1.0),
      Deterministic::new(4242),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  let host = build().sample_par(M);
  all_finite(&device, "small-scale stable subordinator");
  assert!(
    device
      .iter()
      .all(|p| p.windows(2).into_iter().all(|w| w[1] >= w[0])),
    "a subordinator path went backwards"
  );
  let median = |paths: &[Array1<f32>]| {
    let last = paths[0].len() - 1;
    let mut v: Vec<f32> = paths.iter().map(|p| p[last]).collect();
    v.sort_by(f32::total_cmp);
    v[v.len() / 2] as f64
  };
  let device_median = median(&device);
  assert!(
    device_median > 0.0,
    "every device path collapsed to zero: median terminal = {device_median:e}"
  );
  // The terminal law spans decades, so its median moves by a fifth between seeds; its log does not.
  let logs = |paths: &[Array1<f32>]| paths.iter().map(|p| p.mapv(f32::ln)).collect::<Vec<_>>();
  means_agree(
    &logs(&host),
    &logs(&device),
    "small-scale stable subordinator terminal log",
  );
}

/// The inverse stable subordinator is a stable subordinator's first-passage clock on a table:
/// the kernel builds and interpolates the same table, so mean, spread and monotonicity agree.
#[test]
fn inverse_alpha_stable_subordinator_agrees_with_the_cpu_law() {
  let build = || {
    InverseAlphaStableSubordinator::<f32, _>::new(
      0.7,
      1.0,
      N,
      Some(1.0),
      256,
      None,
      Deterministic::new(223),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  assert_eq!(device[0][0], 0.0, "the clock starts at the origin");
  all_finite(&device, "inverse stable subordinator");
  assert!(
    device
      .iter()
      .all(|p| p.windows(2).into_iter().all(|w| w[1] >= w[0])),
    "an inverse clock ran backwards"
  );
  agrees(
    terminal_mean(&host),
    terminal_mean(&device),
    0.03,
    "inverse stable subordinator terminal mean",
  );
  agrees(
    terminal_std(&host),
    terminal_std(&device),
    0.06,
    "inverse stable subordinator terminal spread",
  );
}

/// A table finer than the kernels' per-path table samples on the host, and
/// is then the host build to the bit.
#[test]
fn a_finer_table_keeps_the_inverse_subordinator_on_the_host() {
  let build = || {
    InverseAlphaStableSubordinator::<f32, _>::new(
      0.7,
      1.0,
      64,
      Some(1.0),
      600,
      None,
      Deterministic::new(227),
    )
  };
  assert_eq!(build().on::<Device>().sample_par(8), build().sample_par(8));
}
