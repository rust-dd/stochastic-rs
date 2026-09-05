//! The processes whose increment is a draw rather than a step: a Poisson
//! count, an inverse-Gaussian subordinator, and Brownian motion under that
//! clock. Each is one expression in the kernel because its sampler needs no
//! rejection — a Poisson count by Knuth's product of uniforms, an
//! inverse-Gaussian draw by Michael-Schucany-Haas — so what these cases pin
//! is that the device draws the same law the host's own sampler draws.

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::jump::bilateral_gamma::BilateralGamma;
use stochastic_rs_stochastic::jump::bilateral_gamma::BilateralGammaMotion;
use stochastic_rs_stochastic::jump::hawkes_jd::HawkesJD;
use stochastic_rs_stochastic::jump::ig::Ig;
use stochastic_rs_stochastic::jump::nig::Nig;
use stochastic_rs_stochastic::jump::vg::Vg;
use stochastic_rs_stochastic::process::poisson::Poisson;
use stochastic_rs_stochastic::process::subordinator::alpha_stable::AlphaStableSubordinator;
use stochastic_rs_stochastic::process::subordinator::gamma_subordinator::GammaSubordinator;
use stochastic_rs_stochastic::process::subordinator::ig_subordinator::IGSubordinator;
use stochastic_rs_stochastic::process::subordinator::poisson_subordinator::PoissonSubordinator;
use stochastic_rs_stochastic::process::subordinator::tempered_stable::TemperedStableSubordinator;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::Device;
use super::common::M;
use super::common::agrees;
use super::common::all_finite;
use super::common::terminal_mean;
use super::common::terminal_std;
use super::common::within;

const N: usize = 253;

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
    || IGSubordinator::<f32, _>::new(1.0, 2.0, N, Some(0.0), Some(1.0), Deterministic::new(107));
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, f32::INFINITY, "IG subordinator");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.05,
    "IG subordinator terminal mean",
  );
}

/// The inverse-Gaussian process itself, under the same draw.
#[test]
fn inverse_gaussian_agrees_with_the_cpu_law() {
  let build = || Ig::<f32, _>::new(1.0, N, Some(0.0), Some(1.0), Deterministic::new(109));
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, f32::INFINITY, "inverse Gaussian");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.05,
    "inverse Gaussian terminal mean",
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

/// A positive-stable subordinator is non-decreasing, and its increments are
/// heavy-tailed enough that the terminal mean is dominated by rare large
/// jumps. What is compared is therefore the median, which the tail does not
/// move, alongside the monotonicity the transform guarantees.
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

/// The Hawkes jump diffusion carries its own intensity as a second component
/// the kernel excites and mean-reverts. At most one jump per step, as the
/// host's Bernoulli test takes it, so what the terminal mean pins is that the
/// device's excitement loop matches the host's.
#[test]
fn hawkes_jump_diffusion_agrees_with_the_cpu_law() {
  let build = || {
    HawkesJD::<f32, _>::new(
      0.02,
      0.2,
      1.0,
      0.5,
      2.0,
      -0.02,
      0.05,
      N,
      Some(0.0),
      Some(1.0),
      Deterministic::new(139),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "Hawkes jump diffusion");
  let (host, dev) = (
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
  );
  assert!(
    (host - dev).abs() < 0.02,
    "Hawkes jump diffusion terminal mean: host {host}, device {dev}"
  );
}

/// A gamma subordinator is non-decreasing and positive, and its terminal mean
/// is `ν t / λ`. The kernel draws it by Marsaglia-Tsang, whose rejection loop
/// is bounded; what this pins is that the bounded loop still produces the
/// law.
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

/// A tempered-stable subordinator: the kernel draws the candidates above the
/// truncation and keeps each with the tempering probability, so the sum it
/// builds is the thinned one the host builds by the same test. The path is
/// non-decreasing because every kept jump is positive.
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

/// Poisson in count mode is a running sum of exponential inter-arrival times,
/// which the kernel draws by inverse CDF from its own uniform. The terminal
/// arrival time of `n - 1` of them is Gamma(n - 1, 1/lambda), so its mean and
/// spread are what carry the law. Horizon mode has no grid and stays on the
/// host whatever the backend, which the last assertion pins.
#[test]
fn poisson_arrivals_agree_with_the_cpu_law() {
  let build = || Poisson::<f32, _>::new(4.0, Some(N), None, Deterministic::new(71));
  let device = build().on::<Device>().sample_par(M);
  let host = build().sample_par(M);
  assert_eq!(device.len(), M);
  assert_eq!(device[0].len(), N);
  assert_eq!(device[0][0], 0.0, "every path starts at the origin");
  all_finite(&device, "Poisson arrivals");
  assert!(
    device
      .iter()
      .all(|p| p.windows(2).into_iter().all(|w| w[1] >= w[0])),
    "Poisson arrivals must not run backwards"
  );
  agrees(
    terminal_mean(&host),
    terminal_mean(&device),
    0.02,
    "Poisson terminal arrival",
  );
  agrees(
    terminal_std(&host),
    terminal_std(&device),
    0.08,
    "Poisson arrival spread",
  );
  let horizon = Poisson::<f32, _>::new(4.0, None, Some(1.0), Deterministic::new(71))
    .on::<Device>()
    .sample();
  assert_eq!(horizon[0], 0.0, "horizon mode still starts at the origin");
}
