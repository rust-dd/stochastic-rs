//! The processes whose increment is a draw rather than a step: a Poisson
//! count, an inverse-Gaussian subordinator, and Brownian motion under that
//! clock. Each is one expression in the kernel because its sampler needs no
//! rejection — a Poisson count by Knuth's product of uniforms, an
//! inverse-Gaussian draw by Michael-Schucany-Haas — so what these cases pin
//! is that the device draws the same law the host's own sampler draws.

use ndarray::Array1;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::jump::bilateral_gamma::BilateralGamma;
use stochastic_rs_stochastic::jump::bilateral_gamma::BilateralGammaMotion;
use stochastic_rs_stochastic::jump::hawkes_jd::HawkesJD;
use stochastic_rs_stochastic::jump::ig::Ig;
use stochastic_rs_stochastic::jump::nig::Nig;
use stochastic_rs_stochastic::jump::vg::Vg;
use stochastic_rs_stochastic::process::hawkes::Hawkes;
use stochastic_rs_stochastic::process::multivariate_hawkes::MultivariateHawkes;
use stochastic_rs_stochastic::process::poisson::Poisson;
use stochastic_rs_stochastic::process::subordinator::alpha_stable::AlphaStableSubordinator;
use stochastic_rs_stochastic::process::subordinator::gamma_subordinator::GammaSubordinator;
use stochastic_rs_stochastic::process::subordinator::ig_subordinator::IGSubordinator;
use stochastic_rs_stochastic::process::subordinator::inverse_alpha_stable::InverseAlphaStableSubordinator;
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

/// The inverse-Gaussian clock stays positive on a grid fine enough to expose
/// the draw's cancellation, on the device and on the host alike.
///
/// Michael-Schucany-Haas takes the small root as a difference of two terms
/// that agree to more digits the finer the grid is: at `n = 253` the two
/// still differ in single precision, at `n = 2048` they do not, and the root
/// came out zero or negative — a clock that runs backwards, and a NaN as soon
/// as the normal inverse Gaussian takes its square root. Every path was NaN
/// on the device and a quarter of them on the host before the roots were
/// rewritten as a sum and a quotient.
#[test]
fn the_inverse_gaussian_clock_survives_a_fine_grid() {
  const FINE: usize = 2_048;
  let clock = || {
    IGSubordinator::<f32, _>::new(
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

/// Hawkes in count mode: the host thins Ogata's proposals, the kernel runs
/// the exact two-uniform recursion, and both are the process with intensity
/// `mu + Σ alpha e^{-beta (t - t_k)}` — so the time of the last event, whose
/// mean the branching ratio sets, and the clustering of the waits, which the
/// excitation alone produces, have to agree.
#[test]
fn hawkes_agrees_with_the_cpu_law() {
  const EVENTS: usize = 64;
  let build = || Hawkes::<f32, _>::new(1.0, 0.5, 1.5, Some(EVENTS), None, Deterministic::new(157));
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  assert!(device.iter().all(|p| p.len() == EVENTS));
  assert!(
    device.iter().all(|p| p[0] == 0.0),
    "a path did not start at zero"
  );
  assert!(
    device
      .iter()
      .all(|p| p.windows(2).into_iter().all(|w| w[1] >= w[0])),
    "an event time went backwards"
  );
  // A wait is positive, but the sum is `f32`: past `t ≈ 40` a wait under
  // `t · 2⁻²⁴ ≈ 2e-6` rounds away, and with an intensity around 3 that is
  // `≈ 1e-5` of the waits — the host sampler shows the same handful on this
  // batch. What a broken stream would give is a *rate*, not a handful, so the
  // bound is three orders above the arithmetic's own.
  let ties = |paths: &[Array1<f32>]| {
    paths
      .iter()
      .map(|p| p.windows(2).into_iter().filter(|w| w[1] == w[0]).count())
      .sum::<usize>()
  };
  let (pairs, stuck) = (PATHS * (EVENTS - 1), ties(&device));
  assert!(
    stuck * 10_000 < pairs,
    "{stuck} of {pairs} device waits rounded to zero, host had {}",
    ties(&host)
  );
  agrees(
    terminal_mean(&host),
    terminal_mean(&device),
    0.03,
    "Hawkes last event time",
  );
  agrees(
    terminal_std(&host),
    terminal_std(&device),
    0.06,
    "Hawkes last event time spread",
  );
  // The waits' coefficient of variation is one for a Poisson stream and
  // above it under excitation; it is the statistic the excitation moves and
  // the baseline does not.
  let clustering = |paths: &[Array1<f32>]| {
    let waits: Vec<f64> = paths
      .iter()
      .flat_map(|p| p.windows(2).into_iter().map(|w| (w[1] - w[0]) as f64))
      .collect();
    let n = waits.len() as f64;
    let mean = waits.iter().sum::<f64>() / n;
    let var = waits.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / n;
    var.sqrt() / mean
  };
  let (h, d) = (clustering(&host), clustering(&device));
  assert!(h > 1.1, "the host waits do not cluster: {h}");
  agrees(h, d, 0.03, "Hawkes wait clustering");
}

/// The horizon mode has a random length and no grid, so a device build
/// samples on the host and is the host build to the bit.
#[test]
fn hawkes_horizon_mode_keeps_the_process_on_the_host() {
  let build = || Hawkes::<f32, _>::new(1.0, 0.5, 1.5, None, Some(10.0), Deterministic::new(163));
  assert_eq!(build().on::<Device>().sample_par(8), build().sample_par(8));
}

/// A small scale keeps the kernel's stable subordinator on the host's law
/// rather than collapsing it to zero.
///
/// The kernel folds the transform's scale as `(c·dt)^{1/α}`, and in `f32` that
/// factor underflows to zero long before the path it scales does: at `α = 0.2`
/// and `c·dt = 9.8e-10` it is `8.8e-46`, zero as a float, while the terminal
/// value near `(c·T)^{1/α} = 1e-30` is an ordinary single. Every device path
/// was then identically zero against a host that was still right — the host
/// folds in `f64`, so only the kernel copy of the transform showed it. The
/// same fold in the kernel's own precision turned 7.9 % of the points of an
/// `α = 0.1` device path into NaN.
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
  agrees(
    median(&host),
    device_median,
    0.05,
    "small-scale stable subordinator terminal median",
  );
}

/// The inverse stable subordinator is the first-passage clock of a stable
/// subordinator built on a table in its own argument: the kernel builds the
/// same table from the same positive-stable increments and interpolates the
/// same way, so the clock's terminal law — mean and spread — and its
/// monotonicity have to agree.
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

/// The bivariate Hawkes process in count mode, its two components on one
/// launch: every path carries the right number of events split across the
/// components, each component's list opens at the origin and advances, and
/// the last event's time, its spread and the share of events on the first
/// component agree with the host's thinning sampler.
#[test]
fn bivariate_hawkes_agrees_with_the_cpu_law() {
  const EVENTS: usize = 64;
  let build = || {
    MultivariateHawkes::<f32, _>::new(
      array![1.0_f32, 0.7],
      array![[0.3_f32, 0.2], [0.1, 0.4]],
      array![[1.5_f32, 1.5], [2.0, 2.0]],
      5.0,
      Deterministic::new(163),
    )
    .with_count(EVENTS)
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  for path in &device {
    assert_eq!(path.len(), 2);
    assert_eq!(path[0].len() + path[1].len(), EVENTS + 2);
    for component in path {
      assert!(
        component[0] == 0.0 && component.windows(2).into_iter().all(|w| w[1] >= w[0]),
        "a component's events went backwards"
      );
    }
  }
  // As in the univariate case, a wait below `t · 2⁻²⁴` rounds away in `f32`;
  // the bound is three orders above that rate, so a stream that stalled would
  // still fail here.
  let ties = |paths: &[Vec<Array1<f32>>]| {
    paths
      .iter()
      .flatten()
      .map(|c| c.windows(2).into_iter().filter(|w| w[1] == w[0]).count())
      .sum::<usize>()
  };
  let stuck = ties(&device);
  assert!(
    stuck * 10_000 < PATHS * EVENTS,
    "{stuck} of {} device waits rounded to zero, host had {}",
    PATHS * EVENTS,
    ties(&host)
  );
  let last: fn(&[Vec<Array1<f32>>]) -> Vec<f64> = |paths| {
    paths
      .iter()
      .map(|p| p[0][p[0].len() - 1].max(p[1][p[1].len() - 1]) as f64)
      .collect()
  };
  let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
  let std = |v: &[f64]| {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
  };
  let (h, d) = (last(&host), last(&device));
  agrees(mean(&h), mean(&d), 0.03, "bivariate Hawkes last event time");
  agrees(
    std(&h),
    std(&d),
    0.06,
    "bivariate Hawkes last event time spread",
  );
  let share = |paths: &[Vec<Array1<f32>>]| {
    paths
      .iter()
      .map(|p| (p[0].len() - 1) as f64 / EVENTS as f64)
      .sum::<f64>()
      / paths.len() as f64
  };
  agrees(
    share(&host),
    share(&device),
    0.03,
    "bivariate Hawkes share of events on the first component",
  );
}

/// Per-pair decays, a third component or the horizon mode have no family;
/// the device build samples on the host and is the host build to the bit.
#[test]
fn per_pair_decays_a_third_component_or_a_horizon_keep_the_multivariate_hawkes_on_the_host() {
  let pairs = || {
    MultivariateHawkes::<f32, _>::new(
      array![1.0_f32, 0.7],
      array![[0.3_f32, 0.2], [0.1, 0.4]],
      array![[1.5_f32, 2.0], [2.0, 1.5]],
      5.0,
      Deterministic::new(167),
    )
    .with_count(16)
  };
  assert_eq!(pairs().on::<Device>().sample_par(4), pairs().sample_par(4));
  let three = || {
    MultivariateHawkes::<f32, _>::new(
      array![1.0_f32, 0.7, 0.5],
      array![[0.3_f32, 0.1, 0.1], [0.1, 0.3, 0.1], [0.1, 0.1, 0.3]],
      array![[2.0_f32; 3], [2.0; 3], [2.0; 3]],
      5.0,
      Deterministic::new(173),
    )
    .with_count(16)
  };
  assert_eq!(three().on::<Device>().sample_par(4), three().sample_par(4));
  let horizon = || {
    MultivariateHawkes::<f32, _>::new(
      array![1.0_f32, 0.7],
      array![[0.3_f32, 0.2], [0.1, 0.4]],
      array![[1.5_f32, 1.5], [2.0, 2.0]],
      3.0,
      Deterministic::new(179),
    )
  };
  assert_eq!(
    horizon().on::<Device>().sample_par(4),
    horizon().sample_par(4)
  );
  assert_eq!(horizon().on::<Device>().sample(), horizon().sample());
}
