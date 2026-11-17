use ndarray::Array1;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::jump::hawkes_jd::HawkesJd;
use stochastic_rs_stochastic::params::Horizon;
use stochastic_rs_stochastic::process::hawkes::Hawkes;
use stochastic_rs_stochastic::process::multivariate_hawkes::MultivariateHawkes;
use stochastic_rs_stochastic::process::poisson::Poisson;

use super::*;

/// The Hawkes jump diffusion carries its own intensity as a second component
/// the kernel excites and mean-reverts. At most one jump per step, as the
/// host's Bernoulli test takes it, so what the terminal mean pins is that the
/// device's excitement loop matches the host's.
#[test]
fn hawkes_jump_diffusion_agrees_with_the_cpu_law() {
  let build = || {
    HawkesJd::<f32, _>::new(
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

/// Poisson in count mode is a running sum of exponential inter-arrival times,
/// which the kernel draws by inverse CDF from its own uniform. The terminal
/// arrival time of `n - 1` of them is Gamma(n - 1, 1/lambda), so its mean and
/// spread are what carry the law. Horizon mode has no grid and stays on the
/// host whatever the backend, which the last assertion pins.
#[test]
fn poisson_arrivals_agree_with_the_cpu_law() {
  let build = || Poisson::<f32, _>::new(4.0, Horizon::Events(N - 1), Deterministic::new(71));
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
  let horizon = Poisson::<f32, _>::new(4.0, Horizon::Until(1.0), Deterministic::new(71))
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
  let build = || {
    Hawkes::<f32, _>::new(
      1.0,
      0.5,
      1.5,
      Horizon::Events(EVENTS),
      Deterministic::new(157),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  assert!(device.iter().all(|p| p.len() == EVENTS + 1));
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
  let (pairs, stuck) = (PATHS * EVENTS, ties(&device));
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
  let build =
    || Hawkes::<f32, _>::new(1.0, 0.5, 1.5, Horizon::Until(10.0), Deterministic::new(163));
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
      Horizon::Events(EVENTS),
      Deterministic::new(163),
    )
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
      Horizon::Events(16),
      Deterministic::new(167),
    )
  };
  assert_eq!(pairs().on::<Device>().sample_par(4), pairs().sample_par(4));
  let three = || {
    MultivariateHawkes::<f32, _>::new(
      array![1.0_f32, 0.7, 0.5],
      array![[0.3_f32, 0.1, 0.1], [0.1, 0.3, 0.1], [0.1, 0.1, 0.3]],
      array![[2.0_f32; 3], [2.0; 3], [2.0; 3]],
      Horizon::Events(16),
      Deterministic::new(173),
    )
  };
  assert_eq!(three().on::<Device>().sample_par(4), three().sample_par(4));
  let horizon = || {
    MultivariateHawkes::<f32, _>::new(
      array![1.0_f32, 0.7],
      array![[0.3_f32, 0.2], [0.1, 0.4]],
      array![[1.5_f32, 1.5], [2.0, 2.0]],
      Horizon::Until(3.0),
      Deterministic::new(179),
    )
  };
  assert_eq!(
    horizon().on::<Device>().sample_par(4),
    horizon().sample_par(4)
  );
  assert_eq!(horizon().on::<Device>().sample(), horizon().sample());
}
