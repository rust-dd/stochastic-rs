use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::interest::duffie_kan_jump_exp::DuffieKanJumpExp;
use stochastic_rs_stochastic::jump::bates::Bates;
use stochastic_rs_stochastic::params::Drift;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::volatility::bns::Bns;
use stochastic_rs_stochastic::volatility::hkde::Hkde;
use stochastic_rs_stochastic::volatility::svcgmy::Svcgmy;

use super::*;

/// Kou's double-exponential jump sizes have no closed-form aggregate, so the
/// kernel sums them in a bounded loop. What this pins is that the sum it
/// builds carries the same law the host's own per-jump draws build.
#[test]
fn kou_jump_heston_agrees_with_the_cpu_law() {
  let build = || {
    Hkde::<f32, _>::new(
      0.02,
      2.0,
      0.04,
      0.3,
      -0.7,
      0.04,
      3.0,
      0.4,
      25.0,
      20.0,
      253,
      100.0,
      Some(1.0),
      VarianceBoundary::Floor,
      Deterministic::new(131),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  assert!(
    device.iter().all(|p| p[1].iter().all(|&v| v >= 0.0)),
    "the truncated variance went negative"
  );
  assert!(
    device.iter().all(|p| p[0].iter().all(|&v| v > 0.0)),
    "the log-price form let the spot reach zero"
  );
  let host = build().sample_par(M);
  agrees(
    terminal_mean(&host, 0),
    terminal_mean(&device, 0),
    0.05,
    "Kou-jump Heston terminal spot",
  );
  agrees(
    terminal_mean(&host, 1),
    terminal_mean(&device, 1),
    0.08,
    "Kou-jump Heston terminal variance",
  );
}

/// The jump-augmented Duffie-Kan model. The host walks its jump times
/// sequentially and the device draws a Poisson count per step; the waiting
/// time is memoryless, so the two are the same law, which is what the second
/// factor's terminal mean pins.
#[test]
fn jump_duffie_kan_agrees_with_the_cpu_law() {
  let build = || {
    DuffieKanJumpExp::<f32, _>::new(
      0.5,
      0.2,
      0.1,
      -0.3,
      -0.5,
      0.1,
      0.02,
      0.1,
      0.05,
      -0.3,
      0.01,
      0.08,
      3.0,
      0.01,
      253,
      Some(0.03),
      Some(0.01),
      Some(1.0),
      Deterministic::new(137),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  let host = build().sample_par(M);
  for (c, what) in [(0usize, "rate"), (1, "jumping factor")] {
    agrees(
      terminal_mean(&host, c),
      terminal_mean(&device, c),
      0.10,
      &format!("jump Duffie-Kan terminal {what}"),
    );
  }
}

/// The Barndorff-Nielsen-Shephard variance is driven by a compound-Poisson
/// subordinator of gamma jumps, which the kernel draws as one gamma whose
/// shape is the step's jump count times a single jump's.
///
/// The variance this model reaches is large — of order one, not of order a
/// percent — so the spot is a lognormal whose log has a standard deviation
/// near two, and whose *mean* is therefore set by a handful of paths in the
/// right tail. What is compared for the spot is the mean of its logarithm,
/// which is the drift the two sides actually share; the variance itself is
/// compared directly.
#[test]
fn barndorff_nielsen_shephard_agrees_with_the_cpu_law() {
  let build = || {
    Bns::<f32, _>::new(
      100.0,
      0.04,
      2.0,
      0.02,
      8.0,
      1.0,
      253,
      Some(1.0),
      Deterministic::new(229),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  assert!(
    device.iter().all(|p| p[1].iter().all(|&v| v >= 0.0)),
    "the gamma-driven variance went negative"
  );
  assert!(
    device.iter().all(|p| p[0].iter().all(|&v| v > 0.0)),
    "the log-Euler asset let the spot reach zero"
  );
  let host = build().sample_par(M);
  let mean_log = |paths: &[[Array1<f32>; 2]]| {
    let last = paths[0][0].len() - 1;
    paths.iter().map(|p| (p[0][last] as f64).ln()).sum::<f64>() / paths.len() as f64
  };
  agrees(
    mean_log(&host),
    mean_log(&device),
    0.05,
    "BNS terminal log-spot",
  );
  agrees(
    terminal_mean(&host, 1),
    terminal_mean(&device, 1),
    0.08,
    "BNS terminal variance",
  );
}

/// Bates (1996) in the spot: the jumps multiply the spot by `1 + Y` for a
/// normal `Y`, compounded across the jumps a step sees, which the kernels
/// draw as a product law rather than an aggregated sum.
#[test]
fn bates_1996_agrees_with_the_cpu_law() {
  for boundary in [VarianceBoundary::Floor, VarianceBoundary::Reflect] {
    let build = || {
      Bates::<f32, _, _>::new(
        Drift::Mu(0.02),
        3.0,
        -0.02,
        0.08,
        2.0,
        0.3,
        -0.7,
        SimdNormal::<f32>::new(-0.02, 0.05),
        253,
        100.0,
        0.04,
        Some(1.0),
        boundary,
        Deterministic::new(97),
      )
    };
    const PATHS: usize = 3 * M;
    let device = build().on::<Device>().sample_par(PATHS);
    assert!(
      device.iter().all(|p| p[1].iter().all(|&v| v >= 0.0)),
      "the variance went negative (boundary: {boundary:?})"
    );
    let host = build().sample_par(PATHS);
    agrees(
      terminal_mean(&host, 0),
      terminal_mean(&device, 0),
      0.02,
      "Bates 1996 terminal spot",
    );
    agrees(
      terminal_mean(&host, 1),
      terminal_mean(&device, 1),
      0.05,
      "Bates 1996 terminal variance",
    );
    let spread = |paths: &[[Array1<f32>; 2]]| {
      let n = paths.len() as f64;
      let mean = terminal_mean(paths, 0);
      (paths
        .iter()
        .map(|p| (p[0][252] as f64 - mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt()
    };
    agrees(
      spread(&host),
      spread(&device),
      0.06,
      "Bates 1996 spot spread",
    );
  }
}

/// Where the Feller condition fails the two boundary treatments part: the
/// truncated variance sits at zero for a good share of paths while the
/// reflected one bounces off it. Each variant has to agree with its own host
/// on that share, and the two shares have to differ — which is what pins the
/// variant a launch runs, not just the shared drift.
#[test]
fn bates_1996_reflection_moves_the_boundary() {
  let share_at_zero = |paths: &[[Array1<f32>; 2]]| {
    paths.iter().filter(|p| p[1][252] < 1e-4).count() as f64 / paths.len() as f64
  };
  let mut shares = [0.0f64; 2];
  for (k, boundary) in [VarianceBoundary::Floor, VarianceBoundary::Reflect]
    .into_iter()
    .enumerate()
  {
    let build = || {
      Bates::<f32, _, _>::new(
        Drift::Mu(0.02),
        0.0,
        0.0,
        0.005,
        2.0,
        0.6,
        -0.5,
        SimdNormal::<f32>::new(-0.02, 0.05),
        253,
        100.0,
        0.04,
        Some(1.0),
        boundary,
        Deterministic::new(101),
      )
    };
    const PATHS: usize = 3 * M;
    let device = build().on::<Device>().sample_par(PATHS);
    let host = build().sample_par(PATHS);
    let (h, d) = (share_at_zero(&host), share_at_zero(&device));
    assert!(
      (h - d).abs() < 0.03,
      "Bates 1996 share of variances at zero (boundary: {boundary:?}): host {h}, device {d}"
    );
    shares[k] = d;
  }
  assert!(
    shares[0] - shares[1] > 0.1,
    "truncation and reflection are indistinguishable here: {shares:?}"
  );
}

/// Several jumps a step is where a product law differs from a sum: with an
/// intensity above one per step, `∏(1 + Y) − 1` carries the cross terms a sum
/// drops, and with positive exponential sizes those terms lift the mean
/// log-spot by about a quarter over the horizon — a margin a summing kernel
/// would miss and a statistic whose standard error here is a hundredth of it,
/// where the spot's own mean sits under a tail too heavy to pin.
#[test]
fn bates_1996_compounds_several_jumps_a_step() {
  let build = || {
    Bates::<f32, _, _>::new(
      Drift::Mu(0.02),
      30.0,
      0.2,
      0.08,
      2.0,
      0.3,
      -0.5,
      SimdExp::<f32>::new(5.0),
      26,
      100.0,
      0.04,
      Some(1.0),
      VarianceBoundary::Floor,
      Deterministic::new(103),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert!(
    device.iter().all(|p| p[0].iter().all(|&s| s > 0.0)),
    "positive multiplicative jumps cannot take the spot below zero"
  );
  let mean_log_spot = |paths: &[[Array1<f32>; 2]]| {
    paths.iter().map(|p| (p[0][25] as f64).ln()).sum::<f64>() / paths.len() as f64
  };
  let (h, d) = (mean_log_spot(&host), mean_log_spot(&device));
  assert!(
    (h - d).abs() < 0.06,
    "Bates 1996 mean log-spot under compounding jumps: host {h}, device {d}"
  );
}

/// CGMY under a CIR variance: the variance takes the exact square-root step
/// on both machines, so its terminal law is one law; the jumps are sized in
/// their own step against that variance, so the log-price's spread — larger
/// where the variance ran high — is what pins the live sizing. The jump law
/// has a kurtosis above twenty at these parameters, so the spread is read off
/// the quartiles, which a heavy tail leaves alone, and the mean is held
/// against the interquartile range.
#[test]
fn stochastic_volatility_cgmy_agrees_with_the_cpu_law() {
  let build = || {
    Svcgmy::<f32, _>::new(
      2.0,
      6.0,
      0.5,
      2.0,
      0.04,
      0.2,
      0.3,
      253,
      256,
      Some(0.0),
      0.04,
      Some(1.0),
      Deterministic::new(233),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  assert!(
    device.iter().all(|p| p[1].iter().all(|&v| v >= 0.0)),
    "the exact variance step went negative"
  );
  // The reported log-price is `y + ρ v`, so its start is `x0` up to the
  // rounding of `−ρ v0 + ρ v0` in single precision.
  assert!(
    device
      .iter()
      .all(|p| p[0][0].abs() < 1e-6 && (p[1][0] - 0.04).abs() < 1e-7),
    "every path starts at x0 and v0"
  );
  agrees(
    terminal_mean(&host, 1),
    terminal_mean(&device, 1),
    0.03,
    "SV-CGMY terminal variance",
  );
  let spread = |paths: &[[Array1<f32>; 2]], row: usize| {
    let n = paths.len() as f64;
    let mean = terminal_mean(paths, row);
    (paths
      .iter()
      .map(|p| (p[row][252] as f64 - mean).powi(2))
      .sum::<f64>()
      / n)
      .sqrt()
  };
  agrees(
    spread(&host, 1),
    spread(&device, 1),
    0.06,
    "SV-CGMY variance spread",
  );
  let iqr = |paths: &[[Array1<f32>; 2]], at: usize| {
    let mut values: Vec<f64> = paths.iter().map(|p| p[0][at] as f64).collect();
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    values[values.len() * 3 / 4] - values[values.len() / 4]
  };
  agrees(
    iqr(&host, 252),
    iqr(&device, 252),
    0.06,
    "SV-CGMY log-price interquartile range",
  );
  agrees(
    iqr(&host, 63),
    iqr(&device, 63),
    0.06,
    "SV-CGMY quarter-horizon log-price interquartile range",
  );
  let scale = iqr(&host, 252);
  let (h, d) = (terminal_mean(&host, 0), terminal_mean(&device, 0));
  assert!(
    (h - d).abs() < 0.05 * scale,
    "SV-CGMY terminal log-price: host {h}, device {d} against a range of {scale}"
  );
}

/// Below one degree of freedom the host draws the variance from a Poisson
/// mixture the kernels do not carry, so the device build samples on the host
/// and is the host build to the bit.
#[test]
fn a_thin_variance_keeps_the_stochastic_volatility_cgmy_on_the_host() {
  let build = || {
    Svcgmy::<f32, _>::new(
      2.0,
      6.0,
      0.5,
      0.5,
      0.01,
      0.4,
      0.3,
      64,
      64,
      Some(0.0),
      0.04,
      Some(1.0),
      Deterministic::new(239),
    )
  };
  assert_eq!(build().on::<Device>().sample_par(8), build().sample_par(8));
}
