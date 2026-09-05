//! The multi-component processes: the engine steps every component in one
//! kernel, so what a device has to reproduce is each component's own law and
//! the boundary its family promises.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::traits::Fn1D;
use stochastic_rs_stochastic::correlation::heston_stoch_corr::HestonStochCorr;
use stochastic_rs_stochastic::diffusion::fouque::FouqueOU2D;
use stochastic_rs_stochastic::interest::duffie_kan::DuffieKan;
use stochastic_rs_stochastic::interest::duffie_kan_jump_exp::DuffieKanJumpExp;
use stochastic_rs_stochastic::interest::hull_white_2f::HullWhite2F;
use stochastic_rs_stochastic::process::cbms::Cbms;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::bates_svj::BatesSvj;
use stochastic_rs_stochastic::volatility::bergomi::Bergomi;
use stochastic_rs_stochastic::volatility::bns::Bns;
use stochastic_rs_stochastic::volatility::double_heston::DoubleHeston;
use stochastic_rs_stochastic::volatility::heston::Heston;
use stochastic_rs_stochastic::volatility::heston_log::HestonLog;
use stochastic_rs_stochastic::volatility::heston2d::Heston2D;
use stochastic_rs_stochastic::volatility::hkde::Hkde;
use stochastic_rs_stochastic::volatility::multifactor_sabr::MultifactorSabr;
use stochastic_rs_stochastic::volatility::sabr::Sabr;

use super::common::Device;
use super::common::agrees;

/// Fewer paths than the scalar cases use: a system draw carries two paths.
const M: usize = 3_000;

fn terminal_mean_of<const D: usize>(paths: &[[Array1<f32>; D]], component: usize) -> f64 {
  let last = paths[0][component].len() - 1;
  paths.iter().map(|p| p[component][last] as f64).sum::<f64>() / paths.len() as f64
}

fn terminal_mean(paths: &[[Array1<f32>; 2]], component: usize) -> f64 {
  let last = paths[0][component].len() - 1;
  paths.iter().map(|p| p[component][last] as f64).sum::<f64>() / paths.len() as f64
}

/// The spot is a martingale under a zero drift and the variance reverts to
/// theta, so both components have a terminal mean two independent streams can
/// agree on. The variance also may not go negative: its family truncates.
#[test]
fn heston_agrees_with_the_cpu_law() {
  let build = || {
    Heston::<f32, _>::new(
      Some(100.0),
      Some(0.04),
      2.0,
      0.04,
      0.3,
      -0.7,
      0.0,
      253,
      Some(1.0),
      HestonPow::Sqrt,
      Some(false),
      Deterministic::new(17),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  assert_eq!(device.len(), M);
  assert!(
    device
      .iter()
      .all(|p| p[0].iter().chain(p[1].iter()).all(|v| v.is_finite())),
    "a device path left the reals"
  );
  assert!(
    device.iter().all(|p| p[1].iter().all(|&v| v >= 0.0)),
    "the truncated variance went negative"
  );
  let host = build().sample_par(M);
  agrees(
    terminal_mean(&host, 0),
    terminal_mean(&device, 0),
    0.03,
    "Heston terminal spot",
  );
  agrees(
    terminal_mean(&host, 1),
    terminal_mean(&device, 1),
    0.06,
    "Heston terminal variance",
  );
}

/// The reflected variance is the other family, and it may not go negative
/// either.
#[test]
fn reflected_heston_agrees_with_the_cpu_law() {
  let build = || {
    Heston::<f32, _>::new(
      Some(100.0),
      Some(0.04),
      2.0,
      0.04,
      0.3,
      -0.7,
      0.0,
      253,
      Some(1.0),
      HestonPow::Sqrt,
      Some(true),
      Deterministic::new(17),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  assert!(
    device.iter().all(|p| p[1].iter().all(|&v| v >= 0.0)),
    "the reflected variance went negative"
  );
  agrees(
    terminal_mean(&build().sample_par(M), 1),
    terminal_mean(&device, 1),
    0.06,
    "reflected Heston terminal variance",
  );
}

/// SABR steps its volatility by the exact log-normal solution, so the device
/// path must stay positive for the same reason the host's does, and the
/// forward is driftless.
#[test]
fn sabr_agrees_with_the_cpu_law() {
  let build = || {
    Sabr::<f32, _>::new(
      0.4,
      0.5,
      -0.4,
      253,
      Some(100.0),
      Some(0.2),
      Some(1.0),
      Deterministic::new(19),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  assert!(
    device.iter().all(|p| p[1].iter().all(|&v| v > 0.0)),
    "the exact volatility step went non-positive"
  );
  agrees(
    terminal_mean(&build().sample_par(M), 0),
    terminal_mean(&device, 0),
    0.03,
    "SABR terminal forward",
  );
  agrees(
    terminal_mean(&build().sample_par(M), 1),
    terminal_mean(&device, 1),
    0.06,
    "SABR terminal volatility",
  );
}

/// The Bergomi variance is a function of the running sum of its increments,
/// which the device steps as state of its own; what the comparison pins is
/// that this reproduces the host's cumulative form.
#[test]
fn bergomi_agrees_with_the_cpu_law() {
  let build = || {
    Bergomi::<f32, _>::new(
      0.5,
      Some(0.2),
      Some(100.0),
      0.02,
      -0.6,
      253,
      Some(1.0),
      Deterministic::new(21),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  assert!(
    device.iter().all(|p| p[1].iter().all(|&v| v > 0.0)),
    "the log-normal variance went non-positive"
  );
  let host = build().sample_par(M);
  agrees(
    terminal_mean(&host, 0),
    terminal_mean(&device, 0),
    0.04,
    "Bergomi terminal spot",
  );
  agrees(
    terminal_mean(&host, 1),
    terminal_mean(&device, 1),
    0.08,
    "Bergomi terminal variance",
  );
}

/// Two Ornstein-Uhlenbeck factors on one clock, both reverting toward zero.
/// Their terminal means are near zero, where a relative comparison is the
/// ratio of two numbers that are both mostly sampling noise, so what is
/// compared is the spread each factor's own reversion speed sets.
#[test]
fn two_scale_ornstein_uhlenbeck_agrees_with_the_cpu_law() {
  let build = || {
    FouqueOU2D::<f32, _>::new(
      1.0,
      0.3,
      0.25,
      -0.2,
      253,
      Some(0.0),
      Some(0.0),
      Some(1.0),
      Deterministic::new(23),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  let host = build().sample_par(M);
  let spread = |paths: &[[Array1<f32>; 2]], c: usize| {
    let last = paths[0][c].len() - 1;
    let mean = terminal_mean(paths, c);
    (paths
      .iter()
      .map(|p| (p[c][last] as f64 - mean).powi(2))
      .sum::<f64>()
      / paths.len() as f64)
      .sqrt()
  };
  for (c, what) in [(0usize, "slow"), (1, "fast")] {
    agrees(
      spread(&host, c),
      spread(&device, c),
      0.06,
      &format!("{what} factor terminal spread"),
    );
  }
}

/// The log-price form keeps the spot positive by construction, so that is
/// pinned alongside the two terminal means.
#[test]
fn log_heston_agrees_with_the_cpu_law() {
  let build = || {
    HestonLog::<f32, _>::new(
      Some(0.0),
      None,
      None,
      None,
      2.0,
      0.04,
      0.3,
      -0.7,
      253,
      Some(100.0),
      Some(0.04),
      Some(1.0),
      Some(false),
      Deterministic::new(29),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  assert!(
    device.iter().all(|p| p[0].iter().all(|&v| v > 0.0)),
    "the log-price form let the spot reach zero"
  );
  let host = build().sample_par(M);
  agrees(
    terminal_mean(&host, 0),
    terminal_mean(&device, 0),
    0.03,
    "log-Heston terminal spot",
  );
  agrees(
    terminal_mean(&host, 1),
    terminal_mean(&device, 1),
    0.06,
    "log-Heston terminal variance",
  );
}

/// Two variance factors driving one spot: both must stay non-negative and
/// both terminal means must agree.
#[test]
fn double_heston_agrees_with_the_cpu_law() {
  let build = || {
    DoubleHeston::<f32, _>::new(
      Some(100.0),
      Some(0.04),
      Some(0.02),
      2.0,
      0.04,
      0.3,
      -0.7,
      1.0,
      0.02,
      0.2,
      -0.3,
      0.0,
      253,
      Some(1.0),
      Some(false),
      Deterministic::new(31),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  assert!(
    device
      .iter()
      .all(|p| p[1].iter().chain(p[2].iter()).all(|&v| v >= 0.0)),
    "a truncated variance went negative"
  );
  let host = build().sample_par(M);
  for (c, tol, what) in [
    (0usize, 0.04, "spot"),
    (1, 0.08, "first variance"),
    (2, 0.08, "second variance"),
  ] {
    agrees(
      terminal_mean_of(&host, c),
      terminal_mean_of(&device, c),
      tol,
      &format!("double Heston terminal {what}"),
    );
  }
}

/// The correlation component is stepped unbounded and reported through a
/// `tanh`, so it cannot leave `(-1, 1)` however far the state wanders.
#[test]
fn stochastic_correlation_heston_agrees_with_the_cpu_law() {
  let build = || {
    HestonStochCorr::<f32, _>::new(
      0.02,
      100.0,
      0.04,
      2.0,
      0.04,
      0.3,
      -0.3,
      1.0,
      -0.3,
      0.2,
      0.1,
      253,
      Some(1.0),
      Deterministic::new(37),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  assert!(
    device
      .iter()
      .all(|p| p[2].iter().all(|&v| (-1.0..=1.0).contains(&v))),
    "the reported correlation left [-1, 1]"
  );
  let host = build().sample_par(M);
  for (c, tol, what) in [
    (0usize, 0.04, "spot"),
    (1, 0.08, "variance"),
    (2, 0.15, "correlation"),
  ] {
    agrees(
      terminal_mean_of(&host, c),
      terminal_mean_of(&device, c),
      tol,
      &format!("stochastic-correlation Heston terminal {what}"),
    );
  }
}

/// A correlated Brownian pair: both marginals are driftless, so the spread is
/// what carries the law, and the pair's own correlation is what the second
/// component's step exists to reproduce.
#[test]
fn correlated_brownian_agrees_with_the_cpu_law() {
  let build = || Cbms::<f32, _>::new(-0.5, 253, Some(1.0), Deterministic::new(61));
  let device = build().on::<Device>().sample_par(M);
  let host = build().sample_par(M);
  let corr = |paths: &[[Array1<f32>; 2]]| {
    let last = paths[0][0].len() - 1;
    let (mut sxy, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64);
    for p in paths {
      let (x, y) = (p[0][last] as f64, p[1][last] as f64);
      sxy += x * y;
      sxx += x * x;
      syy += y * y;
    }
    sxy / (sxx * syy).sqrt()
  };
  agrees(
    corr(&host),
    corr(&device),
    0.10,
    "correlated Brownian terminal",
  );
}

/// The two-factor Hull-White model reads its mean-reversion level from the
/// curve, so an off-by-one in that indexing moves the terminal rate.
#[test]
fn two_factor_hull_white_agrees_with_the_cpu_law() {
  let build = || {
    HullWhite2F::<f32, _>::new(
      Fn1D::Native(|t: f32| 0.02 + 0.03 * t),
      1.0,
      0.01,
      0.005,
      -0.4,
      0.5,
      Some(0.02),
      Some(1.0),
      253,
      Deterministic::new(67),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  let host = build().sample_par(M);
  agrees(
    terminal_mean(&host, 0),
    terminal_mean(&device, 0),
    0.06,
    "two-factor Hull-White terminal rate",
  );
}

/// Both Duffie-Kan factors share one affine volatility, which the step binds
/// once and both components read.
#[test]
fn duffie_kan_agrees_with_the_cpu_law() {
  let build = || {
    DuffieKan::<f32, _>::new(
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
      253,
      Some(0.03),
      Some(0.01),
      Some(1.0),
      Deterministic::new(71),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  let host = build().sample_par(M);
  for (c, what) in [(0usize, "rate"), (1, "factor")] {
    agrees(
      terminal_mean(&host, c),
      terminal_mean(&device, c),
      0.08,
      &format!("Duffie-Kan terminal {what}"),
    );
  }
}

/// Two Heston assets under one Cholesky factor: four components in one
/// launch. Both variances stay non-negative and every component's terminal
/// mean must agree.
#[test]
fn two_asset_heston_agrees_with_the_cpu_law() {
  let build = || {
    Heston2D::<f32, _>::new(
      [Some(4.6), Some(4.6)],
      [Some(0.04), Some(0.03)],
      [0.0, 0.0],
      [0.04, 0.03],
      [2.0, 1.5],
      [0.3, 0.25],
      [-0.6, 0.2, 0.1, -0.2, 0.3, 0.15],
      253,
      Some(1.0),
      Some(false),
      Deterministic::new(83),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  assert!(
    device
      .iter()
      .all(|p| p[1].iter().chain(p[3].iter()).all(|&v| v >= 0.0)),
    "a truncated variance went negative"
  );
  let host = build().sample_par(M);
  for (c, tol, what) in [
    (0usize, 0.02, "first log-price"),
    (1, 0.08, "first variance"),
    (2, 0.02, "second log-price"),
    (3, 0.08, "second variance"),
  ] {
    agrees(
      terminal_mean_of(&host, c),
      terminal_mean_of(&device, c),
      tol,
      &format!("two-asset Heston terminal {what}"),
    );
  }
}

/// A Heston variance under a jumping log-price: the device draws its own
/// Poisson count and aggregates the jump sizes, so both the variance's law
/// and the jump-compensated spot must agree.
#[test]
fn bates_agrees_with_the_cpu_law() {
  let build = || {
    BatesSvj::<f32, _>::new(
      Some(0.02),
      None,
      None,
      None,
      3.0,
      -0.05,
      0.1,
      0.08,
      2.0,
      0.3,
      -0.7,
      253,
      Some(100.0),
      Some(0.04),
      Some(1.0),
      Some(false),
      Deterministic::new(97),
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
    0.04,
    "Bates terminal spot",
  );
  agrees(
    terminal_mean(&host, 1),
    terminal_mean(&device, 1),
    0.08,
    "Bates terminal variance",
  );
}

/// The quadratic-exponential scheme draws its variance from a moment-matched
/// law with two branches, and the kernel picks between them with a uniform of
/// its own. The variance is non-negative by construction on both branches.
#[test]
fn andersen_qe_heston_agrees_with_the_cpu_law() {
  let build = || {
    Heston::<f32, _>::new(
      Some(100.0),
      Some(0.04),
      2.0,
      0.04,
      0.3,
      -0.7,
      0.0,
      253,
      Some(1.0),
      HestonPow::Sqrt,
      None,
      Deterministic::new(101),
    )
    .qe()
  };
  let device = build().on::<Device>().sample_par(M);
  assert!(
    device.iter().all(|p| p[1].iter().all(|&v| v >= 0.0)),
    "the moment-matched variance went negative"
  );
  assert!(
    device.iter().all(|p| p[0].iter().all(|&v| v > 0.0)),
    "the log-spot form let the spot reach zero"
  );
  let host = build().sample_par(M);
  agrees(
    terminal_mean(&host, 0),
    terminal_mean(&device, 0),
    0.03,
    "QE Heston terminal spot",
  );
  agrees(
    terminal_mean(&host, 1),
    terminal_mean(&device, 1),
    0.06,
    "QE Heston terminal variance",
  );
}

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
      Some(100.0),
      Some(1.0),
      Some(false),
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
      Some(100.0),
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

/// The dynamic SABR is the first family whose coefficients all arrive as
/// curves: the host resolves its term structure per grid point and the kernel
/// reads three of them per step. Two knots are used, so the tabulation has to
/// change value twice — a launch that bound only the first curve, or read the
/// buckets one step out of phase, would show up in the terminal spread.
#[test]
fn dynamic_sabr_agrees_with_the_cpu_law() {
  let build = || {
    MultifactorSabr::<f32, _>::new(
      Some(0.04),
      Some(0.3),
      vec![0.3, 0.7],
      vec![0.5, 0.7, 0.9],
      vec![-0.6, -0.2, 0.3],
      vec![0.8, 0.4, 0.2],
      253,
      Some(1.0),
      Deterministic::new(83),
    )
  };
  // Four times the usual batch: the volatility is log-normal, and the sample
  // standard deviation of a log-normal is tail-driven — at `M` paths it moves
  // by several percent between seeds on the host alone, which is wider than
  // the agreement this case is meant to assert. The tolerance stays tight and
  // the path count carries it.
  const PATHS: usize = 4 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  assert_eq!(device[0][0].len(), 253);
  assert_eq!(device[0][0][0], 0.04, "the forward starts at f0");
  assert_eq!(device[0][1][0], 0.3, "the volatility starts at alpha0");
  assert!(
    device.iter().all(|p| p
      .iter()
      .all(|row| row.iter().all(|v| v.is_finite() && *v >= 0.0))),
    "dynamic SABR: a device path left the non-negative reals"
  );
  let spread = |paths: &[[Array1<f32>; 2]], c: usize| {
    let last = paths[0][c].len() - 1;
    let n = paths.len() as f64;
    let mean = paths.iter().map(|p| p[c][last] as f64).sum::<f64>() / n;
    (paths
      .iter()
      .map(|p| (p[c][last] as f64 - mean).powi(2))
      .sum::<f64>()
      / n)
      .sqrt()
  };
  agrees(
    terminal_mean(&host, 1),
    terminal_mean(&device, 1),
    0.06,
    "dynamic SABR terminal volatility",
  );
  // The volatility's spread is what pins `nu(t)`: its *mean* is `alpha0`
  // whatever `nu` is, since the exact step for `d(alpha) = nu alpha dW` is a
  // martingale — so a launch reading the wrong curve into `nu` passes a mean
  // comparison and fails this one.
  agrees(
    spread(&host, 1),
    spread(&device, 1),
    0.06,
    "dynamic SABR terminal volatility spread",
  );
  agrees(
    spread(&host, 0),
    spread(&device, 0),
    0.10,
    "dynamic SABR terminal forward spread",
  );
}
