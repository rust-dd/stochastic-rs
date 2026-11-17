use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::volatility::bergomi::Bergomi;
use stochastic_rs_stochastic::volatility::multifactor_sabr::MultifactorSabr;
use stochastic_rs_stochastic::volatility::sabr::Sabr;

use super::*;

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
      0.2,
      100.0,
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

/// The dynamic SABR is the first family whose coefficients all arrive as
/// curves: the host resolves its term structure per grid point and the kernel
/// reads three of them per step. Two knots are used, so the tabulation has to
/// change value twice, and a launch that bound only the first curve — or
/// combined the shocks under the wrong one — shows up in the spreads below.
/// A bucket read one step out of phase does not: it moves every statistic
/// here by less than the noise floor, so the tabulation's phase is pinned
/// exactly, against the host's own bucket lookup, in the module's unit tests
/// rather than statistically.
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
