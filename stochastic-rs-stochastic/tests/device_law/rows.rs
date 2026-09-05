//! The models that report a matrix of independent rows — a batch of forward
//! rates, a batch of affine factors — each row a small family of its own with
//! per-row scalars. They reach a device one launch per row, so what these
//! cases pin beyond the law is that row `i` got row `i`'s scalar and its own
//! stream: a launch that reused one row's parameter, or one row's seed, for
//! every row would keep each row's marginal plausible and still be wrong.

use ndarray::Array2;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::interest::adg::Adg;
use stochastic_rs_stochastic::interest::bgm::Bgm;
use stochastic_rs_stochastic::interest::wu_zhang::WuZhangD;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::Device;
use super::common::M;
use super::common::agrees;

const N: usize = 253;

fn row_mean(paths: &[Array2<f32>], row: usize) -> f64 {
  let last = paths[0].ncols() - 1;
  paths.iter().map(|p| p[(row, last)] as f64).sum::<f64>() / paths.len() as f64
}

fn row_spread(paths: &[Array2<f32>], row: usize) -> f64 {
  let last = paths[0].ncols() - 1;
  let n = paths.len() as f64;
  let mean = row_mean(paths, row);
  (paths
    .iter()
    .map(|p| (p[(row, last)] as f64 - mean).powi(2))
    .sum::<f64>()
    / n)
    .sqrt()
}

/// Correlation between two rows' terminal values across the batch: the rows
/// are independent, so a launch that seeded every row alike shows up here as
/// a correlation near one while every marginal stays right.
fn row_corr(paths: &[Array2<f32>], a: usize, b: usize) -> f64 {
  let last = paths[0].ncols() - 1;
  let (ma, mb) = (row_mean(paths, a), row_mean(paths, b));
  let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
  for p in paths {
    let (x, y) = (p[(a, last)] as f64 - ma, p[(b, last)] as f64 - mb);
    sab += x * y;
    saa += x * x;
    sbb += y * y;
  }
  sab / (saa * sbb).sqrt()
}

/// Each rate is a driftless proportional diffusion, so its terminal mean is
/// its own start and its terminal spread grows with its own `lambda`; the
/// three rows are given distinct values of both so a row that read its
/// neighbour's would move.
#[test]
fn bgm_agrees_with_the_cpu_law() {
  let build = || {
    Bgm::<f32, _>::new(
      array![0.15, 0.30, 0.45],
      array![0.03, 0.05, 0.07],
      3,
      Some(1.0),
      N,
      Deterministic::new(41),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  let host = build().sample_par(M);
  assert_eq!(device.len(), M);
  assert_eq!(device[0].dim(), (3, N));
  for (i, x0) in [0.03f32, 0.05, 0.07].iter().enumerate() {
    assert_eq!(device[0][(i, 0)], *x0, "row {i} starts at its own x0");
  }
  assert!(
    device.iter().all(|p| p.iter().all(|v| v.is_finite())),
    "BGM: a device path left the reals"
  );
  for i in 0..3 {
    let (h, d) = (row_mean(&host, i), row_mean(&device, i));
    assert!(
      (h - d).abs() < 0.002,
      "BGM row {i} terminal mean: host {h}, device {d}"
    );
    agrees(
      row_spread(&host, i),
      row_spread(&device, i),
      0.08,
      &format!("BGM row {i} terminal spread"),
    );
  }
  for (a, b) in [(0, 1), (1, 2), (0, 2)] {
    let c = row_corr(&device, a, b);
    assert!(
      c.abs() < 0.08,
      "BGM rows {a} and {b} are correlated on the device ({c}): the rows share a stream"
    );
  }
}

/// Each factor mean-reverts under two shared curves and is observed through a
/// quadratic map of three more, with its own diffusion scale and start. The
/// curves are given slopes large enough that a slot read from its neighbour
/// — `k` for `theta`, `b` for `c` — moves the terminal mean well past the
/// tolerance, and the scales are distinct so a reused one moves the spread.
#[test]
fn adg_agrees_with_the_cpu_law() {
  fn k(t: f32) -> f32 {
    0.01 + 0.02 * t
  }
  fn theta(t: f32) -> f32 {
    0.5 + 0.2 * t
  }
  fn phi(t: f32) -> f32 {
    0.002 + 0.01 * t
  }
  fn b(t: f32) -> f32 {
    0.8 + 0.4 * t
  }
  fn c(t: f32) -> f32 {
    2.0 + 1.0 * t
  }
  let build = || {
    Adg::<f32, _>::new(
      k as fn(f32) -> f32,
      theta as fn(f32) -> f32,
      array![0.01, 0.02, 0.03],
      phi as fn(f32) -> f32,
      b as fn(f32) -> f32,
      c as fn(f32) -> f32,
      N,
      3,
      array![0.02, 0.03, 0.04],
      Some(1.0),
      Deterministic::new(43),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  let host = build().sample_par(M);
  assert_eq!(device.len(), M);
  assert_eq!(device[0].dim(), (3, N));
  for (i, x0) in [0.02f32, 0.03, 0.04].iter().enumerate() {
    let observed = phi(0.0) + b(0.0) * x0 + c(0.0) * x0 * x0;
    assert!(
      (device[0][(i, 0)] - observed).abs() < 1e-6,
      "row {i} starts at the observation of its own x0: {} vs {observed}",
      device[0][(i, 0)]
    );
  }
  assert!(
    device.iter().all(|p| p.iter().all(|v| v.is_finite())),
    "ADG: a device path left the reals"
  );
  for i in 0..3 {
    let (h, d) = (row_mean(&host, i), row_mean(&device, i));
    assert!(
      (h - d).abs() < 0.002,
      "ADG row {i} terminal mean: host {h}, device {d}"
    );
    agrees(
      row_spread(&host, i),
      row_spread(&device, i),
      0.08,
      &format!("ADG row {i} terminal spread"),
    );
  }
  for (a, b) in [(0, 1), (1, 2), (0, 2)] {
    let corr = row_corr(&device, a, b);
    assert!(
      corr.abs() < 0.08,
      "ADG rows {a} and {b} are correlated on the device ({corr}): the rows share a stream"
    );
  }
}

/// Each pair is a rate driven by its own square-root variance; the matrix
/// holds the rates in the first `xn` rows and the variances in the next. The
/// variance's terminal mean is what pins `alpha` and `beta`, its spread
/// `nu`, and the rate's spread `lambda` times the variance level — so each of
/// the four per-pair scalars has a statistic that moves if it is misread.
#[test]
fn wu_zhang_agrees_with_the_cpu_law() {
  let build = || {
    WuZhangD::<f32, _>::new(
      array![0.02, 0.04, 0.06],
      array![2.0, 1.5, 1.0],
      array![0.2, 0.3, 0.4],
      array![0.6, 0.8, 1.0],
      array![0.03, 0.04, 0.05],
      array![0.03, 0.035, 0.04],
      3,
      Some(1.0),
      N,
      Deterministic::new(47),
    )
  };
  // Four times the usual batch: the second pair's variance sits near the
  // Feller boundary (`2 beta alpha = 0.12` against `nu^2 = 0.09`), so its
  // terminal mean's standard error at `M` paths is a third of the tolerance
  // below, which is too thin to call an agreement between two independent
  // draws. The tolerance stays; the path count carries it.
  const PATHS: usize = 4 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  assert_eq!(device[0].dim(), (6, N));
  for (i, (x0, v0)) in [(0.03f32, 0.03f32), (0.04, 0.035), (0.05, 0.04)]
    .iter()
    .enumerate()
  {
    assert_eq!(device[0][(i, 0)], *x0, "rate row {i} starts at its own x0");
    assert_eq!(
      device[0][(3 + i, 0)],
      *v0,
      "variance row {i} starts at its own v0"
    );
  }
  assert!(
    device
      .iter()
      .all(|p| p.iter().all(|v| v.is_finite() && *v >= 0.0)),
    "Wu-Zhang: a device path left the non-negative reals"
  );
  for row in 0..6 {
    let (h, d) = (row_mean(&host, row), row_mean(&device, row));
    assert!(
      (h - d).abs() < 0.002,
      "Wu-Zhang row {row} terminal mean: host {h}, device {d}"
    );
    agrees(
      row_spread(&host, row),
      row_spread(&device, row),
      0.10,
      &format!("Wu-Zhang row {row} terminal spread"),
    );
  }
  // Different pairs are independent; a rate and its own variance are not.
  for (a, b) in [(0, 1), (1, 2), (3, 4), (0, 5)] {
    let c = row_corr(&device, a, b);
    assert!(
      c.abs() < 0.08,
      "Wu-Zhang rows {a} and {b} are correlated on the device ({c}): the pairs share a stream"
    );
  }
}
