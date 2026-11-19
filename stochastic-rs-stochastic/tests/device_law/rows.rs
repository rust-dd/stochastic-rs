//! The models that report a matrix of independent rows, one launch per row: beyond the law,
//! these pin that row `i` got its own scalar and stream, which every marginal alone would miss.

use ndarray::Array2;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::multi_gbm::MultiGbm;
use stochastic_rs_stochastic::interest::adg::Adg;
use stochastic_rs_stochastic::interest::lmm::Lmm;
use stochastic_rs_stochastic::interest::wu_zhang::WuZhang;
use stochastic_rs_stochastic::noise::mcgns::Mcgns;
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

/// Correlation between two rows' terminal values across the batch: the rows are independent,
/// so a launch that seeded every row alike shows a correlation near one with right marginals.
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

/// Each factor reverts under two shared curves and is observed through a quadratic map of three
/// more. Steep curves and distinct scales make a slot or scale read from its neighbour visible.
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

/// Each pair is a rate driven by its own square-root variance, rates in the first `xn` rows.
/// The variance's mean pins `alpha` and `beta`, its spread `nu`, the rate's spread `lambda`.
#[test]
fn wu_zhang_agrees_with_the_cpu_law() {
  let build = || {
    WuZhang::<f32, _>::new(
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
  // Four times the usual batch: the second pair's variance sits near the Feller boundary
  // (`2 beta alpha = 0.12` against `nu^2 = 0.09`), and at `M` paths its mean's error is too wide.
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

/// Terminal log-returns of a basket: mean `(mu - sigma^2 / 2) t`, spread `sigma sqrt(t)` and
/// cross-correlation `rho`, all exact, so each parameter set has a statistic that pins it.
fn log_returns(paths: &[Array2<f32>], row: usize) -> Vec<f64> {
  let last = paths[0].ncols() - 1;
  paths
    .iter()
    .map(|p| (p[(row, last)] as f64 / p[(row, 0)] as f64).ln())
    .collect()
}

fn corr(a: &[f64], b: &[f64]) -> f64 {
  let n = a.len() as f64;
  let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
  let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
  for (x, y) in a.iter().zip(b) {
    sab += (x - ma) * (y - mb);
    saa += (x - ma).powi(2);
    sbb += (y - mb).powi(2);
  }
  sab / (saa * sbb).sqrt()
}

/// Three correlated assets in one launch under one Cholesky factor: a launch that combined the
/// shocks through the wrong factor, or none, keeps every marginal right and loses these.
#[test]
fn multi_gbm_agrees_with_the_cpu_law() {
  let build = || {
    MultiGbm::<f32, _>::new(
      array![0.05, 0.02, 0.08],
      array![0.2, 0.3, 0.15],
      array![[1.0, 0.6, -0.3], [0.6, 1.0, 0.2], [-0.3, 0.2, 1.0]],
      N,
      array![100.0, 50.0, 80.0],
      Some(1.0),
      Deterministic::new(53),
    )
  };
  const PATHS: usize = 4 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  assert_eq!(device[0].dim(), (3, N));
  for (i, x0) in [100.0f32, 50.0, 80.0].iter().enumerate() {
    assert_eq!(device[0][(i, 0)], *x0, "asset {i} starts at its own x0");
  }
  let (hl, dl): (Vec<_>, Vec<_>) = (
    (0..3).map(|i| log_returns(&host, i)).collect(),
    (0..3).map(|i| log_returns(&device, i)).collect(),
  );
  for i in 0..3 {
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let sd = |v: &[f64]| {
      let m = mean(v);
      (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    };
    assert!(
      (mean(&hl[i]) - mean(&dl[i])).abs() < 0.01,
      "basket asset {i} log-return mean: host {}, device {}",
      mean(&hl[i]),
      mean(&dl[i])
    );
    agrees(
      sd(&hl[i]),
      sd(&dl[i]),
      0.05,
      &format!("basket asset {i} log-return spread"),
    );
  }
  for (a, b, rho) in [(0, 1, 0.6), (1, 2, 0.2), (0, 2, -0.3)] {
    let (h, d) = (corr(&hl[a], &hl[b]), corr(&dl[a], &dl[b]));
    assert!(
      (h - d).abs() < 0.04 && (d - rho).abs() < 0.04,
      "basket assets {a},{b} correlation: host {h}, device {d}, requested {rho}"
    );
  }
}

/// A basket too wide for a launch stays on the host on any backend, silently: this pins the
/// sampler's shape and law so the fallback cannot break unnoticed.
#[test]
fn multi_gbm_wider_than_four_assets_still_samples() {
  let rho = Array2::<f32>::eye(5);
  let paths = MultiGbm::<f32, _>::new(
    array![0.01, 0.02, 0.03, 0.04, 0.05],
    array![0.1, 0.2, 0.3, 0.2, 0.1],
    rho,
    64,
    array![1.0, 2.0, 3.0, 4.0, 5.0],
    Some(1.0),
    Deterministic::new(59),
  )
  .on::<Device>()
  .sample_par(16);
  assert_eq!(paths.len(), 16);
  assert!(
    paths
      .iter()
      .all(|p| p.dim() == (5, 64) && p.iter().all(|v| v.is_finite()))
  );
}

/// Three correlated noise streams, every point a draw: the per-stream spread is `sqrt(dt)` and
/// the cross-correlation over all points the requested one, which a wrong factor loses.
#[test]
fn mcgns_agrees_with_the_cpu_law() {
  let build = || {
    Mcgns::<f32, _>::new(
      array![[1.0, 0.6, -0.3], [0.6, 1.0, 0.2], [-0.3, 0.2, 1.0]],
      N,
      Some(1.0),
      Deterministic::new(61),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  let host = build().sample_par(M);
  assert_eq!(device.len(), M);
  assert_eq!(device[0].dim(), (3, N));
  let flat = |paths: &[Array2<f32>], row: usize| -> Vec<f64> {
    paths
      .iter()
      .flat_map(|p| p.row(row).iter().map(|v| *v as f64).collect::<Vec<_>>())
      .collect()
  };
  let sd = |v: &[f64]| {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
  };
  let (hs, ds): (Vec<_>, Vec<_>) = (
    (0..3).map(|i| flat(&host, i)).collect(),
    (0..3).map(|i| flat(&device, i)).collect(),
  );
  for i in 0..3 {
    agrees(
      sd(&hs[i]),
      sd(&ds[i]),
      0.03,
      &format!("correlated noise stream {i} spread"),
    );
  }
  for (a, b, rho) in [(0, 1, 0.6), (1, 2, 0.2), (0, 2, -0.3)] {
    let (h, d) = (corr(&hs[a], &hs[b]), corr(&ds[a], &ds[b]));
    assert!(
      (h - d).abs() < 0.02 && (d - rho).abs() < 0.02,
      "noise streams {a},{b} correlation: host {h}, device {d}, requested {rho}"
    );
  }
}

/// Three forwards under the spot measure's drift coupling with unequal accruals: the first has
/// reset and never moves, the others freeze exactly at their resets, a third and five sixths in.
#[test]
fn lmm_agrees_with_the_cpu_law() {
  let build = || {
    Lmm::<f32, _>::new(
      array![0.0_f32, 0.5, 1.25, 1.5],
      array![0.03_f32, 0.035, 0.04],
      array![0.2_f32, 0.25, 0.3],
      N,
      Some(1.5),
      Deterministic::new(193),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  assert!(device.iter().all(|p| p.nrows() == 3 && p.ncols() == N));
  assert!(
    device.iter().all(|p| p.row(0).iter().all(|&v| v == 0.03)),
    "the first forward reset at the origin and must not move"
  );
  assert!(
    device
      .iter()
      .all(|p| p[(1, N - 1)] == p[(1, N - 2)] && p[(2, N - 1)] == p[(2, N - 2)]),
    "a forward past its reset date must be frozen"
  );
  assert!(
    device.iter().any(|p| p[(2, N / 2)] != p[(2, 0)]),
    "the third forward is live half way in"
  );
  for row in 1..3 {
    agrees(
      row_mean(&host, row),
      row_mean(&device, row),
      0.02,
      &format!("LMM forward {row} terminal mean"),
    );
    agrees(
      row_spread(&host, row),
      row_spread(&device, row),
      0.06,
      &format!("LMM forward {row} terminal spread"),
    );
  }
}

/// With a correlation matrix the shocks go through its Cholesky factor and the drifts through
/// its correlations; two live forwards' log-return correlation moves if either is dropped.
#[test]
fn lmm_correlated_forwards_agree_with_the_cpu_law() {
  let build = || {
    Lmm::<f32, _>::new(
      array![0.0_f32, 0.5, 1.25, 1.5],
      array![0.03_f32, 0.035, 0.04],
      array![0.2_f32, 0.25, 0.3],
      N,
      Some(1.5),
      Deterministic::new(197),
    )
    .with_correlation(array![
      [1.0_f32, 0.7, 0.4],
      [0.7, 1.0, 0.7],
      [0.4, 0.7, 1.0]
    ])
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  // Over the window where both forwards are live, up to the second's reset a third of the way
  // in, the correlation is the matrix's own; after it the third moves alone and dilutes it.
  let shared = |paths: &[Array2<f32>], row: usize| -> Vec<f64> {
    paths
      .iter()
      .map(|p| (p[(row, N / 3)] as f64 / p[(row, 0)] as f64).ln())
      .collect()
  };
  let (h, d) = (
    corr(&shared(&host, 1), &shared(&host, 2)),
    corr(&shared(&device, 1), &shared(&device, 2)),
  );
  assert!(d > 0.5, "the forwards' log-returns do not correlate: {d}");
  agrees(h, d, 0.05, "LMM forward log-return correlation");
  agrees(
    row_mean(&host, 2),
    row_mean(&device, 2),
    0.02,
    "LMM correlated forward terminal mean",
  );
}

/// The drift coupling made large enough to see (rates and volatility a half, unit accruals), so
/// a wrong coupling moves a live forward's mean log-return; without and with a correlation.
#[test]
fn lmm_drift_coupling_agrees_with_the_cpu_law() {
  for correlated in [false, true] {
    let build = || {
      let lmm = Lmm::<f32, _>::new(
        array![0.0_f32, 1.0, 2.0, 3.0],
        array![0.5_f32, 0.5, 0.5],
        array![0.5_f32, 0.5, 0.5],
        N,
        Some(3.0),
        Deterministic::new(211),
      );
      if correlated {
        lmm.with_correlation(array![
          [1.0_f32, 0.7, 0.4],
          [0.7, 1.0, 0.7],
          [0.4, 0.7, 1.0]
        ])
      } else {
        lmm
      }
    };
    const PATHS: usize = 4 * M;
    let device = build().on::<Device>().sample_par(PATHS);
    let host = build().sample_par(PATHS);
    for row in 1..3 {
      let mean = |paths: &[Array2<f32>]| {
        let r = log_returns(paths, row);
        r.iter().sum::<f64>() / r.len() as f64
      };
      let (h, d) = (mean(&host), mean(&device));
      assert!(
        (h - d).abs() < 0.02,
        "LMM forward {row} mean log-return (correlated: {correlated}): host {h}, device {d}"
      );
    }
  }
}

/// A tenor whose first reset lies ahead: every forward is live at first and freezes only once
/// its date has passed, which a device curve that froze the first from the start would break.
#[test]
fn lmm_forward_starting_tenor_agrees_with_the_cpu_law() {
  let build = || {
    Lmm::<f32, _>::new(
      array![0.5_f32, 1.0, 1.5, 2.0],
      array![0.03_f32, 0.035, 0.04],
      array![0.2_f32, 0.25, 0.3],
      N,
      Some(1.5),
      Deterministic::new(229),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert!(
    device.iter().any(|p| p[(0, N / 4)] != p[(0, 0)]),
    "the first forward is live before its reset"
  );
  assert!(
    device
      .iter()
      .all(|p| p[(0, N - 1)] == p[(0, N - 2)] && p[(1, N - 1)] == p[(1, N - 2)]),
    "a forward past its reset date must be frozen"
  );
  for row in 0..3 {
    agrees(
      row_spread(&host, row),
      row_spread(&device, row),
      0.06,
      &format!("LMM forward-starting forward {row} terminal spread"),
    );
  }
}

/// Five forwards exceed a launch's four slots, so the device build samples on
/// the host and is the host build to the bit.
#[test]
fn more_than_four_forwards_keep_lmm_on_the_host() {
  let build = || {
    Lmm::<f32, _>::new(
      array![0.0_f32, 0.5, 1.0, 1.5, 2.0, 2.5],
      array![0.03_f32, 0.035, 0.04, 0.045, 0.05],
      array![0.2_f32, 0.2, 0.2, 0.2, 0.2],
      64,
      Some(2.5),
      Deterministic::new(199),
    )
  };
  assert_eq!(build().on::<Device>().sample_par(8), build().sample_par(8));
}
