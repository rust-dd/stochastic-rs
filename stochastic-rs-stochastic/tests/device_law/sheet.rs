//! The sheet: a two-dimensional field rather than a path, produced by the
//! device's own circulant-embedding pipeline rather than the Euler engine.
//! What the device has to reproduce is the field's law point by point — the
//! spread at grid points and of the differences between them — against the
//! host sampler. The correction the sampler adds is a product of two normals,
//! so the spread is read as an interquartile range, which that product's
//! tails leave alone where a variance would wander.

use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::sheet::fbs::Fbs;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::Device;
use super::common::M;
use super::common::agrees;

/// Sheets per comparison: enough that an interquartile range is stable to
/// the tolerance below.
const SHEETS: usize = 6 * M;

/// The mean of one grid point across the sheets.
fn point_mean(sheets: &[Array2<f32>], i: usize, j: usize) -> f64 {
  sheets.iter().map(|s| s[(i, j)] as f64).sum::<f64>() / sheets.len() as f64
}

/// The interquartile range of `values`.
fn iqr(mut values: Vec<f64>) -> f64 {
  values.sort_by(|a, b| a.partial_cmp(b).unwrap());
  values[3 * values.len() / 4] - values[values.len() / 4]
}

/// The interquartile range of one grid point across the sheets.
fn point_spread(sheets: &[Array2<f32>], i: usize, j: usize) -> f64 {
  iqr(sheets.iter().map(|s| s[(i, j)] as f64).collect())
}

/// The interquartile range of the difference between two grid points.
fn increment_spread(sheets: &[Array2<f32>], p: (usize, usize), q: (usize, usize)) -> f64 {
  iqr(
    sheets
      .iter()
      .map(|s| (s[p] - s[q]) as f64)
      .collect(),
  )
}

/// The device pipeline carries the host field's law: every sheet has the
/// grid's shape and finite values, the grid points are centred where the
/// host's are, and their spreads and the spreads of their differences agree.
#[test]
fn the_sheet_pipeline_matches_the_cpu_field() {
  let build = || Fbs::<f32, _>::new(0.7, 17, 9, 1.0, Deterministic::new(19));
  let device = build().on::<Device>().sample_par(SHEETS);
  let host = build().sample_par(SHEETS);
  assert_eq!(device.len(), SHEETS);
  assert!(
    device
      .iter()
      .all(|s| s.dim() == (17, 9) && s.iter().all(|v| v.is_finite())),
    "a device sheet has the wrong shape or left the reals"
  );
  for (i, j) in [(16, 8), (8, 4), (0, 8), (16, 0)] {
    let (h, d) = (point_mean(&host, i, j), point_mean(&device, i, j));
    let scale = point_spread(&host, i, j) / (SHEETS as f64).sqrt();
    assert!(
      (h - d).abs() < 5.0 * scale,
      "Fbs ({i},{j}) mean: host {h}, device {d}"
    );
    agrees(
      point_spread(&host, i, j),
      point_spread(&device, i, j),
      0.06,
      &format!("Fbs ({i},{j}) spread"),
    );
  }
  for (p, q) in [((16, 8), (8, 4)), ((0, 8), (16, 8)), ((16, 0), (16, 8))] {
    agrees(
      increment_spread(&host, p, q),
      increment_spread(&device, p, q),
      0.06,
      &format!("Fbs increment {p:?} − {q:?} spread"),
    );
  }
}

/// A grid whose embedding sides are not powers of two has no radix-2
/// transform; the device build samples on the host and is the host build to
/// the bit.
#[test]
fn a_grid_off_the_powers_of_two_keeps_the_sheet_on_the_host() {
  let build = || Fbs::<f32, _>::new(0.7, 12, 9, 1.0, Deterministic::new(23));
  assert_eq!(build().on::<Device>().sample_par(4), build().sample_par(4));
  assert_eq!(build().on::<Device>().sample(), build().sample());
}
