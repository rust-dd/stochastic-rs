//! Device laws of the tempered-stable processes built from their shot-noise series: the host
//! sorts every term onto the grid and the series block sums the same terms into their cells.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::jump::cgmy::Cgmy;
use stochastic_rs_stochastic::jump::cts::Cts;
use stochastic_rs_stochastic::jump::kobol::Kobol;
use stochastic_rs_stochastic::jump::rdts::Rdts;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::Device;
use super::common::M;
use super::common::agrees;
use super::common::all_finite;
use super::common::terminal_mean;
use super::common::terminal_std;

const N: usize = 253;

/// Terms per path: enough that the truncation error of the series is well
/// below the statistics compared, on both machines alike.
const J: usize = 256;

fn std_at(paths: &[Array1<f32>], at: usize) -> f64 {
  let n = paths.len() as f64;
  let mean = paths.iter().map(|p| p[at] as f64).sum::<f64>() / n;
  (paths
    .iter()
    .map(|p| (p[at] as f64 - mean).powi(2))
    .sum::<f64>()
    / n)
    .sqrt()
}

/// The terminal mean against the terminal spread, not against itself: the drift and the
/// asymmetric jumps set it near zero, where a ratio would see only noise.
fn mean_agrees(host: &[Array1<f32>], device: &[Array1<f32>], what: &str) {
  let (h, d) = (terminal_mean(host), terminal_mean(device));
  let scale = terminal_std(host);
  assert!(
    (h - d).abs() < 0.05 * scale,
    "{what}: host {h}, device {d} against a spread of {scale}"
  );
}

fn law_agrees(host: &[Array1<f32>], device: &[Array1<f32>], x0: f32, what: &str) {
  assert_eq!(device.len(), host.len());
  assert_eq!(device[0].len(), N);
  assert_eq!(device[0][0], x0, "{what}: every path starts at x0");
  all_finite(device, what);
  mean_agrees(host, device, what);
  agrees(
    terminal_std(host),
    terminal_std(device),
    0.06,
    &format!("{what} terminal spread"),
  );
  // A quarter of the way in as well: the series' terms land by arrival time, and a kernel
  // filing them into the wrong cells would keep the terminal law while moving this one.
  agrees(
    std_at(host, N / 4),
    std_at(device, N / 4),
    0.06,
    &format!("{what} quarter-horizon spread"),
  );
}

#[test]
fn cgmy_agrees_with_the_cpu_law() {
  let build = || {
    Cgmy::<f32, _>::new(
      1.0,
      2.0,
      6.0,
      0.5,
      N,
      J,
      Some(0.0),
      Some(1.0),
      Deterministic::new(131),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  law_agrees(&host, &device, 0.0, "CGMY");
}

#[test]
fn classical_tempered_stable_agrees_with_the_cpu_law() {
  let build = || {
    Cts::<f32, _>::new(
      2.0,
      6.0,
      0.5,
      N,
      J,
      Some(0.5),
      Some(1.0),
      Deterministic::new(137),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  law_agrees(&host, &device, 0.5, "CTS");
}

#[test]
fn kobol_agrees_with_the_cpu_law() {
  let build = || {
    Kobol::<f32, _>::new(
      1.0,
      2.0,
      1.0,
      3.0,
      3.0,
      0.5,
      N,
      J,
      Some(0.0),
      Some(1.0),
      Deterministic::new(139),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  law_agrees(&host, &device, 0.0, "Kobol");
}

#[test]
fn rapidly_decreasing_tempered_stable_agrees_with_the_cpu_law() {
  let build = || {
    Rdts::<f32, _>::new(
      2.0,
      6.0,
      0.5,
      N,
      J,
      Some(0.0),
      Some(1.0),
      Deterministic::new(149),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  law_agrees(&host, &device, 0.0, "RDTS");
}

/// Above `α = 1` the terms past the last carry a tenth of the variance and, with unequal
/// tempering, of the mean: the launch's remainder constants against the host's per-path ones.
#[test]
fn classical_tempered_stable_agrees_with_the_cpu_law_above_one() {
  let build = || {
    Cts::<f32, _>::new(
      2.0,
      6.0,
      1.5,
      N,
      J,
      Some(0.0),
      Some(1.0),
      Deterministic::new(157),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  law_agrees(&host, &device, 0.0, "CTS at α = 1.5");
  agrees(
    terminal_std(&device),
    1.0,
    0.06,
    "CTS at α = 1.5: unit variance",
  );
}

/// Unequal sides above `α = 1`: the remainder's drift is what centres the series.
#[test]
fn kobol_with_unequal_sides_agrees_with_the_cpu_law_above_one() {
  let build = || {
    Kobol::<f32, _>::new(
      0.4,
      2.0,
      1.0,
      3.0,
      5.0,
      1.5,
      N,
      J,
      Some(0.0),
      Some(1.0),
      Deterministic::new(163),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  law_agrees(&host, &device, 0.0, "Kobol at α = 1.5");
}

/// A grid longer than the kernels' series cells samples on the host, and is
/// then the host build to the bit.
#[test]
fn a_longer_grid_keeps_the_series_process_on_the_host() {
  let build = || {
    Cgmy::<f32, _>::new(
      1.0,
      2.0,
      6.0,
      0.5,
      600,
      J,
      Some(0.0),
      Some(1.0),
      Deterministic::new(151),
    )
  };
  assert_eq!(build().on::<Device>().sample_par(8), build().sample_par(8));
}
