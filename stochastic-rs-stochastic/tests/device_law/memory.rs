//! Device laws of the processes whose step reads the whole past — the rough
//! volatility models' memory integrals, the fractional stable motion's moving
//! average, the ARIMA recursions — which the engine's history block convolves
//! exactly as the host does, so the comparisons here are between one scheme
//! run on two machines.

use ndarray::Array1;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::autoregressive::arima::Arima;
use stochastic_rs_stochastic::autoregressive::sarima::Sarima;
use stochastic_rs_stochastic::process::lfsm::Lfsm;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::fbates_svj::FBatesSvj;
use stochastic_rs_stochastic::volatility::fheston::RoughHeston;
use stochastic_rs_stochastic::volatility::rbergomi::RoughBergomi;

use super::common::Device;
use super::common::M;
use super::common::agrees;
use super::common::all_finite;
use super::common::terminal_std;

const N: usize = 253;

fn mean_at(paths: &[[Array1<f32>; 2]], row: usize, at: usize) -> f64 {
  paths.iter().map(|p| p[row][at] as f64).sum::<f64>() / paths.len() as f64
}

fn std_at(paths: &[[Array1<f32>; 2]], row: usize, at: usize, f: impl Fn(f64) -> f64) -> f64 {
  let n = paths.len() as f64;
  let mean = paths.iter().map(|p| f(p[row][at] as f64)).sum::<f64>() / n;
  (paths
    .iter()
    .map(|p| (f(p[row][at] as f64) - mean).powi(2))
    .sum::<f64>()
    / n)
    .sqrt()
}

fn std_of(paths: &[Array1<f32>], at: usize) -> f64 {
  let n = paths.len() as f64;
  let mean = paths.iter().map(|p| p[at] as f64).sum::<f64>() / n;
  (paths
    .iter()
    .map(|p| (p[at] as f64 - mean).powi(2))
    .sum::<f64>()
    / n)
    .sqrt()
}

/// The interquartile range at one grid point: the spread statistic a heavy
/// tail leaves alone.
fn iqr_at(paths: &[Array1<f32>], at: usize) -> f64 {
  let mut values: Vec<f64> = paths.iter().map(|p| p[at] as f64).collect();
  values.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
  values[values.len() * 3 / 4] - values[values.len() / 4]
}

#[test]
fn rough_heston_agrees_with_the_cpu_law() {
  let build = || {
    RoughHeston::<f32, _>::new(
      0.3,
      Some(0.2),
      0.04,
      2.0,
      0.3,
      None,
      None,
      Some(1.0),
      N,
      Deterministic::new(61),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  assert_eq!(device[0][0].len(), N);
  assert_eq!(device[0][0][0], 1.0, "every spot path starts at s0");
  assert!(
    (device[0][1][0] - 0.04).abs() < 1e-6,
    "every variance path starts at v0²"
  );
  agrees(
    mean_at(&host, 0, N - 1),
    mean_at(&device, 0, N - 1),
    0.03,
    "rough Heston terminal spot",
  );
  agrees(
    mean_at(&host, 1, N - 1),
    mean_at(&device, 1, N - 1),
    0.05,
    "rough Heston terminal variance",
  );
  // The variance spread is where the memory term shows: it is the integral
  // of the local factor, and a kernel reading the wrong lags moves it.
  agrees(
    std_at(&host, 1, N - 1, |v| v),
    std_at(&device, 1, N - 1, |v| v),
    0.08,
    "rough Heston terminal variance spread",
  );
}

#[test]
fn fractional_bates_agrees_with_the_cpu_law() {
  let build = || {
    FBatesSvj::<f32, _>::new(
      0.3,
      0.02,
      100.0,
      0.04,
      0.04,
      2.0,
      0.3,
      -0.6,
      3.0,
      -0.02,
      0.05,
      N,
      Some(1.0),
      Deterministic::new(67),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device[0][0][0], 100.0, "every spot path starts at s0");
  agrees(
    mean_at(&host, 0, N - 1),
    mean_at(&device, 0, N - 1),
    0.03,
    "fractional Bates terminal spot",
  );
  agrees(
    mean_at(&host, 1, N - 1),
    mean_at(&device, 1, N - 1),
    0.05,
    "fractional Bates terminal variance",
  );
  agrees(
    std_at(&host, 0, N - 1, |s| s),
    std_at(&device, 0, N - 1, |s| s),
    0.06,
    "fractional Bates terminal spot spread",
  );
  // As for rough Heston, the variance spread is the statistic the memory
  // term moves; the spot's own moments barely see it.
  agrees(
    std_at(&host, 1, N - 1, |v| v),
    std_at(&device, 1, N - 1, |v| v),
    0.08,
    "fractional Bates terminal variance spread",
  );
}

#[test]
fn rough_bergomi_agrees_with_the_cpu_law() {
  let build = || {
    RoughBergomi::<f32, _>::new(
      0.1,
      1.5,
      Some(0.2),
      Some(100.0),
      0.02,
      -0.7,
      N,
      Some(1.0),
      Deterministic::new(71),
    )
  };
  const PATHS: usize = 4 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert!(
    device.iter().all(|p| p[1].iter().all(|&v| v > 0.0)),
    "the exponential variance cannot reach zero"
  );
  agrees(
    mean_at(&host, 0, N - 1),
    mean_at(&device, 0, N - 1),
    0.03,
    "rough Bergomi terminal spot",
  );
  // The log-variance is Gaussian with the Volterra driver's variance, so its
  // mean and spread pin both the convolution weights and the exponent curve.
  let (h, d) = (
    host.iter().map(|p| (p[1][N - 1] as f64).ln()).sum::<f64>() / PATHS as f64,
    device.iter().map(|p| (p[1][N - 1] as f64).ln()).sum::<f64>() / PATHS as f64,
  );
  assert!(
    (h - d).abs() < 0.07,
    "rough Bergomi mean terminal log-variance: host {h}, device {d}"
  );
  agrees(
    std_at(&host, 1, N - 1, f64::ln),
    std_at(&device, 1, N - 1, f64::ln),
    0.05,
    "rough Bergomi terminal log-variance spread",
  );
}

/// The stable innovations have no variance, so the spread is read off the
/// quartiles — at the horizon and a quarter of the way in, since the ratio of
/// the two is what the self-similarity index sets.
#[test]
fn lfsm_agrees_with_the_cpu_law() {
  let build = || {
    Lfsm::<f32, _>::new(
      1.7,
      0.2,
      0.7,
      0.1,
      N,
      Some(0.0),
      Some(1.0),
      Deterministic::new(73),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device[0][0], 0.0, "every path starts at x0");
  all_finite(&device, "LFSM");
  agrees(
    iqr_at(&host, N - 1),
    iqr_at(&device, N - 1),
    0.06,
    "LFSM terminal interquartile range",
  );
  agrees(
    iqr_at(&host, N / 4),
    iqr_at(&device, N / 4),
    0.06,
    "LFSM quarter-horizon interquartile range",
  );
}

#[test]
fn arima_agrees_with_the_cpu_law() {
  let build = || {
    Arima::<f32, _>::new(
      array![0.6_f32, -0.2],
      array![0.3_f32],
      1,
      0.5,
      N,
      Deterministic::new(79),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device[0].len(), N);
  agrees(
    std_of(&host, 0),
    std_of(&device, 0),
    0.05,
    "ARIMA first point spread (a draw, not a start value)",
  );
  agrees(
    std_of(&host, N / 4),
    std_of(&device, N / 4),
    0.05,
    "ARIMA quarter-length spread",
  );
  agrees(
    terminal_std(&host),
    terminal_std(&device),
    0.05,
    "ARIMA terminal spread",
  );
}

#[test]
fn sarima_agrees_with_the_cpu_law() {
  let build = || {
    Sarima::<f32, _>::new(
      array![0.5_f32],
      array![0.2_f32],
      array![0.3_f32],
      array![],
      0,
      1,
      12,
      0.5,
      N,
      Deterministic::new(83),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  agrees(
    std_of(&host, N / 4),
    std_of(&device, N / 4),
    0.05,
    "SARIMA quarter-length spread",
  );
  agrees(
    terminal_std(&host),
    terminal_std(&device),
    0.05,
    "SARIMA terminal spread",
  );
}

/// A series longer than the kernels' per-path history samples on the host,
/// and is then the host build to the bit.
#[test]
fn a_longer_series_keeps_the_process_on_the_host() {
  let build = || {
    Arima::<f32, _>::new(
      array![0.6_f32],
      array![0.3_f32],
      0,
      0.5,
      600,
      Deterministic::new(89),
    )
  };
  assert_eq!(build().on::<Device>().sample_par(8), build().sample_par(8));
}
