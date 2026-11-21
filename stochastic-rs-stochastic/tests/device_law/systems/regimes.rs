use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::regime_switching::RegimeSwitchingDiffusion;

use super::*;

/// Three volatility regimes under a generator `Q`. Each regime's terminal occupancy and the
/// mean number of switches pin the chain: a successor drawn from the wrong row moves both.
#[test]
fn regime_switching_agrees_with_the_cpu_law() {
  let build = || {
    RegimeSwitchingDiffusion::<f32, _>::new(
      0.03,
      array![[-2.0, 1.5, 0.5], [1.0, -3.0, 2.0], [0.5, 0.5, -1.0]],
      array![0.1, 0.25, 0.4],
      0,
      253,
      100.0,
      Some(1.0),
      Deterministic::new(79),
    )
  };
  const PATHS: usize = 4 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  assert_eq!(device[0][0].len(), 253);
  assert_eq!(device[0][0][0], 100.0, "the spot starts at s0");
  assert_eq!(
    device[0][1][0], 0.0,
    "the chain starts in the initial regime"
  );
  assert!(
    device.iter().all(|[s, z]| {
      s.iter().all(|x| x.is_finite() && *x > 0.0)
        && z.iter().all(|r| *r == r.round() && (0.0..=2.0).contains(r))
    }),
    "regime switching: a device path left its domain"
  );
  let last = 252;
  let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
  let log_spot = |paths: &[[Array1<f32>; 2]]| -> Vec<f64> {
    paths
      .iter()
      .map(|[s, _]| (s[last] as f64 / 100.0).ln())
      .collect()
  };
  let (hs, ds) = (log_spot(&host), log_spot(&device));
  assert!(
    (mean(&hs) - mean(&ds)).abs() < 0.01,
    "regime switching log-spot mean: host {}, device {}",
    mean(&hs),
    mean(&ds)
  );
  let sd = |v: &[f64]| {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
  };
  agrees(sd(&hs), sd(&ds), 0.05, "regime switching log-spot spread");
  let occupancy = |paths: &[[Array1<f32>; 2]], regime: f32| {
    paths.iter().filter(|[_, z]| z[last] == regime).count() as f64 / paths.len() as f64
  };
  for r in 0..3 {
    let (h, d) = (occupancy(&host, r as f32), occupancy(&device, r as f32));
    assert!(
      (h - d).abs() < 0.02,
      "regime {r} terminal occupancy: host {h}, device {d}"
    );
  }
  let switches = |paths: &[[Array1<f32>; 2]]| {
    paths
      .iter()
      .map(|[_, z]| z.windows(2).into_iter().filter(|w| w[0] != w[1]).count() as f64)
      .sum::<f64>()
      / paths.len() as f64
  };
  agrees(
    switches(&host),
    switches(&device),
    0.05,
    "regime switching mean switch count",
  );
}

/// Five regimes stay on the host whatever the backend, and still produce the
/// model's own shape and law.
#[test]
fn regime_switching_wider_than_four_regimes_still_samples() {
  let q = array![
    [-1.0, 0.25, 0.25, 0.25, 0.25],
    [0.25, -1.0, 0.25, 0.25, 0.25],
    [0.25, 0.25, -1.0, 0.25, 0.25],
    [0.25, 0.25, 0.25, -1.0, 0.25],
    [0.25, 0.25, 0.25, 0.25, -1.0]
  ];
  let paths = RegimeSwitchingDiffusion::<f32, _>::new(
    0.02,
    q,
    array![0.1, 0.15, 0.2, 0.25, 0.3],
    2,
    64,
    50.0,
    Some(1.0),
    Deterministic::new(83),
  )
  .on::<Device>()
  .sample_par(16);
  assert_eq!(paths.len(), 16);
  assert!(
    paths
      .iter()
      .all(|[s, z]| s.len() == 64 && z.iter().all(|r| (0.0..=4.0).contains(r)))
  );
}
