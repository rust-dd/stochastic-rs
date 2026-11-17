use ndarray::Array3;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::traits::Fn1D;
use stochastic_rs_stochastic::diffusion::fouque::FouqueOu2D;
use stochastic_rs_stochastic::diffusion::wishart::Wishart;
use stochastic_rs_stochastic::interest::duffie_kan::DuffieKan;
use stochastic_rs_stochastic::interest::hull_white_2f::HullWhite2F;
use stochastic_rs_stochastic::process::cbms::Cbms;

use super::*;

/// Two Ornstein-Uhlenbeck factors on one clock, both reverting toward zero.
/// Their terminal means are near zero, where a relative comparison is the
/// ratio of two numbers that are both mostly sampling noise, so what is
/// compared is the spread each factor's own reversion speed sets.
#[test]
fn two_scale_ornstein_uhlenbeck_agrees_with_the_cpu_law() {
  let build = || {
    FouqueOu2D::<f32, _>::new(
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

/// The two-dimensional Wishart process by its exact step on both machines:
/// the terminal mean of every entry has a closed form both have to hit, the
/// spreads have to agree, and the device path has to stay in the cone.
#[test]
fn wishart_agrees_with_the_cpu_law_and_its_closed_form_mean() {
  let build = || {
    Wishart::<f32, _>::new(
      3.5,
      array![[-0.5_f32, 0.1], [0.05, -0.3]],
      array![[0.3_f32, 0.1], [0.0, 0.2]],
      array![[1.0_f32, 0.2], [0.2, 0.5]],
      253,
      Some(1.0),
      Deterministic::new(241),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  assert!(device.iter().all(|p| p.dim() == (253, 2, 2)));
  assert!(
    device.iter().all(|p| (0..253).all(|j| {
      p[(j, 0, 0)] >= 0.0
        && p[(j, 1, 1)] >= 0.0
        && p[(j, 0, 0)] * p[(j, 1, 1)] - p[(j, 0, 1)] * p[(j, 1, 0)] >= -1e-5
        && p[(j, 0, 1)] == p[(j, 1, 0)]
    })),
    "a device matrix left the cone or lost its symmetry"
  );
  let entry_mean = |paths: &[Array3<f32>], r: usize, c: usize| {
    paths.iter().map(|p| p[(252, r, c)] as f64).sum::<f64>() / paths.len() as f64
  };
  let entry_spread = |paths: &[Array3<f32>], r: usize, c: usize| {
    let n = paths.len() as f64;
    let mean = entry_mean(paths, r, c);
    (paths
      .iter()
      .map(|p| (p[(252, r, c)] as f64 - mean).powi(2))
      .sum::<f64>()
      / n)
      .sqrt()
  };
  let closed = build().mean(1.0);
  for (r, c) in [(0, 0), (0, 1), (1, 1)] {
    let (h, d, m) = (
      entry_mean(&host, r, c),
      entry_mean(&device, r, c),
      closed[(r, c)] as f64,
    );
    let se = entry_spread(&host, r, c) / (PATHS as f64).sqrt();
    assert!(
      (d - m).abs() < 4.0 * se,
      "Wishart entry ({r},{c}) terminal mean: device {d} vs closed form {m}, standard error {se}"
    );
    assert!(
      (h - d).abs() < 4.0 * se * std::f64::consts::SQRT_2,
      "Wishart entry ({r},{c}) terminal mean: host {h}, device {d}"
    );
    agrees(
      entry_spread(&host, r, c),
      entry_spread(&device, r, c),
      0.06,
      &format!("Wishart entry ({r},{c}) terminal spread"),
    );
  }
}

/// Below a degree of three the host's step can take its rank-adaptive branch
/// and below two its Poisson mixture, neither of which a family carries; the
/// device build samples on the host and is the host build to the bit. So
/// does a one-dimensional process, which the two-slot family does not fit.
#[test]
fn a_low_degree_or_one_dimension_keeps_wishart_on_the_host() {
  let low = || {
    Wishart::<f32, _>::new(
      2.5,
      array![[-0.5_f32, 0.1], [0.05, -0.3]],
      array![[0.3_f32, 0.1], [0.0, 0.2]],
      array![[1.0_f32, 0.2], [0.2, 0.5]],
      64,
      Some(1.0),
      Deterministic::new(251),
    )
  };
  assert_eq!(low().on::<Device>().sample_par(8), low().sample_par(8));
  let one = || {
    Wishart::<f32, _>::new(
      3.5,
      array![[-0.5_f32]],
      array![[0.3_f32]],
      array![[1.0_f32]],
      64,
      Some(1.0),
      Deterministic::new(257),
    )
  };
  assert_eq!(one().on::<Device>().sample_par(8), one().sample_par(8));
}
