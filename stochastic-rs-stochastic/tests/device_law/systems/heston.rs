use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::correlation::heston_stoch_corr::HestonStochCorr;
use stochastic_rs_stochastic::device::Cpu;
use stochastic_rs_stochastic::params::Drift;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::PathSampler;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::double_heston::DoubleHeston;
use stochastic_rs_stochastic::volatility::heston::Euler;
use stochastic_rs_stochastic::volatility::heston::Heston;
use stochastic_rs_stochastic::volatility::heston_hull_white_2f::HestonHullWhite2F;
use stochastic_rs_stochastic::volatility::heston_hull_white_2f::HestonHullWhite2FPaths;
use stochastic_rs_stochastic::volatility::heston_log::HestonLog;
use stochastic_rs_stochastic::volatility::heston2d::Heston2D;
use stochastic_rs_stochastic::volatility::multifactor_heston::MultifactorHeston;
use stochastic_rs_stochastic::volatility::multifactor_heston::MultifactorHestonPaths;

use super::*;

/// The spot is a martingale under zero drift and the variance reverts to theta, so both
/// terminal means can agree across streams; the truncating family keeps the variance non-negative.
#[test]
fn heston_agrees_with_the_cpu_law() {
  let build = || {
    Heston::<f32, Euler, _>::new(
      100.0,
      0.04,
      2.0,
      0.04,
      0.3,
      -0.7,
      0.0,
      253,
      Some(1.0),
      HestonPow::Sqrt,
      VarianceBoundary::Floor,
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
    Heston::<f32, Euler, _>::new(
      100.0,
      0.04,
      2.0,
      0.04,
      0.3,
      -0.7,
      0.0,
      253,
      Some(1.0),
      HestonPow::Sqrt,
      VarianceBoundary::Reflect,
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

/// The log-price form keeps the spot positive by construction, so that is
/// pinned alongside the two terminal means.
#[test]
fn log_heston_agrees_with_the_cpu_law() {
  let build = || {
    HestonLog::<f32, _>::new(
      Drift::Mu(0.0),
      2.0,
      0.04,
      0.3,
      -0.7,
      253,
      100.0,
      0.04,
      Some(1.0),
      VarianceBoundary::Floor,
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
      100.0,
      0.04,
      0.02,
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
      VarianceBoundary::Floor,
      Deterministic::new(31),
    )
  };
  let device = slots(build().on::<Device>().sample_par(M));
  assert!(
    device
      .iter()
      .all(|p| p[1].iter().chain(p[2].iter()).all(|&v| v >= 0.0)),
    "a truncated variance went negative"
  );
  let host = slots(build().sample_par(M));
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
  let device = slots(build().on::<Device>().sample_par(M));
  assert!(
    device
      .iter()
      .all(|p| p[2].iter().all(|&v| (-1.0..=1.0).contains(&v))),
    "the reported correlation left [-1, 1]"
  );
  let host = slots(build().sample_par(M));
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

/// Two Heston assets under one Cholesky factor, four components in one launch: both variances
/// stay non-negative and every component's terminal mean agrees.
#[test]
fn two_asset_heston_agrees_with_the_cpu_law() {
  let build = || {
    Heston2D::<f32, _>::new(
      [4.6, 4.6],
      [0.04, 0.03],
      [0.0, 0.0],
      [0.04, 0.03],
      [2.0, 1.5],
      [0.3, 0.25],
      [-0.6, 0.2, 0.1, -0.2, 0.3, 0.15],
      253,
      Some(1.0),
      VarianceBoundary::Floor,
      Deterministic::new(83),
    )
  };
  let device = slots(build().on::<Device>().sample_par(M));
  assert!(
    device
      .iter()
      .all(|p| p[1].iter().chain(p[3].iter()).all(|&v| v >= 0.0)),
    "a truncated variance went negative"
  );
  let host = slots(build().sample_par(M));
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

/// The quadratic-exponential scheme draws its variance from a moment-matched law with two
/// branches, picked by the kernel's own uniform; both branches are non-negative by construction.
#[test]
fn andersen_qe_heston_agrees_with_the_cpu_law() {
  let build = || {
    Heston::<f32, Euler, _>::new(
      100.0,
      0.04,
      2.0,
      0.04,
      0.3,
      -0.7,
      0.0,
      253,
      Some(1.0),
      HestonPow::Sqrt,
      VarianceBoundary::Floor,
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

/// The two-factor model rides the double-Heston family. Each factor's correlation with the spot
/// is compared too: a dropped or swapped `rho_k` keeps every marginal right and moves only that.
#[test]
fn multifactor_heston_agrees_with_the_cpu_law() {
  let build = || {
    MultifactorHeston::<f32, 2, _>::new(
      100.0,
      [0.04, 0.02],
      [2.0, 1.0],
      [0.04, 0.03],
      [0.3, 0.2],
      [-0.7, -0.3],
      0.03,
      253,
      Some(1.0),
      Deterministic::new(67),
    )
  };
  const PATHS: usize = 4 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  assert_eq!(device[0].s.len(), 253);
  assert_eq!(device[0].s[0], 100.0, "the spot starts at s0");
  assert_eq!(device[0].v[0][0], 0.04, "the first factor starts at its v0");
  assert_eq!(
    device[0].v[1][0], 0.02,
    "the second factor starts at its v0"
  );
  assert!(
    device.iter().all(|p| {
      p.s.iter().all(|x| x.is_finite())
        && p
          .v
          .iter()
          .all(|f| f.iter().all(|x| x.is_finite() && *x >= 0.0))
    }),
    "multifactor Heston: a device path left its domain"
  );
  let last = 252;
  let log_spot = |paths: &[MultifactorHestonPaths<f32, 2>]| -> Vec<f64> {
    paths
      .iter()
      .map(|p| (p.s[last] as f64 / 100.0).ln())
      .collect()
  };
  let factor = |paths: &[MultifactorHestonPaths<f32, 2>], k: usize| -> Vec<f64> {
    paths.iter().map(|p| p.v[k][last] as f64).collect()
  };
  let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
  let sd = |v: &[f64]| {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
  };
  let corr = |a: &[f64], b: &[f64]| {
    let (ma, mb) = (mean(a), mean(b));
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
      sab += (x - ma) * (y - mb);
      saa += (x - ma).powi(2);
      sbb += (y - mb).powi(2);
    }
    sab / (saa * sbb).sqrt()
  };
  let (hs, ds) = (log_spot(&host), log_spot(&device));
  assert!(
    (mean(&hs) - mean(&ds)).abs() < 0.01,
    "multifactor Heston log-spot mean: host {}, device {}",
    mean(&hs),
    mean(&ds)
  );
  agrees(sd(&hs), sd(&ds), 0.05, "multifactor Heston log-spot spread");
  for k in 0..2 {
    let (hv, dv) = (factor(&host, k), factor(&device, k));
    assert!(
      (mean(&hv) - mean(&dv)).abs() < 0.002,
      "multifactor Heston factor {k} terminal mean: host {}, device {}",
      mean(&hv),
      mean(&dv)
    );
    agrees(
      sd(&hv),
      sd(&dv),
      0.08,
      &format!("multifactor Heston factor {k} spread"),
    );
    let (hc, dc) = (corr(&hs, &hv), corr(&ds, &dv));
    assert!(
      (hc - dc).abs() < 0.05,
      "multifactor Heston spot/factor-{k} correlation: host {hc}, device {dc}"
    );
  }
}

/// One factor rides the same family with the second zeroed, and three factors stay on the host
/// on any backend: both still produce the model's shape and law.
#[test]
fn multifactor_heston_other_factor_counts_still_sample() {
  let one = MultifactorHeston::<f32, 1, _>::new(
    100.0,
    [0.04],
    [2.0],
    [0.04],
    [0.3],
    [-0.7],
    0.03,
    64,
    Some(1.0),
    Deterministic::new(71),
  )
  .on::<Device>()
  .sample_par(32);
  assert!(
    one
      .iter()
      .all(|p| p.s.len() == 64 && p.v[0].len() == 64 && p.s.iter().all(|x| x.is_finite()))
  );
  let three = MultifactorHeston::<f32, 3, _>::new(
    100.0,
    [0.04, 0.02, 0.01],
    [2.0, 1.0, 0.5],
    [0.04, 0.03, 0.02],
    [0.3, 0.2, 0.1],
    [-0.7, -0.3, 0.1],
    0.03,
    64,
    Some(1.0),
    Deterministic::new(73),
  )
  .on::<Device>()
  .sample_par(32);
  assert!(
    three
      .iter()
      .all(|p| p.s.len() == 64 && p.v.len() == 3 && p.s.iter().all(|x| x.is_finite()))
  );
}

/// A slope on the drift target and every pair correlated, so a curve slot or a factor entry
/// read from its neighbour moves a moment.
fn hybrid(seed: u64) -> HestonHullWhite2F<f32, Euler, Deterministic> {
  fn theta_r(t: f32) -> f32 {
    0.9 * (0.03 + 0.02 * t)
  }
  let rho = ndarray::array![
    [1.0_f32, -0.3, 0.35, 0.08],
    [-0.3, 1.0, 0.25, -0.15],
    [0.35, 0.25, 1.0, -0.4],
    [0.08, -0.15, -0.4, 1.0],
  ];
  HestonHullWhite2F::new(
    1.0,
    0.06,
    1.5,
    0.04,
    0.3,
    theta_r as fn(f32) -> f32,
    0.9,
    0.02,
    0.015,
    0.6,
    rho,
    Some(0.03),
    65,
    Some(1.0),
    Deterministic::new(seed),
  )
}

/// First, second and cross terminal moments of `(log S, v, r, u, ∫r)`, which between them read
/// every parameter and every factor entry.
fn hybrid_moments(paths: &[HestonHullWhite2FPaths<f32>]) -> Vec<(String, Vec<f64>)> {
  let last = paths[0].s.len() - 1;
  let names = ["log S", "v", "r", "u", "∫r"];
  let rows = paths
    .iter()
    .map(|p| [&p.s, &p.v, &p.r, &p.u, &p.integrated_rate].map(|row| row[last] as f64))
    .map(|[s, v, r, u, i]| [s.ln(), v, r, u, i])
    .collect::<Vec<_>>();
  let mut out = vec![];
  for a in 0..5 {
    out.push((names[a].to_string(), rows.iter().map(|x| x[a]).collect()));
    for b in a..5 {
      let name = format!("{} · {}", names[a], names[b]);
      out.push((name, rows.iter().map(|x| x[a] * x[b]).collect()));
    }
  }
  out
}

fn hybrid_agrees(
  what: &str,
  host: &[HestonHullWhite2FPaths<f32>],
  device: &[HestonHullWhite2FPaths<f32>],
) {
  for p in device {
    assert_eq!(
      (p.s[0], p.v[0], p.r[0]),
      (1.0, 0.06, 0.03),
      "{what}: the start"
    );
    assert_eq!(
      (p.u[0], p.integrated_rate[0]),
      (0.0, 0.0),
      "{what}: the start"
    );
    let mut acc = 0.0_f32;
    for i in 1..p.r.len() {
      acc += 0.5 * (p.r[i - 1] + p.r[i]) * (1.0 / 64.0);
      assert_eq!(
        p.integrated_rate[i], acc,
        "{what}: ∫r is the rate row's trapezoid"
      );
    }
    let rows = [&p.s, &p.v, &p.r, &p.u];
    assert!(rows.iter().all(|row| row.iter().all(|x| x.is_finite())));
    assert!(p.v.iter().all(|&v| v >= 0.0), "{what}: a negative variance");
  }
  for ((name, h), (_, d)) in hybrid_moments(host).iter().zip(&hybrid_moments(device)) {
    super::super::common::means_agree_values(h, d, &format!("{what}: terminal {name}"));
  }
}

/// On `Cpu` the engine's route is the host sampler bit for bit; on the device the law agrees.
#[test]
fn heston_hull_white_2f_agrees_with_the_cpu_law() {
  const PATHS: usize = 4 * M;
  assert_eq!(
    hybrid(23).on::<Cpu>().sample(),
    hybrid(23).sampler().sample()
  );
  let device = hybrid(23).on::<Device>();
  assert!(device.device_ready());
  hybrid_agrees(
    "Euler",
    &hybrid(23).sample_par(PATHS),
    &device.sample_par(PATHS),
  );
}

/// The quadratic-exponential family at a vol-of-vol whose `ψ` crosses 1.5, so both branches run.
#[test]
fn heston_hull_white_2f_qe_agrees_with_the_cpu_law() {
  const PATHS: usize = 4 * M;
  let qe = |seed| {
    let mut rho = hybrid(seed).rho().clone();
    for (i, j) in [(1, 2), (2, 1), (1, 3), (3, 1)] {
      rho[(i, j)] = 0.0;
    }
    hybrid(seed).with_rho(rho).with_sigma(0.6).qe()
  };
  assert_eq!(qe(29).on::<Cpu>().sample(), qe(29).sampler().sample());
  let host = qe(29).sample_par(PATHS);
  assert!(host.iter().any(|p| p.v.iter().any(|&v| v == 0.0)));
  hybrid_agrees("QE", &host, &qe(29).on::<Device>().sample_par(PATHS));
}
