use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::device::Cpu;
use stochastic_rs_stochastic::traits::PathSampler;
use stochastic_rs_stochastic::volatility::heston::Euler;
use stochastic_rs_stochastic::volatility::heston_hull_white_2f::HestonHullWhite2F;
use stochastic_rs_stochastic::volatility::heston_hull_white_2f::HestonHullWhite2FPaths;

use super::*;

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

/// With no variance and no rate volatility the path is the drift alone, `S = exp(∫r)`: a curve
/// slot read one grid point off, or a trapezoid term dropped, moves a row far past rounding.
#[test]
fn heston_hull_white_2f_drifts_as_the_host_does_without_noise() {
  fn steep(t: f32) -> f32 {
    0.03 + t
  }
  let quiet = || {
    let model = hybrid(31).with_v0(0.0).with_theta(0.0).with_sigma(0.0);
    let model = model.with_sigma1(0.0).with_sigma2(0.0);
    model.with_theta_r(steep as fn(f32) -> f32)
  };
  let (host, device) = (quiet().sample(), quiet().on::<Device>().sample());
  for (h, d) in [(&host.s, &device.s), (&host.r, &device.r)] {
    for (i, (&h, &d)) in h.iter().zip(d).enumerate() {
      assert!(
        (h - d).abs() <= 2e-5 * h.abs(),
        "point {i}: host {h}, device {d}"
      );
    }
  }
  assert!(
    host.r[64] > 0.3 && host.s[64] > 1.1,
    "the drift is not small"
  );
  assert!(device.u.iter().chain(&device.v).all(|&x| x == 0.0));
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
