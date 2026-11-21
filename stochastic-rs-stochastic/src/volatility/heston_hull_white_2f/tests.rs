use ndarray::Array2;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::euler::EulerSpec;
use crate::euler::families::Family;
use crate::euler::families::host_step;
use crate::interest::hull_white_2f::HullWhite2F;
use crate::traits::ProcessExt;
use crate::traits::RealExt;

mod truncation;

/// Holds the mean rate of Grzelak et al. (2011), experiment 4.2, at 3 percent.
fn flat_theta(_t: f64) -> f64 {
  1.1 * 0.03
}

fn rho_paper() -> Array2<f64> {
  array![
    [1.0, -0.3, 0.35, 0.08],
    [-0.3, 1.0, 0.0, 0.0],
    [0.35, 0.0, 1.0, -0.4],
    [0.08, 0.0, -0.4, 1.0],
  ]
}

fn paper(n: usize, seed: u64) -> HestonHullWhite2F<f64, Euler, Deterministic> {
  HestonHullWhite2F::new(
    1.0,
    0.2,
    0.8,
    0.2,
    0.2,
    flat_theta as fn(f64) -> f64,
    1.1,
    0.01,
    0.015,
    0.8,
    rho_paper(),
    Some(0.03),
    n,
    Some(1.0),
    Deterministic::new(seed),
  )
}

fn check_shape_and_start(path: &HestonHullWhite2FPaths<f64>, n: usize) {
  let rows = [&path.s, &path.v, &path.r, &path.u, &path.integrated_rate];
  assert!(rows.iter().all(|row| row.len() == n));
  assert_eq!(rows.map(|row| row[0]), [1.0, 0.2, 0.03, 0.0, 0.0]);
  assert!(path.s.iter().all(|s| *s > 0.0), "spot stays positive");
  assert!(path.v.iter().all(|v| *v >= 0.0), "variance is non-negative");
}

#[test]
fn sample_has_five_rows_from_the_initial_state_under_both_schemes() {
  check_shape_and_start(&paper(64, 1).sample(), 64);
  check_shape_and_start(&paper(64, 1).qe().sample(), 64);
}

#[test]
fn seeded_samples_reproduce_and_seeds_differ() {
  assert_eq!(paper(32, 7).sample(), paper(32, 7).sample());
  assert_eq!(paper(32, 7).qe().sample(), paper(32, 7).qe().sample());
  assert_ne!(paper(32, 7).sample(), paper(32, 8).sample());
  assert_ne!(paper(32, 7).qe().sample(), paper(32, 8).qe().sample());
  let model = paper(32, 7);
  let mut sampler = model.sampler();
  assert_ne!(
    sampler.sample(),
    sampler.sample(),
    "consecutive paths of one sampler differ"
  );
}

#[test]
fn sample_par_is_bit_identical_across_thread_pools() {
  let under = |threads: usize| {
    rayon::ThreadPoolBuilder::new()
      .num_threads(threads)
      .build()
      .unwrap()
      .install(|| paper(16, 11).qe().sample_par(70))
  };
  let one = under(1);
  assert_eq!(one, under(3));
  assert_eq!(one, under(8));
  assert_ne!(one[0], one[69], "paths of one call differ");
}

/// With both rate volatilities at zero the rate rows are `HullWhite2F`'s deterministic recursion.
#[test]
fn zero_rate_volatility_reproduces_hull_white_2f() {
  let n = 33;
  let rates = HullWhite2F::<f64>::new(
    flat_theta as fn(f64) -> f64,
    1.1,
    0.0,
    0.0,
    -0.4,
    0.8,
    Some(0.03),
    Some(1.0),
    n,
    Unseeded,
  );
  let [r, u] = rates.sample();
  let path = paper(n, 3).with_sigma1(0.0).with_sigma2(0.0).sample();
  for i in 0..n {
    assert!((path.r[i] - r[i]).abs() < 1e-12, "r mismatch at {i}");
    assert!((path.u[i] - u[i]).abs() < 1e-12, "u mismatch at {i}");
  }
}

#[test]
fn theta_r_is_tabulated_once_per_sampler() {
  use std::sync::atomic::AtomicUsize;
  use std::sync::atomic::Ordering;
  static CALLS: AtomicUsize = AtomicUsize::new(0);
  fn counted(_t: f64) -> f64 {
    CALLS.fetch_add(1, Ordering::Relaxed);
    1.1 * 0.03
  }
  let n = 16;
  let model = paper(n, 1).with_theta_r(counted as fn(f64) -> f64);
  let mut sampler = model.sampler();
  for _ in 0..5 {
    sampler.sample();
  }
  assert_eq!(CALLS.load(Ordering::Relaxed), n);
}

#[test]
fn a_non_finite_theta_r_is_nan_from_its_grid_point_on() {
  fn broken_late(t: f64) -> f64 {
    if t < 0.5 { 1.1 * 0.03 } else { f64::NAN }
  }
  let n = 33;
  let path = paper(n, 1)
    .with_theta_r(broken_late as fn(f64) -> f64)
    .sample();
  for i in 0..n {
    let nan = i >= 16;
    assert_eq!(path.r[i].is_nan(), nan, "r at {i}");
    assert_eq!(path.s[i].is_nan(), nan, "s at {i}");
    assert_eq!(path.integrated_rate[i].is_nan(), nan, "∫r at {i}");
    assert!(path.v[i].is_finite() && path.u[i].is_finite());
  }
}

#[test]
fn integrated_rate_row_is_the_trapezoidal_sum_of_the_rate_row() {
  let n = 65;
  let path = paper(n, 5).qe().sample();
  let dt = 1.0 / (n as f64 - 1.0);
  let mut acc = 0.0;
  for i in 1..n {
    acc += 0.5 * (path.r[i - 1] + path.r[i]) * dt;
    assert_eq!(path.integrated_rate[i], acc, "integrated rate at {i}");
  }
}

/// Standard errors a statistic may sit from its reference, on the best of [`SEEDS`].
const Z_MAX: f64 = 3.0;
const SEEDS: [u64; 3] = [20_240_601, 1312, 42];

/// `|z|` of `E[S_T exp(-∫r)]` against `S_0 = 1`: exact per step under Euler, and under QE up
/// to the asset step's own bias (Andersen 2008, §4.3).
fn discounted_spot_z<Sch: HybridScheme>(model: &HestonHullWhite2F<f64, Sch, Deterministic>) -> f64
where
  HestonHullWhite2F<f64, Sch, Deterministic>: ProcessExt<f64, Output = HestonHullWhite2FPaths<f64>>,
{
  let n = model.n;
  let m = 20_000;
  let values = model.sample_map(m, |path| {
    path.s[n - 1] * (-path.integrated_rate[n - 1]).exp()
  });
  let mean = values.iter().sum::<f64>() / m as f64;
  let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (m as f64 - 1.0);
  ((mean - 1.0) / (var / m as f64).sqrt()).abs()
}

#[test]
fn discounted_spot_is_a_martingale_under_both_schemes() {
  let euler = f64::min_or_nan_fold(SEEDS.map(|seed| discounted_spot_z(&paper(33, seed))));
  assert!(
    euler < Z_MAX,
    "Euler: every seed is {euler:.2} SE or more off"
  );
  let qe = f64::min_or_nan_fold(SEEDS.map(|seed| discounted_spot_z(&paper(33, seed).qe())));
  assert!(qe < Z_MAX, "QE: every seed is {qe:.2} SE or more off");
}

/// Every pair correlated, the rate-variance entries included, each entry a value of its own.
fn rho_full() -> Array2<f64> {
  array![
    [1.0, -0.3, 0.35, 0.08],
    [-0.3, 1.0, 0.25, -0.15],
    [0.35, 0.25, 1.0, -0.4],
    [0.08, -0.15, -0.4, 1.0],
  ]
}

/// Sample correlations of one Euler step's increments, linear in the shocks but where the
/// reported variance truncates at zero (1.3 % of steps); `log S − ∫r` drops the rate shock.
fn one_step_correlations(seed: u64) -> Array2<f64> {
  let m = 100_000;
  let model = paper(2, seed).with_rho(rho_full());
  let steps = model.sample_map(m, |path| {
    [
      path.s[1].ln() - path.integrated_rate[1],
      path.v[1],
      path.r[1],
      path.u[1],
    ]
  });
  let mean: [f64; 4] = std::array::from_fn(|i| steps.iter().map(|x| x[i]).sum::<f64>() / m as f64);
  let cov = |i: usize, j: usize| {
    steps
      .iter()
      .map(|x| (x[i] - mean[i]) * (x[j] - mean[j]))
      .sum::<f64>()
  };
  Array2::from_shape_fn((4, 4), |(i, j)| cov(i, j) / (cov(i, i) * cov(j, j)).sqrt())
}

/// The worst `|z|` of the six sample correlations against `reference`, at the standard error
/// `(1 − ρ²)/√m` of a Gaussian pair.
fn correlation_worst_z(sample: &Array2<f64>, reference: &Array2<f64>) -> f64 {
  let pairs = (0..4).flat_map(|i| (0..i).map(move |j| (i, j)));
  f64::max_or_nan_fold(pairs.map(|(i, j)| {
    let rho = reference[(i, j)];
    ((sample[(i, j)] - rho) / ((1.0 - rho * rho) / 100_000_f64.sqrt())).abs()
  }))
}

#[test]
fn one_euler_step_has_the_full_correlation_matrix() {
  let samples = SEEDS.map(one_step_correlations);
  let best = f64::min_or_nan_fold(samples.iter().map(|c| correlation_worst_z(c, &rho_full())));
  assert!(
    best < Z_MAX,
    "every seed has an entry {best:.2} SE or more off"
  );
  let mut swapped = rho_full();
  for (i, j) in [(1, 2), (2, 1)] {
    swapped[(i, j)] = rho_full()[(1, 3)];
  }
  for (i, j) in [(1, 3), (3, 1)] {
    swapped[(i, j)] = rho_full()[(1, 2)];
  }
  for sample in &samples {
    let z = correlation_worst_z(sample, &swapped);
    assert!(
      z > 5.0,
      "the rate-variance entries swapped are only {z:.2} SE away"
    );
  }
}

/// The worst `|z|` over `(K, call)` of Grzelak's `Exe13_12.py`, COS on the affine H1-HW model,
/// which the hybrid with `σ₂ = 0` is up to an approximation gap below the Monte Carlo error.
fn exe13_12_worst_z(seed: u64) -> f64 {
  fn theta_r(t: f64) -> f64 {
    let (lambda, eta, r) = (1.12_f64, 0.01_f64, 0.1_f64);
    lambda * r + eta * eta / (2.0 * lambda) * (1.0 - (-2.0 * lambda * t).exp())
  }
  let rho = array![
    [1.0, -0.8, 0.5, 0.0],
    [-0.8, 1.0, 0.0, 0.0],
    [0.5, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
  ];
  let n = 301;
  let model = HestonHullWhite2F::new(
    100.0,
    0.02,
    0.5,
    0.05,
    0.3,
    theta_r as fn(f64) -> f64,
    1.12,
    0.01,
    0.0,
    1.0,
    rho,
    Some(0.1),
    n,
    Some(15.0),
    Deterministic::new(seed),
  )
  .qe();
  const REFERENCE: [(f64, f64); 5] = [
    (200.0, 60.796698),
    (300.0, 45.629827),
    (448.168907, 28.455323),
    (600.0, 16.583892),
    (800.0, 7.554386),
  ];
  let m = 20_000;
  let terminal = model.sample_map(m, |path| (path.s[n - 1], path.integrated_rate[n - 1]));
  f64::max_or_nan_fold(REFERENCE.map(|(k, cos)| {
    let payoffs = terminal
      .iter()
      .map(|(s, int_r)| (s - k).max(0.0) * (-int_r).exp())
      .collect::<Vec<_>>();
    let mean = payoffs.iter().sum::<f64>() / m as f64;
    let var = payoffs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (m as f64 - 1.0);
    ((mean - cos) / (var / m as f64).sqrt()).abs()
  }))
}

#[test]
fn one_factor_limit_prices_grzelak_exe13_12_calls_within_monte_carlo_error() {
  let best = f64::min_or_nan_fold(SEEDS.map(exe13_12_worst_z));
  assert!(
    best < Z_MAX,
    "every seed has a call {best:.2} SE or more from COS"
  );
}

#[test]
#[should_panic(expected = "n must satisfy `n >= 2`, got n = 1")]
fn rejects_n_below_two() {
  let _ = paper(1, 0);
}

#[test]
#[should_panic(expected = "sigma1 must satisfy `sigma1 >= T::zero()`, got sigma1 = -0.01")]
fn rejects_a_negative_rate_volatility() {
  let _ = paper(8, 0).with_sigma1(-0.01);
}

#[test]
#[should_panic(
  expected = "rho must satisfy `every Cholesky pivot of rho > T::zero()`, got pivot 2 ="
)]
fn rejects_an_indefinite_correlation_matrix() {
  let rho = array![
    [1.0, 0.9, 0.9, 0.0],
    [0.9, 1.0, -0.9, 0.0],
    [0.9, -0.9, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
  ];
  let _ = paper(8, 0).with_rho(rho);
}

#[test]
#[should_panic(
  expected = "rho[(0, 0)] must satisfy `(rho[(i, i)] - T::one()).abs() <= tol`, got rho[(0, 0)] = 2.0"
)]
fn rejects_a_covariance_matrix() {
  let _ = paper(8, 0).with_rho(Array2::<f64>::eye(4) * 2.0);
}

#[test]
#[should_panic(
  expected = "rho[(1, 2)] must satisfy `rho[(1, 2)] == T::zero()`, got rho[(1, 2)] = 0.2"
)]
fn qe_rejects_a_rate_variance_correlation() {
  let mut rho = rho_paper();
  rho[(1, 2)] = 0.2;
  rho[(2, 1)] = 0.2;
  let _ = paper(8, 0).with_rho(rho).qe();
}

#[test]
#[should_panic(
  expected = "rho[(1, 3)] must satisfy `rho[(1, 3)] == T::zero()`, got rho[(1, 3)] = 0.1"
)]
fn with_rho_after_qe_rejects_a_rate_variance_correlation() {
  let mut rho = rho_paper();
  rho[(1, 3)] = 0.1;
  rho[(3, 1)] = 0.1;
  let _ = paper(8, 0).qe().with_rho(rho);
}

#[test]
#[should_panic(expected = "kappa must satisfy `kappa > T::zero()`, got kappa = 0.0")]
fn with_kappa_after_qe_rejects_zero() {
  let _ = paper(8, 0).qe().with_kappa(0.0);
}

#[test]
#[should_panic(expected = "theta must satisfy `theta > T::zero()`, got theta = 0.0")]
fn with_theta_after_qe_rejects_zero() {
  let _ = paper(8, 0).qe().with_theta(0.0);
}

#[test]
#[should_panic(expected = "sigma must satisfy `sigma > T::zero()`, got sigma = 0.0")]
fn with_sigma_after_qe_rejects_zero() {
  let _ = paper(8, 0).qe().with_sigma(0.0);
}

#[test]
fn with_rho_refactors_in_the_scheme_order() {
  let mut rho = rho_paper();
  rho[(0, 2)] = -0.2;
  rho[(2, 0)] = -0.2;
  let fresh = || {
    HestonHullWhite2F::new(
      1.0,
      0.2,
      0.8,
      0.2,
      0.2,
      flat_theta as fn(f64) -> f64,
      1.1,
      0.01,
      0.015,
      0.8,
      rho.clone(),
      Some(0.03),
      16,
      Some(1.0),
      Deterministic::new(9),
    )
  };
  assert_eq!(paper(16, 9).with_rho(rho.clone()).rho(), &rho);
  assert_eq!(
    paper(16, 9).with_rho(rho.clone()).sample(),
    fresh().sample()
  );
  assert_eq!(
    paper(16, 9).qe().with_rho(rho.clone()).sample(),
    fresh().qe().sample()
  );
}

fn family_step(
  family: Family,
  state: [f64; 4],
  spec: EulerSpec<f64>,
  ct: f64,
  u: f64,
  dz: [f64; 4],
) -> [f64; 4] {
  let (code, params) = spec.encode();
  assert_eq!(code, family.code());
  let mut out = [0.0; 4];
  let dt = 1.0 / 32.0;
  host_step(
    family, &state, &params, dt, ct, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, u, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &dz, &mut out,
  );
  out
}

/// The Euler family is the host step's operations in the host's order, so the device kernel
/// and the host sampler compute one recursion.
#[test]
fn the_euler_device_family_steps_as_the_host_does_bit_for_bit() {
  let model = paper(33, 1).with_rho(rho_full());
  let dt = model.dt();
  for (state, dz) in [
    ([0.1, 0.2, 0.03, -0.004], [0.11, -0.2, 0.05, 0.3]),
    ([-0.3, 0.0, -0.01, 0.02], [-0.25, 0.4, -0.1, 0.02]),
    ([0.0, 1e-4, 0.05, 0.0], [0.3, -0.9, 0.2, -0.1]),
  ] {
    let (host, _) = model.euler_step(0.033, dt, state, dz);
    let spec = Euler::euler_spec(&model);
    let device = family_step(Family::HestonHullWhite2F, state, spec, 0.033, 0.0, dz);
    assert_eq!(device, host);
  }
}

/// One QE step written out from Andersen's eq. 17-33, with the rate leg and its trapezoid.
fn qe_reference(eps: f64, state: [f64; 4], ct: f64, u: f64, z: [f64; 4]) -> [f64; 4] {
  let (kappa, theta, rho_xv) = (0.8_f64, 0.2_f64, -0.3_f64);
  let (a, b, sigma1, sigma2) = (1.1, 0.8, 0.01, 0.015);
  let (rho_xr, rho_xu, rho_ru) = (0.35_f64, 0.08_f64, -0.4_f64);
  let dt = 1.0 / 32.0_f64;
  let [log_s, v, r, w] = state;
  let [zv, z_r, z_3, z_4] = z;
  let e = (-kappa * dt).exp();
  let m = theta + (v - theta) * e;
  let s2 =
    v * eps * eps * e / kappa * (1.0 - e) + theta * eps * eps / (2.0 * kappa) * (1.0 - e).powi(2);
  let psi = s2 / (m * m);
  let v_next = if psi <= 1.5 {
    let b2 = 2.0 / psi - 1.0 + (2.0 / psi * (2.0 / psi - 1.0)).sqrt();
    m / (1.0 + b2) * (b2.sqrt() + zv).powi(2)
  } else {
    let p = (psi - 1.0) / (psi + 1.0);
    if u <= p {
      0.0
    } else {
      ((1.0 - p) / (1.0 - u)).ln() * m / (1.0 - p)
    }
  };
  let l22 = (1.0 - rho_ru * rho_ru).sqrt();
  let l32 = (rho_xu - rho_xr * rho_ru) / l22;
  let l33 = (1.0 - rho_xv * rho_xv - rho_xr * rho_xr - l32 * l32).sqrt();
  let r_next = r + (ct + w - a * r) * dt + sigma1 * dt.sqrt() * z_r;
  let w_next = w - b * w * dt + sigma2 * dt.sqrt() * (rho_ru * z_r + l22 * z_3);
  let k0 = -rho_xv * kappa * theta * dt / eps;
  let k1 = 0.5 * dt * (kappa * rho_xv / eps - 0.5) - rho_xv / eps;
  let k2 = 0.5 * dt * (kappa * rho_xv / eps - 0.5) + rho_xv / eps;
  let z_x = rho_xr * z_r + l32 * z_3 + l33 * z_4;
  let log_s_next = log_s
    + 0.5 * (r + r_next) * dt
    + k0
    + k1 * v
    + k2 * v_next
    + (0.5 * dt * (v + v_next)).sqrt() * z_x;
  [log_s_next, v_next, r_next, w_next]
}

/// At `σ = 1` and a variance near zero `ψ = σ²/(2κθ) > 1.5`, the exponential branch: a uniform
/// under `p` is the atom at zero, one above it the tail.
#[test]
fn the_qe_device_family_is_andersens_step_on_both_branches() {
  let low = [0.0, 1e-5, 0.02, 0.01];
  let cases = [
    (0.2, [0.1, 0.2, 0.03, -0.004], 0.4, [0.3, -0.2, 0.5, 0.1]),
    (1.0, low, 0.999, [-0.4, 0.7, -0.3, 0.2]),
    (1.0, low, 0.01, [1.2, -0.1, 0.4, -0.6]),
  ];
  let variances = cases.map(|(eps, state, u, z)| {
    let model = paper(33, 1).with_sigma(eps).qe();
    let dz = z.map(|z| model.dt().sqrt() * z);
    let spec = AndersenQe::euler_spec(&model);
    let device = family_step(Family::HestonHullWhite2FQe, state, spec, 0.033, u, dz);
    let reference = qe_reference(eps, state, 0.033, u, z);
    for (got, want) in device.into_iter().zip(reference) {
      assert!(
        (got - want).abs() <= 1e-12 * want.abs().max(1.0),
        "{got} vs {want}"
      );
    }
    device[1]
  });
  assert!(variances[1] > 1e-3, "the tail of the exponential branch");
  assert_eq!(variances[2], 0.0, "the atom of the exponential branch");
}

/// `sample` and `sample_par` rebuild `∫r` from the launch's rate row; the samplers accumulate
/// their own inside the step.
#[test]
fn the_rebuilt_integrated_rate_is_the_samplers_own_row_bit_for_bit() {
  use crate::traits::process::sample_par_chunked;
  for seed in SEEDS {
    let own = paper(129, seed).sampler().sample();
    let rebuilt = super::device::integrated_rate(&own.r, paper(129, seed).dt());
    assert_eq!(rebuilt, own.integrated_rate);
    assert_eq!(paper(129, seed).sample(), own, "Euler");
    let own = paper(129, seed).qe().sampler().sample();
    assert_eq!(paper(129, seed).qe().sample(), own, "QE");
    let own = sample_par_chunked(&paper(17, seed).qe(), 70);
    assert_eq!(paper(17, seed).qe().sample_par(70), own, "chunked");
    assert_eq!(paper(17, seed).qe().try_sample_par(70).unwrap(), own);
  }
}
