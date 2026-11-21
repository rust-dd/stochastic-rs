use ndarray::Array2;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::interest::hull_white_2f::HullWhite2F;
use crate::traits::ProcessExt;
use crate::traits::RealExt;

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

/// Sample correlations of one Euler step's increments, which are linear in the shocks;
/// `log S − ∫r` drops the rate shock the trapezoid adds to the spot.
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
