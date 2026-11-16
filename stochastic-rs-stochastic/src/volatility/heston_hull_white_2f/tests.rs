use ndarray::Array2;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::interest::hull_white_2f::HullWhite2F;
use crate::traits::ProcessExt;

/// Grzelak, Oosterlee and van Weeren (2011) experiment 4.2, with the drift target holding the mean rate at 3 percent.
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
    0.8,
    0.2,
    0.2,
    flat_theta as fn(f64) -> f64,
    1.1,
    0.01,
    0.015,
    0.8,
    rho_paper(),
    n,
    Some(1.0),
    Some(0.2),
    Some(0.03),
    Some(1.0),
    Deterministic::new(seed),
  )
}

fn check_shape_and_start(path: &Array2<f64>, n: usize) {
  assert_eq!(path.dim(), (ROWS, n));
  assert_eq!(path.column(0).to_vec(), vec![1.0, 0.2, 0.03, 0.0, 0.0]);
  assert!(path.row(0).iter().all(|s| *s > 0.0), "spot stays positive");
  assert!(
    path.row(1).iter().all(|v| *v >= 0.0),
    "variance stays non-negative"
  );
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

/// With both rate volatilities at zero the rate rows are the deterministic recursion `HullWhite2F` runs.
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
    assert!((path[(2, i)] - r[i]).abs() < 1e-12, "r mismatch at {i}");
    assert!((path[(3, i)] - u[i]).abs() < 1e-12, "u mismatch at {i}");
  }
}

#[test]
fn integrated_rate_row_is_the_trapezoidal_sum_of_the_rate_row() {
  let n = 65;
  let path = paper(n, 5).qe().sample();
  let dt = 1.0 / (n as f64 - 1.0);
  let mut acc = 0.0;
  for i in 1..n {
    acc += 0.5 * (path[(2, i - 1)] + path[(2, i)]) * dt;
    assert!(
      (path[(4, i)] - acc).abs() < 1e-14,
      "integrated rate mismatch at {i}"
    );
  }
}

/// `E[S_T exp(-∫r)] = S_0` through the standard error across paths; Euler holds it exactly per step,
/// QE up to the asset step's own small bias (Andersen 2008 §4.3), covered by the band at this vol-of-vol.
fn discounted_spot_holds<Sch: HybridScheme>(
  model: &HestonHullWhite2F<f64, Sch, Deterministic>,
  what: &str,
) {
  let n = model.n;
  let m = 20_000;
  let values = model.sample_map(m, |path| path[(0, n - 1)] * (-path[(4, n - 1)]).exp());
  let mean = values.iter().sum::<f64>() / m as f64;
  let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (m as f64 - 1.0);
  let se = (var / m as f64).sqrt();
  let z = (mean - 1.0) / se;
  assert!(
    z.abs() < 5.0,
    "{what}: E[S_T exp(-∫r)] = {mean} ± {se}, {z:.1} standard errors from S_0 = 1"
  );
}

#[test]
fn discounted_spot_is_a_martingale_under_both_schemes() {
  discounted_spot_holds(&paper(33, 20_240_601), "Euler");
  discounted_spot_holds(&paper(33, 20_240_601).qe(), "QE");
}

/// Grzelak's `Exe13_12.py` (Heston + one-factor Hull-White, `λ = 1.12, η = 0.01, S₀ = 100, T = 15,
/// r = 0.1, γ = 0.3, v̄ = 0.05, v₀ = 0.02, ρ_xr = 0.5, ρ_xv = −0.8, κ = 0.5`) prices calls by COS
/// from the affine H1-HW characteristic function; `(K, call)` below is that script's
/// `CallPutOptionPriceCOSMthd_StochIR` at `N = 2000, L = 15`. The hybrid with `σ₂ = 0` is that model, and the
/// affine approximation's gap to it is below the Monte Carlo error at this η (Grzelak and Oosterlee 2011).
#[test]
fn one_factor_limit_prices_grzelak_exe13_12_calls_within_monte_carlo_error() {
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
    0.5,
    0.05,
    0.3,
    theta_r as fn(f64) -> f64,
    1.12,
    0.01,
    0.0,
    1.0,
    rho,
    n,
    Some(100.0),
    Some(0.02),
    Some(0.1),
    Some(15.0),
    Deterministic::new(1312),
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
  let terminal = model.sample_map(m, |path| (path[(0, n - 1)], path[(4, n - 1)]));
  for (k, cos) in REFERENCE {
    let payoffs = terminal
      .iter()
      .map(|(s, int_r)| (s - k).max(0.0) * (-int_r).exp())
      .collect::<Vec<_>>();
    let mean = payoffs.iter().sum::<f64>() / m as f64;
    let var = payoffs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (m as f64 - 1.0);
    let se = (var / m as f64).sqrt();
    let z = (mean - cos) / se;
    assert!(
      z.abs() < 4.0,
      "K = {k}: Monte Carlo {mean} ± {se} against COS {cos}, {z:.1} standard errors apart"
    );
  }
}

#[test]
#[should_panic(expected = "n must satisfy `n >= 2`, got n = 1")]
fn rejects_n_below_two() {
  let _ = paper(1, 0);
}

#[test]
#[should_panic(
  expected = "sigma1 must satisfy `sigma1.is_finite() && sigma1 >= T::zero()`, got sigma1 = -0.01"
)]
fn rejects_a_negative_rate_volatility() {
  let _ = paper(8, 0).with_sigma1(-0.01);
}

#[test]
#[should_panic(expected = "not positive-definite at pivot 2")]
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
  expected = "the rate-variance correlation rho[(1, 2)] must satisfy `rho[(1, 2)] == T::zero()` under QE, got rho[(1, 2)] = 0.2"
)]
fn qe_rejects_a_rate_variance_correlation() {
  let mut rho = rho_paper();
  rho[(1, 2)] = 0.2;
  rho[(2, 1)] = 0.2;
  let _ = paper(8, 0).with_rho(rho).qe();
}
