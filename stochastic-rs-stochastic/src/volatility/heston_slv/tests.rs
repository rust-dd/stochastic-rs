use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::traits::Grid2D;
use crate::volatility::heston_log::HestonLog;

const N: usize = 64;

fn slv<S: SeedExt>(eta: f64, leverage: impl Into<Fn2D<f64>>, seed: S) -> HestonSlv<f64, S> {
  HestonSlv::new(
    Some(100.0),
    Some(0.04),
    2.0,
    0.04,
    0.3,
    -0.7,
    0.05,
    eta,
    leverage,
    N,
    Some(1.0),
    seed,
  )
}

fn affine_leverage(_t: f64, s: f64) -> f64 {
  0.8 + 0.002 * s
}

/// Under full mixing and unit leverage the recursion is `HestonLog`'s, and
/// the two Gaussian streams are built the same way, so the two processes
/// agree to the bit at the same seed.
#[test]
fn unit_leverage_under_full_mixing_is_heston_log_path_for_path() {
  let [s, v] = slv(1.0, Expr::lit(1.0), Deterministic::new(42)).sample();
  let [s_log, v_log] = HestonLog::new(
    Some(0.05),
    None,
    None,
    None,
    2.0,
    0.04,
    0.3,
    -0.7,
    N,
    Some(100.0),
    Some(0.04),
    Some(1.0),
    Some(false),
    Deterministic::new(42),
  )
  .sample();
  assert_eq!(s, s_log);
  assert_eq!(v, v_log);
}

#[test]
fn seeded_is_deterministic_and_the_seed_matters() {
  let a = slv(0.7, Expr::lit(0.9), Deterministic::new(7));
  let b = a.clone();
  assert_eq!(a.sample(), b.sample());
  let c = slv(0.7, Expr::lit(0.9), Deterministic::new(8));
  assert_ne!(a.sample(), c.sample());
}

/// A grid that tabulates an affine leverage reproduces it by bilinear
/// interpolation, so the two coefficients drive the same path up to
/// rounding.
#[test]
fn a_grid_leverage_matches_the_closure_it_tabulates() {
  let spots = Array1::linspace(1.0, 400.0, 400);
  let times = Array1::from_vec(vec![0.0, 1.0]);
  let values = Array2::from_shape_fn((2, 400), |(j, i)| affine_leverage(times[j], spots[i]));
  let grid = Grid2D::new(times, spots, values);
  let [s_grid, v_grid] = slv(1.0, grid, Deterministic::new(3)).sample();
  let [s_fn, v_fn] = slv(
    1.0,
    affine_leverage as fn(f64, f64) -> f64,
    Deterministic::new(3),
  )
  .sample();
  assert_eq!(v_grid, v_fn, "the variance never reads the leverage");
  for (a, b) in s_grid.iter().zip(s_fn.iter()) {
    assert!((a - b).abs() < 1e-9 * a.abs(), "grid {a} vs closure {b}");
  }
}

/// With `eta = 0` the variance has no noise; started at its long-run level
/// it stays there, so the spot is a local-volatility model at `L sqrt(v0)`.
#[test]
fn eta_zero_freezes_the_variance_at_the_long_run_level() {
  let [s, v] = slv(0.0, Expr::lit(1.5), Deterministic::new(11)).sample();
  assert!(v.iter().all(|&x| x == 0.04));
  assert!(s.iter().all(|&x| x.is_finite() && x > 0.0));
}

#[test]
fn only_an_expression_leverage_is_device_ready() {
  assert!(slv(1.0, Expr::lit(1.0), Unseeded).device_ready());
  let closure = slv(1.0, affine_leverage as fn(f64, f64) -> f64, Unseeded);
  assert_eq!(
    closure.device_fallback(),
    Some("a leverage that is a closure, a grid or a Python callable rather than an Expr")
  );
  let grid = Grid2D::new(
    Array1::from_vec(vec![0.0]),
    Array1::from_vec(vec![50.0, 150.0]),
    Array2::from_elem((1, 2), 1.0),
  );
  assert!(!slv(1.0, grid, Unseeded).device_ready());
}

#[test]
fn the_default_is_the_heston_default_under_unit_leverage() {
  let p = HestonSlv::<f64>::default();
  assert_eq!(
    (p.kappa, p.theta, p.sigma, p.rho, p.mu, p.eta),
    (2.0, 0.04, 0.3, -0.7, 0.05, 1.0)
  );
  assert_eq!(
    (p.n, p.t, p.s0, p.v0),
    (252, Some(1.0), Some(100.0), Some(0.04))
  );
  assert!(p.device_ready(), "the default leverage is an expression");
  assert_eq!(p.leverage.call(0.3, 120.0), 1.0);
  let [s, v] = p.sample();
  assert_eq!((s.len(), v.len()), (252, 252));
  assert!(v.iter().all(|&x| x >= 0.0));
}

#[test]
#[should_panic(expected = "n must be at least 2")]
fn rejects_n_below_two() {
  let _ = HestonSlv::<f64, _>::new(
    Some(100.0),
    Some(0.04),
    2.0,
    0.04,
    0.3,
    -0.7,
    0.05,
    1.0,
    Expr::lit(1.0),
    1,
    Some(1.0),
    Unseeded,
  );
}

#[test]
#[should_panic(expected = "eta must be non-negative")]
fn rejects_a_negative_mixing_fraction() {
  let _ = slv(-0.5, Expr::lit(1.0), Unseeded);
}
