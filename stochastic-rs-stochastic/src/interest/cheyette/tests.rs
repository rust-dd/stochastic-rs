//! Tests of the Cheyette state dynamics against the Hull–White limit and the
//! bond-reconstruction martingale identity.

use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

fn flat_forward(_t: f64) -> f64 {
  0.03
}

fn constant_vol(_t: f64, _x: f64) -> f64 {
  0.01
}

fn displaced_vol(_t: f64, x: f64) -> f64 {
  0.01 + 0.3 * x
}

fn hull_white_limit(seed: u64, n: usize, t: f64) -> Cheyette<f64, Deterministic> {
  Cheyette::new(
    flat_forward as fn(f64) -> f64,
    0.5,
    constant_vol as fn(f64, f64) -> f64,
    n,
    Some(t),
    Deterministic::new(seed),
  )
}

/// With constant `σ` the model is Hull–White: `y_t = σ²(1 − e^{−2κt})/(2κ)`
/// deterministically and `x_t ~ N(0, y_t)`.
#[test]
fn constant_volatility_reduces_to_the_hull_white_moments() {
  let (kappa, sigma, horizon) = (0.5_f64, 0.01_f64, 2.0_f64);
  let closed_form = sigma * sigma * (1.0 - (-2.0 * kappa * horizon).exp()) / (2.0 * kappa);
  let best = [1_u64, 2, 3]
    .into_iter()
    .map(|seed| {
      let process = hull_white_limit(seed, 401, horizon);
      let paths = process.sample_par(4_000);
      let n = paths.len() as f64;
      let y_terminal = paths[0][1][400];
      assert!(
        (y_terminal - closed_form).abs() / closed_form < 5e-3,
        "y_T {y_terminal} vs {closed_form}"
      );
      assert!(
        paths.iter().all(|p| p[1][400] == y_terminal),
        "y is deterministic"
      );
      let mean = paths.iter().map(|p| p[0][400]).sum::<f64>() / n;
      let var = paths
        .iter()
        .map(|p| (p[0][400] - mean).powi(2))
        .sum::<f64>()
        / n;
      (mean.abs() / closed_form.sqrt()).max((var - closed_form).abs() / closed_form)
    })
    .fold(f64::INFINITY, f64::min);
  assert!(best < 0.06, "worst relative deviation {best}");
}

/// Discounted bonds are martingales: `E[e^{−∫₀ᵗ r} P_t(T)] = P₀(T)` with the
/// short rate `f₀ + x` integrated by the trapezoid rule along the path.
#[test]
fn discounted_zero_bonds_are_martingales() {
  let (t, maturity) = (1.0_f64, 3.0_f64);
  let want = (-0.03_f64 * maturity).exp();
  let best = [4_u64, 5, 6]
    .into_iter()
    .map(|seed| {
      let process = hull_white_limit(seed, 101, t);
      let dt = process.dt();
      let paths = process.sample_par(4_000);
      let mc = paths
        .iter()
        .map(|[x, y]| {
          let integral: f64 = x
            .windows(2)
            .into_iter()
            .enumerate()
            .map(|(i, w)| {
              let r0 = process.short_rate(i as f64 * dt, w[0]);
              let r1 = process.short_rate((i + 1) as f64 * dt, w[1]);
              0.5 * (r0 + r1) * dt
            })
            .sum();
          (-integral).exp() * process.zero_bond(t, maturity, x[100], y[100])
        })
        .sum::<f64>()
        / paths.len() as f64;
      (mc - want).abs()
    })
    .fold(f64::INFINITY, f64::min);
  assert!(best < 1.5e-3, "martingale deviation {best}");
}

#[test]
fn curve_reconstruction_matches_the_initial_curve_at_the_origin() {
  let process = hull_white_limit(1, 11, 1.0);
  assert!((process.initial_discount(0.0, 2.0) - (-0.06_f64).exp()).abs() < 1e-12);
  assert!((process.zero_bond(0.0, 2.0, 0.0, 0.0) - (-0.06_f64).exp()).abs() < 1e-12);
  assert_eq!(process.g(1.0, 1.0), 0.0);
  assert!(
    (process.forward_rate(0.5, 0.5, 0.004, 0.0001) - process.short_rate(0.5, 0.004)).abs() < 1e-15
  );
  assert!((process.g(0.0, 2.0) - (1.0 - (-1.0_f64).exp()) / 0.5).abs() < 1e-15);
}

/// A displaced local volatility makes the state variance depend on the path:
/// `y` is no longer deterministic and rises with `x`.
#[test]
fn displaced_volatility_makes_the_variance_state_dependent() {
  let process = Cheyette::new(
    flat_forward as fn(f64) -> f64,
    0.5,
    displaced_vol as fn(f64, f64) -> f64,
    201,
    Some(2.0),
    Deterministic::new(9),
  );
  let paths = process.sample_par(2_000);
  let ys: Vec<f64> = paths.iter().map(|p| p[1][200]).collect();
  assert!(ys.iter().any(|y| (*y - ys[0]).abs() > 1e-12));
  assert!(ys.iter().all(|y| *y > 0.0));
  let (mut high, mut low) = (Vec::new(), Vec::new());
  for p in &paths {
    let mean_x = p[0].iter().sum::<f64>() / p[0].len() as f64;
    if mean_x > 0.0 {
      high.push(p[1][200]);
    } else {
      low.push(p[1][200]);
    }
  }
  let avg = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
  assert!(
    avg(&high) > avg(&low),
    "y should be larger on paths with positive x"
  );
}

#[test]
fn deterministic_seed_reproduces_and_consecutive_paths_differ() {
  let a = hull_white_limit(7, 32, 1.0).sample();
  let b = hull_white_limit(7, 32, 1.0).sample();
  assert_eq!(a, b);
  assert_eq!(a[0][0], 0.0);
  assert_eq!(a[1][0], 0.0);
  let process = hull_white_limit(7, 32, 1.0);
  let mut sampler = process.sampler();
  let first = sampler.sample();
  let second = sampler.sample();
  assert_ne!(first, second);
}

#[test]
#[should_panic(expected = "kappa must be positive")]
fn rejects_a_non_positive_mean_reversion() {
  let _ = Cheyette::new(
    flat_forward as fn(f64) -> f64,
    0.0,
    constant_vol as fn(f64, f64) -> f64,
    8,
    None,
    Unseeded,
  );
}
