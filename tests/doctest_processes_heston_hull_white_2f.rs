// docs: processes#heston-hull-white-2f-hybrid
//! Backs the Heston Hull-White 2F hybrid example on the processes catalog page.

use stochastic_rs::ndarray::array;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::volatility::heston_hull_white_2f::HestonHullWhite2F;
use stochastic_rs::traits::ProcessExt;

/// The short rate's drift target, here holding its mean at 3 percent under `a = 1.1`.
fn theta_r(_t: f64) -> f64 {
  1.1 * 0.03
}

#[test]
fn heston_hull_white_2f_sample() {
  // Drivers in the order (x, v, r, u): equity-variance −0.3, equity-rate 0.35 and 0.08, rate-rate −0.4.
  let rho = array![
    [1.0, -0.3, 0.35, 0.08],
    [-0.3, 1.0, 0.0, 0.0],
    [0.35, 0.0, 1.0, -0.4],
    [0.08, 0.0, -0.4, 1.0],
  ];
  let hybrid = HestonHullWhite2F::<f64, _, _>::new(
    1.0, // s0
    0.2, // v0
    0.8, // kappa
    0.2, // theta
    0.2, // sigma
    theta_r as fn(f64) -> f64,
    1.1,   // a
    0.01,  // sigma1
    0.015, // sigma2
    0.8,   // b
    rho,
    Some(0.03), // r0
    253,
    Some(1.0), // t
    Deterministic::new(42),
  );
  // Spot, variance, short rate, auxiliary factor and the integrated short rate, 253 points each.
  let path = hybrid.sample();
  assert_eq!(path.s.len(), 253);
  assert_eq!((path.s[0], path.v[0], path.r[0]), (1.0, 0.2, 0.03));
  assert_eq!((path.u[0], path.integrated_rate[0]), (0.0, 0.0));

  // The same model under Andersen's quadratic-exponential variance step.
  let discounted = hybrid
    .qe()
    .sample_map(1_000, |p| p.s[252] * (-p.integrated_rate[252]).exp());
  let mean = discounted.iter().sum::<f64>() / 1_000.0;
  assert!(
    (mean - 1.0).abs() < 0.1,
    "discounted spot is a martingale: {mean}"
  );
}
