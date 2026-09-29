// docs: tutorials/heston
//! Simulates seeded Heston paths under the default Euler scheme and under Andersen's QE scheme.

use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::volatility::HestonPow;
use stochastic_rs::stochastic::volatility::heston::Heston;
use stochastic_rs::traits::ProcessExt;

#[test]
fn heston_paths_under_euler_and_qe() {
  let euler = Heston::<f64, _>::new(
    Some(100.0), // s0
    Some(0.04),  // v0, a variance
    2.0,         // kappa
    0.04,        // theta
    0.3,         // sigma, the volatility of variance
    -0.7,        // rho
    0.03,        // mu
    252,         // n grid points, t = 0 included
    Some(1.0),   // t
    HestonPow::Sqrt,
    Some(false), // truncate the variance at zero instead of reflecting it
    Deterministic::new(42),
  );
  let [s, v] = euler.sample();
  assert_eq!((s.len(), v.len()), (252, 252));
  assert!(v.iter().all(|&x| x >= 0.0));

  // Same parameters, Andersen (2008) quadratic-exponential variance step.
  let paths = euler.qe().sample_par(2_000);
  let mean_st = paths.iter().map(|[s, _]| s[251]).sum::<f64>() / 2_000.0;
  assert!((mean_st - 100.0 * 0.03_f64.exp()).abs() < 2.0);
}
