// docs: processes#heston-stochastic-volatility
//! Backs the Heston example on the processes catalog page.

use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::volatility::HestonPow;
use stochastic_rs::stochastic::volatility::heston::Heston;
use stochastic_rs::traits::ProcessExt;

#[test]
fn heston_two_factor_sample() {
  let p = Heston::<f64, _>::new(
    Some(100.0),
    Some(0.04),
    2.0,
    0.04,
    0.3,
    -0.7,
    0.03,
    1_000,
    Some(1.0),
    HestonPow::Sqrt,
    Some(true),
    Unseeded,
  );
  let [s_path, v_path] = p.sample();
  assert_eq!(s_path.len(), 1_000);
  assert_eq!(v_path.len(), 1_000);
  assert!(v_path.iter().all(|&v| v >= 0.0));
}
