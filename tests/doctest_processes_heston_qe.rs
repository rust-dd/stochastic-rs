// docs: processes#discretisation-schemes
//! Backs the Andersen QE discretisation example on the processes catalog page.

use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::params::VarianceBoundary;
use stochastic_rs::stochastic::volatility::HestonPow;
use stochastic_rs::stochastic::volatility::heston::Heston;
use stochastic_rs::traits::ProcessExt;

#[test]
fn heston_qe_scheme_sample() {
  let qe = Heston::<f64>::new(
    100.0,
    0.04,
    2.0,
    0.04,
    0.3,
    -0.7,
    0.03,
    1_000,
    Some(1.0),
    HestonPow::Sqrt,
    VarianceBoundary::Reflect,
    Unseeded,
  )
  .qe();
  let [s_path, v_path] = qe.sample();
  assert_eq!(s_path.len(), 1_000);
  assert_eq!(v_path.len(), 1_000);
}
