//! A `rand_distr` law fills the jump-size slot, so the dev-dependency pairs with the `rand` the library is built on.

use rand_distr::Normal;
use stochastic_rs::prelude::*;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::process::cpoisson::CompoundPoisson;
use stochastic_rs::stochastic::process::poisson::Poisson;

#[test]
fn a_rand_distr_normal_is_a_jump_size_law() {
  let law = Normal::<f64>::new(0.1, 0.2).unwrap();
  let poisson = Poisson::new(4.0, Some(64), Some(1.0), Unseeded);
  let process = CompoundPoisson::new(law, poisson, Deterministic::new(5));
  let [times, cumulative, jumps] = process.sample();
  assert_eq!(times.len(), cumulative.len());
  assert_eq!(times.len(), jumps.len());
  assert!(cumulative.iter().all(|x| x.is_finite()));
}
