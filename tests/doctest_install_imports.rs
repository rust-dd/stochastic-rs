// docs: getting-started/installation-rust#umbrella-crate-everything
//! Backs the umbrella-crate import example on the Rust installation page.

use stochastic_rs::prelude::*;
use stochastic_rs::quant::pricing::heston::HestonPricer;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::diffusion::gbm::Gbm;

#[test]
fn umbrella_imports_resolve() {
  let p = Gbm::<f64, _>::new(0.05, 0.2, 32, Some(100.0), Some(1.0), Deterministic::new(1));
  let path = p.sample();
  assert_eq!(path.len(), 32);

  fn _type_check(_: &HestonPricer) {}
}
