// docs: getting-started/quickstart#1-simulate-an-ou-path
//! Backs vignette 1 (simulate an OU path) on the quickstart page.

use stochastic_rs::prelude::*;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::diffusion::ou::Ou;

#[test]
fn simulate_an_ou_path() {
  let p = Ou::<f64, _>::new(2.0, 0.0, 1.0, 1_000, Some(0.0), Some(1.0), Unseeded);
  let path = p.sample();
  assert_eq!(path.len(), 1_000);
  assert!(path.mean().unwrap().is_finite());
}
