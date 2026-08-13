// docs: getting-started/installation-rust#verify-the-install
//! Backs the "verify the install" example on the Rust installation page.

use stochastic_rs::prelude::*;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::diffusion::ou::Ou;

#[test]
fn ou_path_has_the_requested_length() {
  let p = Ou::<f64, _>::new(2.0, 0.0, 1.0, 1_000, Some(0.0), Some(1.0), Unseeded);
  let path = p.sample();
  assert_eq!(path.len(), 1_000, "OU path of length {}", path.len());
}
