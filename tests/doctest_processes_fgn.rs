// docs: processes#fractional-brownian-motion
//! Backs the fGN example on the processes catalog page.

use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::noise::fgn::Fgn;
use stochastic_rs::traits::ProcessExt;

#[test]
fn fgn_sample() {
  let fgn = Fgn::<f64, _>::new(
    /* hurst */ 0.3,
    /* n */ 4096,
    /* t */ Some(1.0),
    Deterministic::new(42),
  );
  let increments = fgn.sample();
  assert_eq!(increments.len(), 4096);
}
