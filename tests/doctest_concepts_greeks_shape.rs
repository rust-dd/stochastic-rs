// docs: concepts/traits#greeksext-and-the-greeks-aggregator
//! Backs the `Greeks` field list on the traits-overview concept page.
//! `Greeks` is a plain `f64` struct (not generic over `T`), verified here
//! by constructing the real type with exactly the 9 named fields shown.

use stochastic_rs::traits::Greeks;

#[test]
fn greeks_has_the_nine_documented_fields() {
  let g = Greeks {
    delta: 0.5,
    gamma: 0.1,
    vega: 0.2,
    theta: -0.05,
    rho: 0.3,
    vanna: 0.4,
    charm: 0.05,
    volga: 0.6,
    veta: -0.02,
  };
  assert_eq!(g.as_array().len(), 9);
}
