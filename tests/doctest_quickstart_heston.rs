// docs: getting-started/quickstart#2-price-a-heston-european-call
//! Backs vignette 2 (price a Heston European call) on the quickstart
//! page.

use stochastic_rs::prelude::*;
use stochastic_rs::quant::pricing::heston::HestonPricer;

#[test]
fn price_a_heston_european_call() {
  let pricer = HestonPricer::builder(
    /* s */ 100.0, /* v0 */ 0.04, /* k */ 100.0, /* r */ 0.03,
    /* rho */ -0.5, /* kappa */ 2.0, /* theta */ 0.04, /* sigma */ 0.3,
  )
  .q(0.0)
  .tau(1.0)
  .build();

  let price = pricer.calculate_price();
  let greeks = pricer.greeks();
  assert!(price > 0.0);
  assert!(greeks.vega > 0.0);
}
