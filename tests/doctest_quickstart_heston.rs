// docs: getting-started/quickstart#2-price-a-heston-european-call
//! Backs vignette 2 (price a Heston European call) on the quickstart
//! page.

use stochastic_rs::prelude::*;
use stochastic_rs::quant::OptionType;
use stochastic_rs::quant::pricing::heston::HestonPricer;

#[test]
fn price_a_heston_european_call() {
  let model = HestonPricer::new(
    /* v0 */ 0.04, /* rho */ -0.5, /* kappa */ 2.0, /* theta */ 0.04,
    /* sigma */ 0.3, /* lambda */ None,
  );
  let (s, k, r, q, tau) = (100.0, 100.0, 0.03, 0.0, 1.0);

  let price = model.price_call(s, k, r, q, tau);
  let greeks = model.greeks(s, k, r, q, tau, OptionType::Call);
  assert!(price > 0.0);
  assert!(greeks.vega > 0.0);
}
