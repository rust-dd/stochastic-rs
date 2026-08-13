// docs: quant#heston-fourier-pricer-with-cui-analytic-jacobian
//! Backs the Heston pricer example on the quant catalog page. `calculate_price`
//! always returns the call price (Heston's put comes from
//! `calculate_call_put().1`, not a separate code path), so this example
//! prices a call and reports the aggregated Greeks for it.

use stochastic_rs::quant::pricing::heston::HestonPricer;
use stochastic_rs::traits::GreeksExt;
use stochastic_rs::traits::PricerExt;

#[test]
fn heston_pricer_price_and_greeks() {
  let pricer = HestonPricer::builder(
    /* s */ 100.0, /* v0 */ 0.04, /* k */ 100.0, /* r */ 0.03,
    /* rho */ -0.5, /* kappa */ 2.0, /* theta */ 0.04, /* sigma */ 0.3,
  )
  .q(0.0)
  .tau(1.0)
  .build();

  let price = pricer.calculate_price();
  let g = pricer.greeks();

  assert!(price > 0.0);
  assert!((0.0..=1.0).contains(&g.delta));
  assert!(g.vega > 0.0);
  assert!(g.vanna.is_finite());
}
