// docs: quant#black-scholes-merton-closed-form-european-call
//! Backs the BSM example on the quant catalog page.

use stochastic_rs::quant::pricing::bsm::BSMPricer;
use stochastic_rs::quant::types::OptionType;
use stochastic_rs::traits::PricerExt;

#[test]
fn bsm_price_and_greeks() {
  let pricer = BSMPricer::builder(
    /* s */ 100.0, /* v */ 0.2, /* k */ 100.0, /* r */ 0.05,
  )
  .q(0.0)
  .tau(1.0)
  .option_type(OptionType::Call)
  .build();

  let call = pricer.calculate_price();
  let (_, put) = pricer.calculate_call_put(); // (call, put) regardless of option_type
  let delta = pricer.delta();
  let vega = pricer.vega();

  assert!(call > 0.0 && put > 0.0);
  assert!((0.0..=1.0).contains(&delta));
  assert!(vega > 0.0);
}
