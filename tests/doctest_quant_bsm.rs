// docs: quant#black-scholes-merton-closed-form-european-call
//! Backs the BSM example on the quant catalog page.

use stochastic_rs::quant::pricing::bsm::BSMCoc;
use stochastic_rs::quant::pricing::bsm::BSMPricer;
use stochastic_rs::quant::types::OptionType;
use stochastic_rs::traits::ModelPricer;

#[test]
fn bsm_price_and_greeks() {
  // Model state only: volatility + cost-of-carry convention.
  let model = BSMPricer::new(/* v */ 0.2, BSMCoc::Bsm1973);
  // Query: (s, k, r, q, tau).
  let (s, k, r, q, tau) = (100.0, 100.0, 0.05, 0.0, 1.0);

  let call = model.price_call(s, k, r, q, tau);
  let put = model.price_put(s, k, r, q, tau);
  let delta = model.delta(s, k, r, q, tau, OptionType::Call);
  let vega = model.vega(s, k, r, q, tau);

  assert!(call > 0.0 && put > 0.0);
  assert!((0.0..=1.0).contains(&delta));
  assert!(vega > 0.0);
}
