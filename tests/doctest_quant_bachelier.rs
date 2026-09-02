// docs: quant#bachelier-normal-model-and-implied-normal-volatility
//! Backs the Bachelier example on the quant catalog page.

use stochastic_rs::quant::pricing::bachelier::BachelierPricer;
use stochastic_rs::quant::pricing::bachelier::normal_implied_volatility;
use stochastic_rs::quant::types::OptionType;
use stochastic_rs::traits::ModelPricer;

#[test]
fn bachelier_price_and_normal_implied_volatility() {
  // Model state: normal volatility of 20 price units per √year.
  let model = BachelierPricer::new(20.0);
  // Query: (s, k, r, q, tau) — a strike below the forward.
  let (s, k, r, q, tau) = (100.0, 95.0, 0.05, 0.02, 0.75);

  let call = model.price_call(s, k, r, q, tau);
  let put = model.price_put(s, k, r, q, tau);
  let forward = model.forward(s, r, q, tau);
  assert!((call - put - (-r * tau).exp() * (forward - k)).abs() < 1e-9);

  // The inversion recovers the normal volatility from the price.
  let implied = model.implied_volatility(call, s, k, r, q, tau, OptionType::Call);
  assert!((implied - 20.0).abs() < 1e-9);

  // Forward-space entry point for rate options quoted in normal vol (bp).
  let undiscounted = call * (r * tau).exp();
  assert!(
    (normal_implied_volatility(undiscounted, forward, k, tau, OptionType::Call) - 20.0).abs()
      < 1e-9
  );
}
