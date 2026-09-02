// docs: quant#quanto-option-fixed-exchange-rate-foreign-equity
//! Backs the quanto example on the quant catalog page.

use stochastic_rs::quant::pricing::bsm::BSMCoc;
use stochastic_rs::quant::pricing::bsm::BSMPricer;
use stochastic_rs::quant::pricing::quanto::QuantoPricer;
use stochastic_rs::traits::ModelPricer;

#[test]
fn quanto_price_forward_and_merton_reduction() {
  // Model state: asset vol, FX vol, correlation, foreign rate, fixed rate.
  let model = QuantoPricer::new(0.2, 0.12, 0.3, 0.05, 1.5);
  // Query: (s, k, r = domestic rate, q, tau) — the Haug §2.13.4 inputs.
  let (s, k, r, q, tau) = (100.0, 105.0, 0.08, 0.04, 0.5);

  let (call, put) = model.call_put(s, k, r, q, tau);
  assert!((call - 5.2936847941).abs() < 1e-4);
  assert!((put - 12.2976985036).abs() < 1e-4);
  assert_eq!(model.price_call(s, k, r, q, tau), call);

  // The quanto forward carries the adjusted drift r_f − q − ρ σ_S σ_E.
  assert!((model.forward(s, q, tau) - 150.2101470686).abs() < 1e-9);

  // Without correlation and with equal rates it is E_p times Merton (1973).
  let plain = QuantoPricer::new(0.2, 0.12, 0.0, r, 1.5);
  let merton = BSMPricer::new(0.2, BSMCoc::Merton1973);
  assert!(
    (plain.price_call(s, k, r, q, tau) - 1.5 * merton.price_call(s, k, r, q, tau)).abs() < 1e-12
  );
}
