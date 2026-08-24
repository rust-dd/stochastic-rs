// docs: quant#heston-fourier-pricer-with-cui-analytic-jacobian
//! Backs the Heston pricer example on the quant catalog page. The model
//! holds only its six Heston parameters; spot, strike, rate, dividend
//! yield and maturity travel to the call, so one instance prices a whole
//! strike/maturity grid.

use stochastic_rs::quant::OptionType;
use stochastic_rs::quant::pricing::heston::HestonPricer;
use stochastic_rs::traits::ModelPricer;

#[test]
fn heston_pricer_price_and_greeks() {
  let model = HestonPricer::new(
    /* v0 */ 0.04, /* rho */ -0.5, /* kappa */ 2.0, /* theta */ 0.04,
    /* sigma */ 0.3, /* lambda */ None,
  );
  let (s, k, r, q, tau) = (100.0, 100.0, 0.03, 0.0, 1.0);

  let price = model.price_call(s, k, r, q, tau);
  let g = model.greeks(s, k, r, q, tau, OptionType::Call);

  assert!(price > 0.0);
  assert!((0.0..=1.0).contains(&g.delta));
  assert!(g.vega > 0.0);
  assert!(g.vanna.is_finite());
}
