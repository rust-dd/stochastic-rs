// docs: tutorials/heston
//! Prices a Heston European call in semi-closed form and by Gil-Pelaez Fourier inversion.

use stochastic_rs::quant::CallPut;
use stochastic_rs::quant::OptionType;
use stochastic_rs::quant::pricing::fourier::HestonFourier;
use stochastic_rs::quant::pricing::heston::HestonParams;
use stochastic_rs::quant::pricing::heston::HestonPricer;
use stochastic_rs::traits::ModelPricer;

#[test]
fn heston_call_two_ways() {
  let (s, k, r, q, tau) = (100.0, 100.0, 0.03, 0.01, 1.0);

  // Model state only; the query travels to each call.
  let model = HestonPricer::new(HestonParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  });
  let CallPut { call, put } = model.price_call_put(s, k, r, q, tau);
  let parity = s * (-q * tau).exp() - k * (-r * tau).exp();
  assert!((call - put - parity).abs() < 1e-8);

  // The Fourier model holds no rates: the query's r and q carry and discount its price.
  let fourier = HestonFourier::new(HestonParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  });
  assert!((fourier.price_call(s, k, r, q, tau) - call).abs() < 1e-4);

  let greeks = model.greeks(s, k, r, q, tau, OptionType::Call);
  assert!(greeks.delta > 0.0 && greeks.delta < 1.0 && greeks.vega > 0.0);
}
