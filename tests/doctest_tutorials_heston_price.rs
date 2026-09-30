// docs: tutorials/heston
//! Prices a Heston European call in semi-closed form and by Gil-Pelaez Fourier inversion.

use stochastic_rs::quant::OptionType;
use stochastic_rs::quant::pricing::fourier::HestonFourier;
use stochastic_rs::quant::pricing::heston::HestonPricer;
use stochastic_rs::traits::ModelPricer;

#[test]
fn heston_call_two_ways() {
  let (s, k, r, q, tau) = (100.0, 100.0, 0.03, 0.01, 1.0);

  // Model state only: v0, rho, kappa, theta, sigma, lambda (market price of volatility risk).
  let model = HestonPricer::new(0.04, -0.7, 2.0, 0.04, 0.3, None);
  let (call, put) = model.call_put(s, k, r, q, tau);
  let parity = s * (-q * tau).exp() - k * (-r * tau).exp();
  assert!((call - put - parity).abs() < 1e-8);

  // The Fourier model carries r and q for its characteristic function: pass the same values.
  let fourier = HestonFourier {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
    r,
    q,
  };
  assert!((fourier.price_call(s, k, r, q, tau) - call).abs() < 1e-4);

  let greeks = model.greeks(s, k, r, q, tau, OptionType::Call);
  assert!(greeks.delta > 0.0 && greeks.delta < 1.0 && greeks.vega > 0.0);
}
