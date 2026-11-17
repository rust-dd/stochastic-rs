//! The Heston `P_j` integrals start where the Fourier pricers start, so a short-dated deep
//! out-of-the-money call stays within the quadrature tolerance of zero.

use stochastic_rs_quant::pricing::fourier::GilPelaezPricer;
use stochastic_rs_quant::pricing::fourier::HestonFourier;
use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::pricing::heston::HestonPricer;
use stochastic_rs_quant::traits::ModelPricer;
use stochastic_rs_quant::vol_surface::ModelSurface;

const P: HestonParams = HestonParams {
  v0: 0.04,
  kappa: 1.5,
  theta: 0.04,
  sigma: 0.3,
  rho: -0.7,
};

#[test]
fn the_closed_form_and_gil_pelaez_agree_to_the_quadrature_tolerance_at_short_maturities() {
  let closed = HestonPricer::new(P);
  let fourier = HestonFourier::new(P);
  for (k, tau) in [(120.0, 0.01), (110.0, 0.01), (150.0, 0.1), (100.0, 0.02)] {
    let a = closed.price_call(100.0, k, 0.05, 0.0, tau);
    let b = GilPelaezPricer::price_call(&fourier, 100.0, k, 0.05, 0.0, tau);
    assert!(a > -1e-7, "closed form at k = {k}, tau = {tau}: {a}");
    assert!(
      (a - b).abs() < 1e-7,
      "k = {k}, tau = {tau}: closed {a} vs gil-pelaez {b}"
    );
  }
}

#[test]
fn the_short_maturity_surface_has_no_negative_price_and_inverts_every_time_value_above_1e_6() {
  let strikes = (0..20)
    .map(|i| 60.0 + 5.0 * f64::from(i))
    .collect::<Vec<f64>>();
  let maturities = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0];
  let pricer = HestonPricer::new(P);
  let surface = pricer.vol_surface(100.0, 0.05, 0.0, &strikes, &maturities);
  for (j, &tau) in maturities.iter().enumerate() {
    for (i, &k) in strikes.iter().enumerate() {
      let price = pricer.price_call(100.0, k, 0.05, 0.0, tau);
      assert!(price > -2e-7, "k = {k}, tau = {tau}: {price}");
      let time_value = price - (100.0 - k * (-0.05 * tau).exp()).max(0.0);
      let vol = surface.ivs[[j, i]];
      assert!(
        time_value <= 1e-6 || !vol.is_nan(),
        "k = {k}, tau = {tau}: {price} has no vol"
      );
    }
  }
}
