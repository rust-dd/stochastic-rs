//! The Heston `P_j` integrals at short maturities: no negative deep out-of-the-money call, and the
//! Black–Scholes limit where the integrand outlasts the quadrature.

use stochastic_rs_quant::pricing::bsm::BsmCoc;
use stochastic_rs_quant::pricing::bsm::BsmPricer;
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
      assert!(price > -1e-8, "k = {k}, tau = {tau}: {price}");
      let time_value = price - (100.0 - k * (-0.05 * tau).exp()).max(0.0);
      let vol = surface.ivs[[j, i]];
      assert!(
        time_value <= 1e-6 || !vol.is_nan(),
        "k = {k}, tau = {tau}: {price} has no vol"
      );
    }
  }
}

/// Below `τ ≈ 1e-5` the integrand outlasts the quadrature's panels and the Gaussian share carries
/// the price: to 1.4e-6 at worst, near `τ = 1.6e-6`, and to rounding as `τ → 0`.
#[test]
fn a_maturity_of_seconds_prices_at_its_black_scholes_limit() {
  let pricer = HestonPricer::new(P);
  let bsm = BsmPricer::new(0.2, BsmCoc::Merton1973);
  for i in 0..=40 {
    let tau = 10f64.powf(-12.0 + 0.2 * f64::from(i));
    let tol = if tau < 1e-8 { 1e-8 } else { 2e-6 };
    for s in [90.0, 110.0] {
      let got = pricer.price_call(s, 100.0, 0.05, 0.0, tau);
      let intrinsic = (s - 100.0 * (-0.05 * tau).exp()).max(0.0);
      assert!(
        (got - intrinsic).abs() < tol,
        "s = {s}, tau = {tau}: {got} vs {intrinsic}"
      );
    }
    if tau < 1e-8 {
      let got = pricer.price_call(100.0, 100.0, 0.05, 0.0, tau);
      let want = bsm.price_call(100.0, 100.0, 0.05, 0.0, tau);
      assert!(
        (got - want).abs() < 1e-8,
        "at the money, tau = {tau}: {got} vs {want}"
      );
    }
  }
}

/// `E₁[∫v] = 3.7e4` at `ρσ > κ`, where a Gaussian that narrow is past the quadrature: the control
/// variance stops at its cap and the price keeps the head `S·1e-8·(E₁ − cap)/(2π) = 5.5e-3`.
#[test]
fn an_exploding_share_measure_variance_keeps_the_quadrature() {
  let pricer = HestonPricer::new(HestonParams {
    kappa: 0.5,
    sigma: 1.0,
    rho: 0.9,
    ..P
  });
  // (τ, call at (100, 100, 0.05, 0)) by Lewis quadrature at 30 digits (mpmath).
  for (tau, want, tol) in [
    (10.0, 40.309405418969645, 1e-9),
    (30.0, 77.97317902193187, 1e-2),
  ] {
    let got = pricer.price_call(100.0, 100.0, 0.05, 0.0, tau);
    assert!((got - want).abs() < tol, "tau = {tau}: {got} vs {want}");
  }
}

/// Calls at `K = 100`, `r = 0.05`, `q = 0` by Lewis quadrature of the chf at 30 digits (mpmath).
#[test]
fn short_maturities_match_the_converged_integral() {
  let pricer = HestonPricer::new(P);
  for (tau, s, want) in [
    (1e-4, 110.0, 10.000499998750002),
    (1e-4, 100.0, 0.08003814985406024),
    (1e-4, 90.0, 0.0),
    (1e-3, 110.0, 10.004999875002083),
    (1e-3, 100.0, 0.254803539033173),
    (1e-3, 90.0, 0.0),
    (1e-2, 110.0, 10.049992063341942),
    (1e-2, 100.0, 0.8225772944091662),
    (1e-2, 90.0, 1.6619165947277462e-10),
  ] {
    let got = pricer.price_call(s, 100.0, 0.05, 0.0, tau);
    assert!(
      (got - want).abs() < 1e-8,
      "s = {s}, tau = {tau}: {got} vs {want}"
    );
  }
}

/// With `v0 = θ = 0` the variance stays at 0 and the integrand never decays: the price is the
/// discounted payoff of the forward.
#[test]
fn a_zero_variance_model_prices_the_forward_payoff() {
  let pricer = HestonPricer::new(HestonParams {
    v0: 0.0,
    theta: 0.0,
    ..P
  });
  let forward = 100.0 * 0.05_f64.exp();
  for k in [90.0, forward, 110.0] {
    let got = pricer.price_call(100.0, k, 0.05, 0.0, 1.0);
    let want = (100.0 - k * (-0.05_f64).exp()).max(0.0);
    assert!((got - want).abs() < 1e-9, "k = {k}: {got} vs {want}");
  }
}

/// A risk premium of −800 puts `E[∫₀^τ v]` past `f64` at `τ = 1`.
#[test]
fn an_expected_variance_past_f64_prices_nan() {
  let pricer = HestonPricer::new(P).with_vol_risk_premium(-800.0);
  let (call, put) = pricer.call_put(100.0, 100.0, 0.05, 0.0, 1.0);
  assert!(call.is_nan() && put.is_nan(), "{call}, {put}");
}
