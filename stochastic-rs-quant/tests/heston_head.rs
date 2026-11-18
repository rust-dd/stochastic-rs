//! The Heston `P_j` integrals at short maturities: no negative deep out-of-the-money call, and the
//! Black–Scholes limit where the integrand outlasts the quadrature.

use stochastic_rs_quant::CallPut;
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

/// Below `τ ≈ 1e-5` the integrand outlasts the walk in `φ`: it is integrated rescaled, and 60
/// standard deviations from the money the Gaussian share is the price.
#[test]
fn a_maturity_of_seconds_prices_at_its_black_scholes_limit() {
  let pricer = HestonPricer::new(P);
  let bsm = BsmPricer::new(0.2, BsmCoc::Merton1973);
  for i in 0..=40 {
    let tau = 10f64.powf(-12.0 + 0.2 * f64::from(i));
    for s in [90.0, 110.0] {
      let got = pricer.price_call(s, 100.0, 0.05, 0.0, tau);
      let intrinsic = (s - 100.0 * (-0.05 * tau).exp()).max(0.0);
      assert!(
        (got - intrinsic).abs() < 1e-8,
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

const HEAVY: HestonParams = HestonParams {
  v0: 0.02,
  kappa: 0.5,
  theta: 0.09,
  sigma: 1.0,
  rho: -0.9,
};

const POSITIVE: HestonParams = HestonParams {
  v0: 0.04,
  kappa: 1.0,
  theta: 0.04,
  sigma: 2.0,
  rho: 0.6,
};

/// `(v0, κ, θ, σ, ρ)`, `τ`, `s` and the call at `K = 100`, `r = 0.05`, `q = 0` by Lewis quadrature
/// at 30 digits (mpmath): a maturity of minutes, 42 standard deviations and 2 from the money.
const MINUTES: [(HestonParams, f64, f64, f64); 8] = [
  (P, 1.4e-6, 101.0, 1.000006999999755),
  (P, 1.4e-6, 100.05, 0.05015550439907494),
  (HEAVY, 2.8e-6, 101.0, 1.00001399999902),
  (HEAVY, 2.8e-6, 100.05, 0.05016745651580408),
  (
    HestonParams { v0: 0.0025, ..P },
    2.2e-5,
    101.0,
    1.0001099999395,
  ),
  (
    HestonParams { v0: 0.0025, ..P },
    2.2e-5,
    100.05,
    0.05025758985321513,
  ),
  (POSITIVE, 1.4e-6, 101.0, 1.000006999999755),
  (POSITIVE, 1.4e-6, 100.05, 0.05015106692168332),
];

#[test]
fn a_maturity_of_minutes_matches_the_converged_integral() {
  for (params, tau, s, want) in MINUTES {
    let got = HestonPricer::new(params).price_call(s, 100.0, 0.05, 0.0, tau);
    assert!(
      (got - want).abs() < 1e-9,
      "{params:?}, s = {s}, tau = {tau}: {got} vs {want}"
    );
  }
}

/// `v0 ≪ σ²τ` leaves a chf far wider than a Gaussian of variance `E[∫v]`, which sets no scale for
/// `φ`: the call stays at its forward's intrinsic value, `100 − 100e^{−0.0005}`.
#[test]
fn a_variance_far_below_its_own_volatility_is_not_rescaled() {
  let pricer = HestonPricer::new(HestonParams {
    v0: 1e-8,
    theta: 0.0,
    ..P
  });
  let got = pricer.price_call(100.0, 100.0, 0.05, 0.0, 1e-2);
  assert!((got - 0.049987502083).abs() < 1e-6, "{got}");
}

/// `E₁[∫v] = 3.7e4` at `ρσ > κ`, where a Gaussian that narrow is past the quadrature: the control
/// variance stops at its cap, and the mean it no longer matches is the head below `φ = 1e-8`.
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
    (10.0, 40.30940541896965, 1e-9),
    (30.0, 77.97317902193187, 1e-6),
  ] {
    let got = pricer.price_call(100.0, 100.0, 0.05, 0.0, tau);
    assert!((got - want).abs() < tol, "tau = {tau}: {got} vs {want}");
  }
  let CallPut { call, put } = pricer.price_call_put(100.0, 100.0, 0.05, 0.0, 40.0);
  assert!(call.is_nan() && put.is_nan(), "{call}, {put}");
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

/// At `τ = 1` a risk premium of −50 puts `E[∫₀^τ v]` past the head term's reach, −800 past `f64`.
#[test]
fn an_expected_variance_past_the_bound_prices_nan() {
  for lambda in [-50.0, -800.0] {
    let pricer = HestonPricer::new(P).with_vol_risk_premium(lambda);
    for k in [80.0, 100.0, 120.0] {
      let CallPut { call, put } = pricer.price_call_put(100.0, k, 0.05, 0.0, 1.0);
      assert!(
        call.is_nan() && put.is_nan(),
        "lambda = {lambda}, k = {k}: {call}, {put}"
      );
    }
  }
}
