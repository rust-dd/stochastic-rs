//! The characteristic-function pricers against references computed outside the crate: mpmath at
//! 30 digits, each value written beside its cell `(s, k, r, q, τ)`.

use stochastic_rs_quant::pricing::fourier::BsmFourier;
use stochastic_rs_quant::pricing::fourier::CarrMadanPricer;
use stochastic_rs_quant::pricing::fourier::FrftCarrMadanPricer;
use stochastic_rs_quant::pricing::fourier::GilPelaezPricer;
use stochastic_rs_quant::pricing::fourier::HestonFourier;
use stochastic_rs_quant::pricing::fourier::HkdeFourier;
use stochastic_rs_quant::pricing::fourier::HkdeParams;
use stochastic_rs_quant::pricing::fourier::LewisPricer;
use stochastic_rs_quant::pricing::fourier::MertonJdFourier;
use stochastic_rs_quant::pricing::fourier::VarianceGammaFourier;
use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::pricing::heston::HestonPricer;
use stochastic_rs_quant::traits::ModelPricer;

const MERTON: MertonJdFourier = MertonJdFourier {
  sigma: 0.15,
  lambda: 0.5,
  mu_j: -0.1,
  sigma_j: 0.2,
};

const VARIANCE_GAMMA: VarianceGammaFourier = VarianceGammaFourier {
  sigma: 0.12,
  theta: -0.14,
  nu: 0.2,
};

/// `ρσ ≥ κ`: the share measure's variance explodes, `E₁[∫v] = 4.0e7` at `τ = 10`.
const EXPLODING: HestonParams = HestonParams {
  v0: 0.04,
  kappa: 1.0,
  theta: 0.25,
  sigma: 3.0,
  rho: 1.0,
};

fn assert_within(name: &str, got: f64, want: f64, tolerance: f64) {
  assert!(
    (got - want).abs() < tolerance,
    "{name}: {got:e} vs {want:e}, off by {:e}",
    got - want
  );
}

/// Merton's series of Black prices at a maturity of hours, where a walk that stops on the chf's
/// envelope truncates the jump part.
#[test]
fn lewis_and_gil_pelaez_hold_a_short_dated_jump_price() {
  let (s, k, r, q, tau) = (100.0, 101.0, 0.05, 0.01, 1e-3);
  let want = 0.005_306_410_568_564_315_2;
  assert_within("blanket", MERTON.price_call(s, k, r, q, tau), want, 1e-13);
  let lewis = LewisPricer::price_call(&MERTON, s, k, r, q, tau);
  assert_within("lewis", lewis, want, 1e-13);
  let gil_pelaez = GilPelaezPricer::price_call(&MERTON, s, k, r, q, tau);
  assert_within("gil-pelaez", gil_pelaez, want, 1e-12);
}

/// Variance Gamma's gamma mixture, deep out of the money: a price of `2.4e-6` on a spot of 100.
#[test]
fn lewis_and_gil_pelaez_hold_a_deep_out_of_the_money_price() {
  let (s, k, r, q, tau) = (100.0, 150.0, 0.05, 0.01, 0.25);
  let want = 2.449_722_647_811_250_4e-6;
  assert_within(
    "blanket",
    VARIANCE_GAMMA.price_call(s, k, r, q, tau),
    want,
    1e-13,
  );
  let lewis = LewisPricer::price_call(&VARIANCE_GAMMA, s, k, r, q, tau);
  assert_within("lewis", lewis, want, 1e-13);
  let gil_pelaez = GilPelaezPricer::price_call(&VARIANCE_GAMMA, s, k, r, q, tau);
  assert_within("gil-pelaez", gil_pelaez, want, 5e-12);
}

/// Lewis's integral and `Se^{−qτ}P₁ − Ke^{−rτ}P₂` agree on the reference to 20 digits.
#[test]
fn every_pricer_holds_a_price_whose_share_measure_explodes() {
  let (s, k, r, q, tau) = (100.0, 100.0, 0.03, 0.01, 10.0);
  let want = 58.504_753_668_889_02;
  let model = HestonFourier::new(EXPLODING);
  assert_within("blanket", model.price_call(s, k, r, q, tau), want, 1e-11);
  let lewis = LewisPricer::price_call(&model, s, k, r, q, tau);
  assert_within("lewis", lewis, want, 1e-11);
  let gil_pelaez = GilPelaezPricer::price_call(&model, s, k, r, q, tau);
  assert_within("gil-pelaez", gil_pelaez, want, 2e-8);
  let closed = HestonPricer::new(EXPLODING).price_call(s, k, r, q, tau);
  assert_within("heston", closed, want, 1e-11);
}

/// HKDE at `ρσ ≥ κ` and `τ = 15`: the jumps' compensator turns the integrand's phase 0.83 radians
/// per unit `u` beside `κ`'s 0.3, and the walk hands over to the tail rule on the sum.
#[test]
fn lewis_holds_a_price_whose_compensator_turns_its_phase() {
  let model = HkdeFourier::new(HkdeParams {
    v0: 0.09,
    kappa: 0.2,
    theta: 0.01,
    sigma: 1.5,
    rho: 0.95,
    lambda: 1.0,
    p_up: 0.4,
    eta1: 10.0,
    eta2: 5.0,
  });
  let (s, r, q, tau) = (100.0, 0.03, 0.01, 15.0);
  // The last strike is the forward, where `κ = 0`.
  for (k, want) in [
    (100.0, 40.091_698_837_618_036),
    (80.0, 46.355_092_420_111_84),
    (134.985_880_757_600_32, 31.704_182_907_960_47),
  ] {
    let name = format!("k = {k}");
    assert_within(&name, model.price_call(s, k, r, q, tau), want, 1e-11);
    let lewis = LewisPricer::price_call(&model, s, k, r, q, tau);
    assert_within(&name, lewis, want, 1e-11);
  }
}

/// `(k, τ)` and Black–Scholes's call at `σ = 0.2`, `(s, r, q) = (100, 0.05, 0)`.
const BLACK_SCHOLES: [(f64, f64, f64); 3] = [
  (80.0, 0.1, 20.399_215_828_168_058),
  (100.0, 1.0, 10.450_583_572_185_567),
  (120.0, 5.0, 20.578_973_389_573_955),
];

/// What the trapezoid rule leaves is its image, `Se^{−2πα/η} = 6.5e-7` at the default `α` and `η`;
/// Simpson's weights left `2.7e-3`.
#[test]
fn carr_madan_is_off_by_the_trapezoid_rules_image_alone() {
  let model = BsmFourier { sigma: 0.2 };
  for (k, tau, want) in BLACK_SCHOLES {
    let got = CarrMadanPricer::default().price_call(&model, 100.0, k, 0.05, 0.0, tau);
    assert_within(&format!("k = {k}, tau = {tau}"), got, want, 1e-6);
  }
}

/// The fractional transform interpolates between its nodes, which leaves `3.7e-5` at worst here.
#[test]
fn the_fractional_transform_is_off_by_its_interpolation_alone() {
  let model = BsmFourier { sigma: 0.2 };
  for (k, tau, want) in BLACK_SCHOLES {
    let got = FrftCarrMadanPricer::default().price_call(&model, 100.0, k, 0.05, 0.0, tau);
    assert_within(&format!("k = {k}, tau = {tau}"), got, want, 5e-5);
  }
}
