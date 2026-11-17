//! `E[e^X] = 1` normalises Gil–Pelaez's share-measure integral, also where `ρσ > κ`.

use num_complex::Complex64;
use stochastic_rs_quant::calibration::hkde::HkdeParams;
use stochastic_rs_quant::calibration::svj::SvjParams;
use stochastic_rs_quant::pricing::fourier::BatesFourier;
use stochastic_rs_quant::pricing::fourier::DoubleHestonFourier;
use stochastic_rs_quant::pricing::fourier::FourierModelExt;
use stochastic_rs_quant::pricing::fourier::GilPelaezPricer;
use stochastic_rs_quant::pricing::fourier::HestonFourier;
use stochastic_rs_quant::pricing::fourier::HkdeFourier;
use stochastic_rs_quant::pricing::fourier::LewisPricer;
use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::pricing::heston::HestonPricer;
use stochastic_rs_quant::traits::ModelPricer;

/// `(κ, σ, ρ)` with `ρσ ≥ κ` at `v0 = θ = 0.04`, and the call at `(100, 100, 0.03, 0.01, 0.1)` by
/// Lewis quadrature of the same chf at 30 digits (mpmath).
const SETS: [(f64, f64, f64, f64); 4] = [
  (1e-4, 0.3, 0.3, 2.5954896882032636),
  (0.05, 0.3, 0.3, 2.5955754154738452),
  (1.0, 2.0, 0.6, 1.9368426300054761),
  (0.3, 0.6, 0.5, 2.528731892864258),
];

fn params(kappa: f64, sigma: f64, rho: f64) -> HestonParams {
  HestonParams {
    v0: 0.04,
    kappa,
    theta: 0.04,
    sigma,
    rho,
  }
}

fn is_one(model: &impl FourierModelExt, name: &str) {
  let one = model.chf(0.1, Complex64::new(0.0, -1.0));
  assert!((one - 1.0).norm() < 1e-13, "{name}: E[e^X] = {one}");
}

#[test]
fn heston_prices_where_rho_sigma_exceeds_kappa() {
  for (kappa, sigma, rho, want) in SETS {
    let p = params(kappa, sigma, rho);
    let model = HestonFourier::new(p);
    is_one(&model, "heston");
    let lewis = LewisPricer::price_call(&model, 100.0, 100.0, 0.03, 0.01, 0.1);
    // Lewis's relative 1e-8 on the integral is 7e-7 in price at `σ = 2`.
    for (name, got, tol) in [
      (
        "blanket",
        model.price_call(100.0, 100.0, 0.03, 0.01, 0.1),
        1e-7,
      ),
      ("lewis", lewis, 2e-6),
      (
        "closed",
        HestonPricer::new(p).price_call(100.0, 100.0, 0.03, 0.01, 0.1),
        1e-7,
      ),
    ] {
      assert!(
        (got - want).abs() < tol,
        "{name} at ({kappa}, {sigma}, {rho}): {got} vs {want}"
      );
    }
  }
}

#[test]
fn the_models_sharing_the_heston_factor_price_there_too() {
  for (kappa, sigma, rho, _) in SETS {
    let p = params(kappa, sigma, rho);
    let bates = BatesFourier::new(SvjParams {
      v0: p.v0,
      kappa,
      theta: p.theta,
      sigma_v: sigma,
      rho,
      lambda: 0.5,
      mu_j: -0.1,
      sigma_j: 0.15,
    });
    let hkde = HkdeFourier::new(HkdeParams {
      v0: p.v0,
      kappa,
      theta: p.theta,
      sigma_v: sigma,
      rho,
      lambda: 0.8,
      p_up: 0.4,
      eta1: 10.0,
      eta2: 5.0,
    });
    let double = DoubleHestonFourier::new(params(1.5, 0.3, -0.7), p);
    is_one(&bates, "bates");
    is_one(&hkde, "hkde");
    is_one(&double, "double heston");
    for (name, gil_pelaez, lewis) in [
      (
        "bates",
        GilPelaezPricer::price_call(&bates, 100.0, 100.0, 0.03, 0.01, 0.1),
        LewisPricer::price_call(&bates, 100.0, 100.0, 0.03, 0.01, 0.1),
      ),
      (
        "hkde",
        GilPelaezPricer::price_call(&hkde, 100.0, 100.0, 0.03, 0.01, 0.1),
        LewisPricer::price_call(&hkde, 100.0, 100.0, 0.03, 0.01, 0.1),
      ),
      (
        "double heston",
        GilPelaezPricer::price_call(&double, 100.0, 100.0, 0.03, 0.01, 0.1),
        LewisPricer::price_call(&double, 100.0, 100.0, 0.03, 0.01, 0.1),
      ),
    ] {
      assert!(
        (gil_pelaez - lewis).abs() < 2e-6,
        "{name} at ({kappa}, {sigma}, {rho}): {gil_pelaez} vs {lewis}"
      );
    }
  }
}
