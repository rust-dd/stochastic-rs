//! Each model's parameters in its paper's order, through every constructor and function.

use stochastic_rs_quant::pricing::heston::HestonParams;

const P: HestonParams = HestonParams {
  v0: 0.04,
  kappa: 1.5,
  theta: 0.04,
  sigma: 0.3,
  rho: -0.7,
};

#[test]
fn sabr_params_are_in_hagan_order_everywhere() {
  use stochastic_rs_quant::instruments::option::SabrVolatility;
  use stochastic_rs_quant::instruments::option::ShiftedSabrVolatility;
  use stochastic_rs_quant::instruments::option::VolatilityModel;
  use stochastic_rs_quant::pricing::sabr::SabrParams;
  use stochastic_rs_quant::pricing::sabr::SabrPricer;
  use stochastic_rs_quant::pricing::sabr::hagan_implied_vol;
  let p = SabrParams {
    alpha: 0.2,
    beta: 1.0,
    rho: -0.3,
    nu: 0.6,
  };
  // Hagan et al. (2002) eq. A.69a, 50 digits (mpmath): 0.1972899165457992224840861.
  let vol = hagan_implied_vol(110.0, 100.0, 1.0, &p);
  assert!((vol - 0.197_289_916_545_799_2).abs() < 1e-12, "{vol}");
  // The expansion's exact f64 here; a change to its arithmetic will most likely move these bits.
  assert_eq!(vol.to_bits(), 0x3fc9_40cb_c5b2_8809, "{vol}");
  assert_eq!(SabrPricer::new(p).params(), p);
  let model = SabrVolatility::new(0.2_f64, 1.0, -0.3, 0.6);
  assert_eq!(model.implied_volatility(100.0, 110.0, 1.0), vol);
  let shifted = ShiftedSabrVolatility::new(0.2_f64, 1.0, -0.3, 0.6, 0.0);
  assert_eq!(shifted.implied_volatility(100.0, 110.0, 1.0), vol);
}

#[test]
#[should_panic(expected = "rho must satisfy `rho > -1.0 && rho < 1.0`, got rho = 1.0")]
fn sabr_params_new_names_the_broken_predicate() {
  let _ = stochastic_rs_quant::pricing::sabr::SabrParams::new(0.2, 1.0, 1.0, 0.6);
}

#[test]
fn every_heston_carrier_takes_the_same_params() {
  use stochastic_rs_quant::OptionType;
  use stochastic_rs_quant::instruments::equity::EuropeanOption;
  use stochastic_rs_quant::pricing::engines::AnalyticHestonEngine;
  use stochastic_rs_quant::pricing::fourier::GilPelaezPricer;
  use stochastic_rs_quant::pricing::fourier::HestonFourier;
  use stochastic_rs_quant::pricing::heston::HestonPricer;
  use stochastic_rs_quant::pricing::heston_adi::HestonAdiPricer;
  use stochastic_rs_quant::traits::ModelPricer;
  use stochastic_rs_quant::traits::PricingEngine;
  use stochastic_rs_quant::traits::PricingResult;
  let (s, k, r, q, tau) = (100.0, 105.0, 0.03, 0.01, 1.0);
  // Heston (1993) call at P, 30 digits by mpmath: 6.02549536300591493.
  let exact = 6.025_495_363_005_915;
  let closed = HestonPricer::new(P).price_call(s, k, r, q, tau);
  let fourier = GilPelaezPricer::price_call(&HestonFourier::new(P), s, k, r, q, tau);
  let engine = AnalyticHestonEngine::with_constants(s, r, q, P);
  let npv = engine
    .calculate(&EuropeanOption::new_tau(k, OptionType::Call, tau))
    .npv();
  let pde = HestonAdiPricer::new(P).price_call(s, k, r, q, tau);
  assert!((fourier - exact).abs() < 1e-6, "{fourier}");
  // HestonPricer integrates from φ = 1e-5; the dropped head is worth 1.27e-5 at P.
  assert!((closed - exact).abs() < 2e-5, "{closed}");
  assert!((closed - npv).abs() < 1e-12, "{closed} vs {npv}");
  assert!((closed - pde).abs() < 5e-2, "{closed} vs {pde}");
  assert_eq!(engine.params, P);
  assert_eq!(HestonPricer::new(P).params(), P);
  assert_eq!(HestonFourier::new(P).params(), P);
}

#[test]
fn every_heston_constructor_rejects_a_bad_literal_by_name() {
  use stochastic_rs_quant::pricing::engines::AnalyticHestonEngine;
  use stochastic_rs_quant::pricing::fourier::DoubleHestonFourier;
  use stochastic_rs_quant::pricing::fourier::HestonFourier;
  use stochastic_rs_quant::pricing::heston::HestonPricer;
  use stochastic_rs_quant::pricing::heston_adi::HestonAdiPricer;
  let good = P;
  let bad = HestonParams { sigma: 0.0, ..P };
  let builds: [fn(HestonParams, HestonParams); 5] = [
    |p, _| {
      HestonPricer::new(p);
    },
    |p, _| {
      HestonAdiPricer::new(p);
    },
    |p, _| {
      HestonFourier::new(p);
    },
    |p, _| {
      AnalyticHestonEngine::with_constants(100.0, 0.03, 0.01, p);
    },
    |p, good| {
      DoubleHestonFourier::new(good, p);
    },
  ];
  for build in builds {
    let payload = std::panic::catch_unwind(|| build(bad, good)).expect_err("must panic");
    let message = payload
      .downcast_ref::<String>()
      .map(String::as_str)
      .unwrap_or_default();
    assert_eq!(
      message,
      "sigma must satisfy `sigma.is_finite() && sigma > 0.0`, got sigma = 0.0"
    );
  }
}

#[test]
#[should_panic(expected = "rho must satisfy `(-1.0..=1.0).contains(&rho)`, got rho = 1.5")]
fn heston_params_new_names_the_broken_predicate() {
  let _ = HestonParams::new(0.04, 1.5, 0.04, 0.3, 1.5);
}
