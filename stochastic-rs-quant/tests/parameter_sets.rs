//! Each model's parameters in its paper's order, through every constructor and function.

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
