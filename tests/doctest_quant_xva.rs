// docs: quant#xva-exposure-profiles-and-cva--dva--fva
//! Backs the XVA example on the quant catalog page.

use ndarray::Array1;
use stochastic_rs::quant::credit::survival_curve::HazardInterpolation;
use stochastic_rs::quant::credit::survival_curve::SurvivalCurve;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::InterpolationMethod;
use stochastic_rs::quant::risk::xva::cva;
use stochastic_rs::quant::risk::xva::fva;
use stochastic_rs::quant::risk::xva::irs::HullWhiteSwapExposure;
use stochastic_rs::simd_rng::Deterministic;

#[test]
fn cva_of_a_par_swap_under_hull_white() {
  // Flat 3 % curve, five-year annual payer swap of 1m notional at par.
  let curve = DiscountCurve::from_zero_rates(
    &Array1::from_vec(vec![0.5, 1.0, 5.0, 10.0]),
    &Array1::from_vec(vec![0.03; 4]),
    InterpolationMethod::LogLinearOnDiscountFactors,
  );
  let mut swap = HullWhiteSwapExposure::new(
    0.1,
    0.01,
    1_000_000.0,
    0.0,
    vec![1.0, 2.0, 3.0, 4.0, 5.0],
    1.0,
  );
  swap.fixed_rate = swap.par_rate(&curve);

  // Exposure profile on the payment dates from 4 000 Hull–White short-rate paths.
  let profile = swap.profile(&curve, 4_000, 0.95, Deterministic::new(7));
  assert!(profile.peak_epe() > 0.0 && profile.epe[4] == 0.0);

  // CVA against a 2 % flat hazard at 60 % LGD, and the symmetric FVA at a 50 bp funding spread.
  let counterparty = SurvivalCurve::from_hazard_rates(
    &Array1::from_vec(vec![1.0, 5.0, 10.0]),
    &Array1::from_vec(vec![0.02; 3]),
    HazardInterpolation::PiecewiseConstantHazard,
  );
  let cva_value = cva(&profile, &counterparty, &curve, 0.6);
  assert!(cva_value > 0.0 && cva_value < 0.01 * swap.notional);
  let _funding = fva(&profile, &curve, 0.005);
}
