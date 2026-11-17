//! Query limits: NaN outside a pricer's domain, never a sentinel.

use ndarray::Array1;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::credit::HazardInterpolation;
use stochastic_rs_quant::credit::SurvivalCurve;
use stochastic_rs_quant::credit::SurvivalPoint;
use stochastic_rs_quant::curves::CurvePoint;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::pricing::sabr::alpha_from_atm_vol;
use stochastic_rs_quant::pricing::snell_envelope::SnellEnvelopePricer;
use stochastic_rs_quant::traits::ModelPricer;

#[test]
fn the_snell_lattice_answers_nan_when_the_crr_probability_leaves_the_unit_interval() {
  // dt = 0.5, u = e^{0.2 sqrt(0.5)} = 1.152, d = 0.868, growth = e^{0.25} = 1.284: p = 1.46.
  let coarse = SnellEnvelopePricer::new(0.2, 2);
  assert!(coarse.price_call(100.0, 100.0, 0.5, 0.0, 1.0).is_nan());
  let detailed = coarse.price_detailed(100.0, 100.0, 0.5, 0.0, 1.0, OptionType::Call);
  assert!(detailed.price.is_nan() && detailed.european_price.is_nan());
  assert!(detailed.exercise_boundary.is_empty());
  let fine = SnellEnvelopePricer::new(0.2, 400);
  assert!(fine.price_call(100.0, 100.0, 0.5, 0.0, 1.0).is_finite());
}

#[test]
#[should_panic(expected = "steps must satisfy `steps >= 1`, got steps = 0")]
fn a_snell_lattice_without_a_step_is_a_precondition() {
  SnellEnvelopePricer::new(0.2, 0);
}

#[test]
fn the_discount_curve_has_no_zero_sentinels() {
  let curve = DiscountCurve::<f64>::from_zero_rates(
    &Array1::from_vec(vec![1.0, 2.0, 5.0]),
    &Array1::from_vec(vec![0.03; 3]),
    InterpolationMethod::LogLinearOnDiscountFactors,
  );
  assert!(curve.simple_forward_rate(2.0, 1.0).is_nan());
  assert!(curve.simple_forward_rate(1.0, 1.0).is_nan());
  assert!((curve.simple_forward_rate(1.0, 2.0) - (0.03_f64.exp() - 1.0)).abs() < 1e-12);
  assert!(curve.forward_rate(2.0, 1.0).is_nan());
  assert!(curve.forward_rate(1.0, 1.0).is_nan());
  assert!((curve.forward_rate(1.0, 2.0) - 0.03).abs() < 1e-12);
  assert!(curve.forward_rate(f64::NAN, 2.0).is_nan() && curve.forward_rate(1.0, f64::NAN).is_nan());
  assert_eq!(curve.forward_rates(&Array1::from_vec(vec![1.0])).len(), 0);
  assert_eq!(curve.forward_rates(&Array1::from_vec(vec![])).len(), 0);
  let forwards = curve.forward_rates(&Array1::from_vec(vec![1.0, 2.0, 2.0]));
  assert!((forwards[0] - 0.03).abs() < 1e-12 && forwards[1].is_nan());
  assert!(curve.par_rate(0.0, 2).is_nan());
  assert!(curve.par_rate(-1.0, 2).is_nan());
  assert!(curve.par_rate(1.0, 0).is_nan());
  assert!(curve.par_rate(f64::INFINITY, 2).is_nan());
}

#[test]
#[should_panic(expected = "points must satisfy `p.time.is_finite()` for every p, got p.time = NaN")]
fn a_discount_curve_pillar_at_a_nan_time_is_a_precondition() {
  let point = |time, discount_factor| CurvePoint {
    time,
    discount_factor,
  };
  let method = InterpolationMethod::LogLinearOnDiscountFactors;
  DiscountCurve::new(vec![point(1.0, 0.97), point(f64::NAN, 0.95)], method);
}

#[test]
#[should_panic(expected = "points must satisfy `p.time.is_finite()` for every p, got p.time = NaN")]
fn a_survival_curve_node_at_a_nan_time_is_a_precondition() {
  let point = |time, survival_probability| SurvivalPoint {
    time,
    survival_probability,
  };
  let method = HazardInterpolation::PiecewiseConstantHazard;
  SurvivalCurve::new(vec![point(1.0, 0.98), point(f64::NAN, 0.95)], method);
}

#[test]
fn alpha_from_atm_vol_is_nan_for_query_data_outside_its_domain_and_panics_only_on_rho() {
  assert!(alpha_from_atm_vol(0.2, -1.0, 1.0, 0.5, -0.3, 0.6).is_nan());
  assert!(alpha_from_atm_vol(0.0, 100.0, 1.0, 0.5, -0.3, 0.6).is_nan());
  assert!(alpha_from_atm_vol(0.2, 100.0, -1.0, 0.5, -0.3, 0.6).is_nan());
  assert!(alpha_from_atm_vol(0.2, 100.0, 1.0, 0.5, -0.3, 0.6).is_finite());
  // Newton from α₀ = 0.2 runs to the negative root of 12.375α² − 18.59α − 0.2.
  assert!(alpha_from_atm_vol(0.2, 100.0, 5.0, 1.0, 0.99, 10.0).is_nan());
  // β = 1, ν = 0: α is σ_ATM itself, however small.
  assert_eq!(alpha_from_atm_vol(1e-9, 1.0, 1.0, 1.0, 0.0, 0.0), 1e-9);
  let err = std::panic::catch_unwind(|| alpha_from_atm_vol(0.2, 100.0, 1.0, 0.5, 1.0, 0.6));
  let err = err.unwrap_err();
  let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
  assert_eq!(
    msg,
    "rho must satisfy `rho > -1.0 && rho < 1.0`, got rho = 1.0"
  );
}
