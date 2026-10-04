//! The error policy at its edge cases: NaN for an evaluation outside its domain, a named panic for a
//! broken precondition, never a plausible sentinel.

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_quant::curves::Compounding;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::portfolio::optimizers::empirical_cvar;
use stochastic_rs_quant::pricing::sabr::hagan_implied_vol;
use stochastic_rs_quant::vol_surface::implied::ImpliedVolSurface;

fn flat_curve() -> DiscountCurve<f64> {
  DiscountCurve::from_zero_rates(
    &Array1::from_vec(vec![0.5, 1.0, 5.0]),
    &Array1::from_vec(vec![0.03; 3]),
    InterpolationMethod::LogLinearOnDiscountFactors,
  )
}

#[test]
fn a_yield_at_a_non_positive_maturity_is_nan() {
  assert!(Compounding::Continuous.zero_rate(0.97_f64, 0.0).is_nan());
  assert!(Compounding::Simple.zero_rate(0.97_f64, -1.0).is_nan());
  assert!(flat_curve().zero_rate(0.0).is_nan());
  assert!((flat_curve().zero_rate(1.0) - 0.03).abs() < 1e-12);
}

#[test]
fn a_query_outside_the_domain_is_nan_not_a_panic() {
  assert!(hagan_implied_vol(-1.0, 100.0, 1.0, 0.2, 0.5, 0.3, -0.2).is_nan());
  assert!(hagan_implied_vol(100.0, 0.0, 1.0, 0.2, 0.5, 0.3, -0.2).is_nan());
  assert!(hagan_implied_vol(100.0, 100.0, 1.0, 0.2, 0.5, 0.3, -0.2).is_finite());
}

#[test]
#[should_panic(expected = "returns must satisfy `!returns.is_empty()`")]
fn an_empty_sample_is_a_precondition() {
  let mut empty: Vec<f64> = Vec::new();
  empirical_cvar(&mut empty, 0.05);
}

#[test]
#[should_panic(expected = "strikes must satisfy `strikes.windows(2).all(|w| w[0] < w[1])`")]
fn an_unsorted_strike_grid_is_rejected() {
  let ivs = Array2::from_elem((1, 3), 0.2);
  let _ = ImpliedVolSurface::from_iv_grid(vec![90.0, 110.0, 100.0], vec![1.0], vec![100.0], ivs);
}
