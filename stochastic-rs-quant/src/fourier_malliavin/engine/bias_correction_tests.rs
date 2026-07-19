//! Equation-level tests for the rate-efficient volatility-of-volatility estimator.
//!
//! Golden coefficients follow Toscano et al., arXiv:2112.14529v3,
//! equations (11) and (51).

use ndarray::Array1;
use ndarray::array;
use num_complex::Complex;

use super::*;

fn fixture(period: f64) -> FMVol<f64> {
  let n = 32usize;
  let prices = (0..=n)
    .map(|index| {
      let x = index as f64;
      0.013 * x + (0.41 * x).sin() * 0.09 + (0.17 * x).cos() * 0.025
    })
    .collect::<Array1<_>>();
  let times = Array1::linspace(0.0, period, n + 1);
  FMVol::try_with_freq(
    prices.as_slice().unwrap(),
    times.as_slice().unwrap(),
    period,
    5,
    10,
  )
  .unwrap()
}

fn equation_11(engine: &FMVol<f64>, big_m: usize, big_l: usize) -> Array1<Complex<f64>> {
  let mm = big_m + big_l;
  let c_v = engine.vol_coeffs(mm);
  let k_const = engine.compute_bias_correction_constant(big_m);
  let scale = std::f64::consts::TAU.powi(2) / engine.period;
  let mut result = Array1::<Complex<f64>>::zeros(2 * big_l + 1);

  for (index, k) in (-(big_l as i64)..=(big_l as i64)).enumerate() {
    let mut derivative = Complex::<f64>::new(0.0, 0.0);
    let mut quarticity = Complex::<f64>::new(0.0, 0.0);
    for h in -(big_m as i64)..=(big_m as i64) {
      let product = c_v[(mm as i64 + h) as usize] * c_v[(mm as i64 + k - h) as usize];
      let weight = 1.0 - h.unsigned_abs() as f64 / (big_m + 1) as f64;
      derivative += product * weight * h as f64 * (h - k) as f64;
      quarticity += product;
    }
    result[index] = (derivative / (big_m + 1) as f64 - k_const * quarticity) * scale;
  }

  result
}

fn assert_complex_close(actual: Complex<f64>, expected: Complex<f64>, tolerance: f64) {
  assert!(
    (actual - expected).norm() <= tolerance,
    "actual={actual:?}, expected={expected:?}"
  );
}

#[test]
fn corrected_coefficients_match_paper_equation_11() {
  let engine = fixture(std::f64::consts::TAU);
  let actual = engine.bias_corrected_volvol_coefficients(3, 2);
  let expected = equation_11(&engine, 3, 2);

  for (actual, expected) in actual.iter().zip(expected.iter()) {
    assert_complex_close(*actual, *expected, 1e-14);
  }
}

#[test]
fn corrected_coefficients_match_paper_convention_golden() {
  let engine = fixture(std::f64::consts::TAU);
  let actual = engine.bias_corrected_volvol_coefficients(3, 2);
  let expected = array![
    Complex::new(-0.000213773293741925, 0.0001249268059488918),
    Complex::new(-0.00007747624156533374, -0.0000020687176392964814),
    Complex::new(-0.0002086341864677661, 0.0),
    Complex::new(-0.00007747624156533368, 0.000002068717639296492),
    Complex::new(-0.000213773293741925, -0.00012492680594889183),
  ];

  for (actual, expected) in actual.iter().zip(expected.iter()) {
    assert_complex_close(*actual, *expected, 1e-14);
  }
}

#[test]
fn corrected_estimators_obey_time_rescaling() {
  let unit = fixture(1.0);
  let paper = fixture(std::f64::consts::TAU);
  let integrated_unit = unit.integrated_volvol_bias_corrected(Some(3));
  let integrated_paper = paper.integrated_volvol_bias_corrected(Some(3));
  let spot_unit = unit.spot_volvol_bias_corrected(&[0.17], Some(3), Some(2))[0];
  let spot_paper =
    paper.spot_volvol_bias_corrected(&[0.17 * std::f64::consts::TAU], Some(3), Some(2))[0];

  let integrated_error = (integrated_unit - integrated_paper * std::f64::consts::TAU.powi(2)).abs();
  let spot_error = (spot_unit - spot_paper * std::f64::consts::TAU.powi(3)).abs();
  assert!(integrated_error < 1e-12, "error={integrated_error}");
  assert!(spot_error < 1e-12, "error={spot_error}");
}

#[test]
fn integrated_correction_uses_the_physical_period_multiplier() {
  for period in [1.0, std::f64::consts::TAU] {
    let engine = fixture(period);
    let raw = engine.integrated_volvol(Some(3));
    let quarticity = engine.integrated_quarticity(Some(3));
    let k_const = engine.compute_bias_correction_constant(3);
    let expected = raw - std::f64::consts::TAU.powi(2) / period * k_const * quarticity;
    let actual = engine.integrated_volvol_bias_corrected(Some(3));

    assert!((actual - expected).abs() < 1e-14);
  }
}

#[test]
fn outer_fejer_average_recovers_integrated_corrected_estimator() {
  let engine = fixture(1.0);
  let big_l = 2usize;
  let count = 2 * big_l + 1;
  let tau = (0..count)
    .map(|index| index as f64 / count as f64)
    .collect::<Array1<_>>();
  let spot = engine.spot_volvol_bias_corrected(tau.as_slice().unwrap(), Some(3), Some(big_l));
  let integrated = engine.integrated_volvol_bias_corrected(Some(3));
  let quadrature = spot.sum() * engine.period / count as f64;

  assert!((integrated - quadrature).abs() < 1e-13);
}
