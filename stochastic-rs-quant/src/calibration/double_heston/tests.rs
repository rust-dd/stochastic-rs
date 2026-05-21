use super::loss::double_heston_call_price;
use super::*;
use crate::LossMetric;
use crate::OptionType;
use crate::traits::Calibrator;

// Reference Heston (v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7)
// S=100, r=0.05, q=0, T=1.0. A double Heston with v1_0+v2_0 = 0.04 and
// factor-1 degenerate (v1_0 ≈ 0) collapses onto the single Heston answer.
const HESTON_REF: [f64; 9] = [
  25.095178, 20.976171, 17.106937, 13.548230, 10.361869, 7.604362, 5.317953, 3.519953, 2.193310,
];
const STRIKES: [f64; 9] = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];

#[test]
fn double_heston_reduces_to_heston_when_one_factor_vanishes() {
  // Factor 2 has v2_0=0 and theta2≈0 so its contribution is negligible.
  let p = DoubleHestonParams {
    v1_0: 0.04,
    kappa1: 1.5,
    theta1: 0.04,
    sigma1: 0.3,
    rho1: -0.7,
    v2_0: 1e-6,
    kappa2: 5.0,
    theta2: 1e-6,
    sigma2: 0.01,
    rho2: 0.0,
  };
  for (i, &k) in STRIKES.iter().enumerate() {
    let price = double_heston_call_price(&p, 100.0, k, 0.05, 0.0, 1.0);
    assert!(
      (price - HESTON_REF[i]).abs() < 0.25,
      "Double Heston (1 active factor) K={k}: got {price:.6}, expected {:.6}",
      HESTON_REF[i]
    );
  }
}

#[test]
fn double_heston_two_factors_produces_sensible_smile() {
  // Standard Christoffersen 2-factor split: one fast-mean-reverting factor,
  // one slow-mean-reverting factor. Prices should be positive, finite,
  // monotonic in strike, and satisfy basic bounds.
  let p = DoubleHestonParams {
    v1_0: 0.02,
    kappa1: 3.0,
    theta1: 0.02,
    sigma1: 0.4,
    rho1: -0.6,
    v2_0: 0.02,
    kappa2: 0.5,
    theta2: 0.03,
    sigma2: 0.2,
    rho2: -0.3,
  };
  let mut prev = f64::INFINITY;
  for &k in STRIKES.iter() {
    let price = double_heston_call_price(&p, 100.0, k, 0.05, 0.0, 1.0);
    // Intrinsic value lower bound: max(S e^{-qT} - K e^{-rT}, 0)
    let intrinsic = (100.0_f64 - k * (-0.05_f64 * 1.0).exp()).max(0.0);
    assert!(
      price.is_finite() && price >= intrinsic - 1e-6 && price <= 100.0 + 1e-6,
      "Double Heston K={k}: got {price:.6} outside [{intrinsic:.6}, 100]"
    );
    assert!(
      price < prev + 1e-6,
      "Double Heston prices should be monotone decreasing in strike: {prev:.6} → {price:.6} at K={k}"
    );
    prev = price;
  }
}

#[test]
fn double_heston_calibrate_to_heston_surface() {
  // Calibrate to Heston prices: we expect the RMSE to be small.
  let n = STRIKES.len();
  let calibrator = DoubleHestonCalibrator::new(
    Some(DoubleHestonParams {
      v1_0: 0.03,
      kappa1: 2.0,
      theta1: 0.03,
      sigma1: 0.25,
      rho1: -0.5,
      v2_0: 0.01,
      kappa2: 0.8,
      theta2: 0.02,
      sigma2: 0.15,
      rho2: -0.4,
    }),
    HESTON_REF.to_vec().into(),
    vec![100.0; n].into(),
    STRIKES.to_vec().into(),
    0.05,
    Some(0.0),
    1.0,
    OptionType::Call,
    false,
  );

  let result = calibrator.calibrate(None).unwrap();
  assert!(
    result.loss.get(LossMetric::Rmse) < 0.6,
    "Double Heston calibration RMSE={:.6}",
    result.loss.get(LossMetric::Rmse)
  );
  // Verify the calibrated price is close to the reference
  let p_out = result.params();
  for (i, &k) in STRIKES.iter().enumerate() {
    let price = double_heston_call_price(&p_out, 100.0, k, 0.05, 0.0, 1.0);
    assert!(
      (price - HESTON_REF[i]).abs() < 1.5,
      "Calibrated price K={k}: got {price:.4}, ref {:.4}",
      HESTON_REF[i]
    );
  }
}

#[test]
fn double_heston_params_to_model() {
  let p = DoubleHestonParams {
    v1_0: 0.02,
    kappa1: 3.0,
    theta1: 0.02,
    sigma1: 0.3,
    rho1: -0.6,
    v2_0: 0.02,
    kappa2: 0.5,
    theta2: 0.02,
    sigma2: 0.15,
    rho2: -0.3,
  };
  let model = p.to_model(0.03, 0.01);
  assert_eq!(model.v1_0, 0.02);
  assert_eq!(model.kappa2, 0.5);
  assert_eq!(model.r, 0.03);
  assert_eq!(model.q, 0.01);
}
