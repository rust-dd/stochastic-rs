//! Calibration regression tests and the projection-box pins.

use super::*;
use crate::traits::Calibrator;

#[test]
fn test_calibrate() {
  let s = vec![
    425.73, 425.73, 425.73, 425.67, 425.68, 425.65, 425.65, 425.68, 425.65, 425.16, 424.78, 425.19,
  ];

  let k = vec![
    395.0, 400.0, 405.0, 410.0, 415.0, 420.0, 425.0, 430.0, 435.0, 440.0, 445.0, 450.0,
  ];

  let c_market = vec![
    30.75, 25.88, 21.00, 16.50, 11.88, 7.69, 4.44, 2.10, 0.78, 0.25, 0.10, 0.10,
  ];

  let r = 0.05;
  let r_d = None;
  let r_f = None;
  let q = None;
  let tau = 1.0;
  let option_type = OptionType::Call;

  let calibrator = BSMCalibrator::new(
    BSMParams { v: 0.2 },
    c_market.into(),
    s.into(),
    k.into(),
    r,
    r_d,
    r_f,
    q,
    tau,
    option_type,
  );

  calibrator.calibrate(None).unwrap();
}

#[test]
fn test_calibrate_from_slices_recovers_constant_sigma() {
  // Generate three synthetic maturity slices from a known constant sigma,
  // then check the joint calibrator recovers it on the whole flattened set.
  use crate::calibration::levy::MarketSlice;

  let s = 100.0_f64;
  let r = 0.03_f64;
  let true_sigma = 0.25_f64;
  let strikes = vec![85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0];

  let make_slice = |tau: f64| -> MarketSlice {
    let prices: Vec<f64> = strikes
      .iter()
      .map(|&k| BSMPricer::new(true_sigma, BSMCoc::Bsm1973).price_call(s, k, r, 0.0, tau))
      .collect();
    MarketSlice {
      strikes: strikes.clone(),
      prices,
      is_call: vec![true; strikes.len()],
      tau,
    }
  };

  let slices = vec![make_slice(0.10), make_slice(0.30), make_slice(0.75)];

  let calibrator = BSMCalibrator::from_slices(
    BSMParams { v: 0.4 }, // intentionally far from the truth
    &slices,
    s,
    r,
    None,
    None,
    None,
    OptionType::Call,
  );
  let result = calibrator.calibrate(None).unwrap();
  println!(
    "recovered sigma = {:.6}  (truth {:.4})  converged = {}",
    result.v, true_sigma, result.converged
  );
  assert!(
    (result.v - true_sigma).abs() < 1e-3,
    "expected ~{}, got {}",
    true_sigma,
    result.v
  );
}

#[test]
fn calibrator_trait_returns_result() {
  use crate::traits::CalibrationResult;
  let s = 100.0_f64;
  let k = 100.0_f64;
  let true_sigma = 0.20_f64;
  let call = BSMPricer::new(true_sigma, BSMCoc::Bsm1973).price_call(s, k, 0.03, 0.0, 0.5);

  let calibrator = BSMCalibrator::new(
    BSMParams { v: 0.4 },
    DVector::from_vec(vec![call]),
    DVector::from_vec(vec![s]),
    DVector::from_vec(vec![k]),
    0.03,
    None,
    None,
    None,
    0.5,
    OptionType::Call,
  );

  let result: Result<BSMCalibrationResult, anyhow::Error> =
    Calibrator::calibrate(&calibrator, None);
  let result = result.expect("trait calibrate must succeed");
  let params = CalibrationResult::params(&result);
  assert!(
    (params.v - true_sigma).abs() < 1e-3,
    "trait Calibrator path recovered sigma {} via CalibrationResult::params (expected ~{})",
    params.v,
    true_sigma
  );
}

/// Quotes generated from `true_sigma`, so the only right answer is known.
fn synthetic(initial: f64) -> BSMCalibrator {
  let (s, r, tau) = (100.0_f64, 0.03_f64, 0.5_f64);
  let strikes = [90.0, 95.0, 100.0, 105.0, 110.0];
  let prices: Vec<f64> = strikes
    .iter()
    .map(|&k| BSMPricer::new(TRUE_SIGMA, BSMCoc::Bsm1973).price_call(s, k, r, 0.0, tau))
    .collect();
  BSMCalibrator::new(
    BSMParams { v: initial },
    DVector::from_vec(prices),
    DVector::from_vec(vec![s; strikes.len()]),
    DVector::from_vec(strikes.to_vec()),
    r,
    None,
    None,
    None,
    tau,
    OptionType::Call,
  )
}

const TRUE_SIGMA: f64 = 0.25;

/// The box on the write path, and the number it prevents. A negative `v`
/// does not announce itself — `d1` and `d2` both flip sign, so the call
/// comes back finite, here *negative* — so the second assertion is what
/// makes this a test of the projection rather than of `abs`.
#[test]
fn sigma_is_reflected_not_clamped() {
  let mut cal = synthetic(0.2);
  LeastSquaresProblem::set_params(&mut cal, &DVector::from_vec(vec![-0.3]));
  assert_eq!(cal.params.v, 0.3, "a negative step must reflect, not clamp");

  let bsm = |v: f64| BSMPricer::new(v, BSMCoc::Bsm1973).price_call(100.0, 100.0, 0.03, 0.0, 0.5);
  assert!(
    bsm(-0.3).is_finite() && (bsm(0.3) - bsm(-0.3)).abs() > 1.0,
    "an unprojected -0.3 prices {} against {} at +0.3",
    bsm(-0.3),
    bsm(0.3)
  );

  LeastSquaresProblem::set_params(&mut cal, &DVector::from_vec(vec![f64::NAN]));
  assert_eq!(cal.params.v, V_MIN, "a projection onto a set must be total");

  cal.set_initial_guess(BSMParams { v: -0.4 });
  assert_eq!(
    cal.params.v, 0.4,
    "the starting point is stored inside the box"
  );
}

/// The read path, which `set_params` does not cover: the optimiser prices
/// the starting point before it has any step to hand back, so a `pub
/// params` written directly reaches [`BSMPricer::new`] unprojected.
#[test]
fn the_optimisers_first_evaluation_is_already_inside_the_box() {
  assert_eq!(
    LeastSquaresProblem::residuals(&synthetic(-0.4)),
    LeastSquaresProblem::residuals(&synthetic(0.4)),
    "the first residual evaluation must not see the raw sign"
  );
}

/// The box must not stall a live calibration: started outside it, by
/// either route in, the fit still lands on the truth.
#[test]
fn a_negative_starting_point_still_recovers_sigma() {
  for (label, result) in [
    (
      "initial guess",
      synthetic(0.2)
        .calibrate(Some(BSMParams { v: -0.4 }))
        .unwrap(),
    ),
    ("pub params field", synthetic(-0.4).calibrate(None).unwrap()),
  ] {
    assert!(
      (result.v - TRUE_SIGMA).abs() < 1e-3,
      "{label}: recovered {} against {TRUE_SIGMA}",
      result.v
    );
  }
}
