use super::LevyCalibrator;
use super::loss::fourier_call_price;
use super::types::LevyCalibrationResult;
use super::types::LevyModel;
use super::types::LevyModelType;
use super::types::MarketSlice;
use crate::CalibrationLossScore;
use crate::LossMetric;
use crate::traits::Calibrator;

// Analytical reference prices (Gil-Pelaez inversion)
// S=100, r=0.05, q=0, T=1.0
const STRIKES: [f64; 9] = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];

// Vg: sigma=0.2, theta=-0.1, nu=0.5
const VG_REF: [f64; 9] = [
  25.056158, 20.941767, 17.091301, 13.572373, 10.453503, 7.795810, 5.640544, 3.991334, 2.793823,
];

// MJD: sigma=0.15, lam=1.0, muj=-0.05, sigj=0.1
const MJD_REF: [f64; 9] = [
  24.537096, 20.322713, 16.420092, 12.915553, 9.877019, 7.340385, 5.303578, 3.729825, 2.557790,
];

#[test]
fn vg_pricer_matches_reference() {
  // Verify our internal Fourier pricer reproduces reference Vg prices
  let params = [0.2, -0.1, 0.5]; // sigma, theta, nu
  for (i, &k) in STRIKES.iter().enumerate() {
    let price = fourier_call_price(
      LevyModelType::VarianceGamma,
      &params,
      100.0,
      k,
      0.05,
      0.0,
      1.0,
    );
    assert!(
      (price - VG_REF[i]).abs() < 0.05,
      "Vg K={k}: got {price:.6}, expected {:.6}",
      VG_REF[i]
    );
  }
}

#[test]
fn mjd_pricer_matches_reference() {
  // Verify our internal Fourier pricer reproduces reference MJD prices
  let params = [0.15, 1.0, -0.05, 0.1]; // sigma, lambda, mu_j, sigma_j
  for (i, &k) in STRIKES.iter().enumerate() {
    let price = fourier_call_price(LevyModelType::MertonJD, &params, 100.0, k, 0.05, 0.0, 1.0);
    assert!(
      (price - MJD_REF[i]).abs() < 0.05,
      "MJD K={k}: got {price:.6}, expected {:.6}",
      MJD_REF[i]
    );
  }
}

#[test]
fn vg_calibrate_recovers_reference_prices() {
  let market = MarketSlice {
    strikes: STRIKES.to_vec(),
    prices: VG_REF.to_vec(),
    is_call: vec![true; 9],
    t: 1.0,
  };

  let calibrator =
    LevyCalibrator::new(LevyModelType::VarianceGamma, 100.0, 0.05, 0.0, vec![market]);

  let result = calibrator.calibrate(None).unwrap();
  assert!(
    result.loss.get(LossMetric::Rmse) < 0.1,
    "Vg RMSE={:.6}",
    result.loss.get(LossMetric::Rmse)
  );
  println!("Vg recovered params: {:?}", result.params);
}

#[test]
fn mjd_calibrate_recovers_reference_prices() {
  let market = MarketSlice {
    strikes: STRIKES.to_vec(),
    prices: MJD_REF.to_vec(),
    is_call: vec![true; 9],
    t: 1.0,
  };

  let calibrator = LevyCalibrator::new(LevyModelType::MertonJD, 100.0, 0.05, 0.0, vec![market]);

  let result = calibrator.calibrate(None).unwrap();
  assert!(
    result.loss.get(LossMetric::Rmse) < 0.1,
    "MJD RMSE={:.6}",
    result.loss.get(LossMetric::Rmse)
  );
  println!("MJD recovered params: {:?}", result.params);
}

#[test]
fn test_levy_vg_calibrate() {
  let market = MarketSlice {
    strikes: vec![90.0, 95.0, 100.0, 105.0, 110.0],
    prices: vec![12.5, 9.0, 6.2, 4.0, 2.3],
    is_call: vec![true, true, true, true, true],
    t: 0.5,
  };

  let calibrator = LevyCalibrator::new(
    LevyModelType::VarianceGamma,
    100.0,
    0.03,
    0.01,
    vec![market],
  );

  let result = calibrator.calibrate(None).unwrap();
  println!("Vg params: {:?}, loss: {:?}", result.params, result.loss);
}

/// Regression: `LevyCalibrationResult::to_model` for `LevyModelType::Nig`
/// must produce a model whose `price_call` matches the calibrated NIG
/// dynamics. The rc.0 implementation wrapped the NIG triple `(α, β, δ)` into
/// a `CGMYFourier` with hardcoded `y = 0.5`, producing prices unrelated to
/// the NIG ChF. After the fix, `to_model` builds a `NigFourier` and the
/// round-trip price agrees with the calibrator's internal `fourier_call_price`.
#[test]
fn nig_to_model_matches_calibrator_pricer() {
  // Pick a representative NIG triple from the literature (Schoutens 2003 §5.3
  // SPX-style: α=10, β=-5, δ=0.5, mildly skewed/heavy-tailed).
  let params = vec![10.0, -5.0, 0.5];
  let s = 100.0;
  let r = 0.05;
  let q = 0.0;
  let t = 1.0;

  // Build a synthetic LevyCalibrationResult with the known params and
  // round-trip to a model.
  let result = LevyCalibrationResult {
    params: params.clone(),
    model_type: LevyModelType::Nig,
    iterations: 0,
    converged: true,
    loss: CalibrationLossScore::default(),
  };
  let model = result.to_model(r, q);

  for &k in &STRIKES {
    let from_calibrator = fourier_call_price(LevyModelType::Nig, &params, s, k, r, q, t);
    let from_model = <LevyModel as crate::traits::ModelPricer>::price_call(&model, s, k, r, q, t);
    assert!(
      (from_model - from_calibrator).abs() < 1e-6,
      "NIG to_model mismatch at K={k}: model={from_model:.10} calibrator={from_calibrator:.10}",
    );
  }
}

#[test]
fn test_levy_merton_calibrate() {
  let market = MarketSlice {
    strikes: vec![90.0, 95.0, 100.0, 105.0, 110.0],
    prices: vec![12.5, 9.0, 6.2, 4.0, 2.3],
    is_call: vec![true, true, true, true, true],
    t: 0.5,
  };

  let calibrator = LevyCalibrator::new(LevyModelType::MertonJD, 100.0, 0.03, 0.01, vec![market]);

  let result = calibrator.calibrate(None).unwrap();
  println!(
    "Merton params: {:?}, loss: {:?}",
    result.params, result.loss
  );
}
