// docs: ai#calibrating-with-a-surrogate
//! Backs the surrogate-calibration example on the AI page.
#![cfg(feature = "ai")]

use ndarray::Array2;
use stochastic_rs::ai::Device;
use stochastic_rs::ai::calibration::HestonSurrogateCalibrator;
use stochastic_rs::ai::volatility::common::TrainConfig;
use stochastic_rs::ai::volatility::heston::HestonNn;
use stochastic_rs::ai::volatility::heston::INPUT_DIM;
use stochastic_rs::ai::volatility::heston::OUTPUT_DIM;
use stochastic_rs::ai::volatility::heston::PARAM_LB;
use stochastic_rs::ai::volatility::heston::PARAM_UB;
use stochastic_rs::quant::calibration::heston::HestonParams;
use stochastic_rs::traits::CalibrationResult;
use stochastic_rs::traits::Calibrator;
use stochastic_rs::traits::ModelPricer;
use stochastic_rs::traits::ToModel;

/// A smooth stand-in for a Heston surface generator, so the example trains
/// in seconds; a production surrogate is trained on pricer output instead.
fn synthetic_surface(params: &[f32]) -> Vec<f32> {
  (0..OUTPUT_DIM)
    .map(|k| {
      let mut v = 0.2 + 0.03 * k as f32 / OUTPUT_DIM as f32;
      for (j, &p) in params.iter().enumerate() {
        let x = (p - 0.5 * (PARAM_LB[j] + PARAM_UB[j])) / (0.5 * (PARAM_UB[j] - PARAM_LB[j]));
        v +=
          (0.08 + 0.02 * (j + 1) as f32) * x * ((k as f32 + 1.0) * (j as f32 + 1.0) * 0.11).sin();
      }
      v
    })
    .collect()
}

#[test]
fn calibrate_heston_on_a_surrogate() {
  // 1. Train a small surrogate on (parameters → surface) pairs.
  let rows = 256;
  let mut params = Array2::<f32>::zeros((rows, INPUT_DIM));
  let mut surfaces = Array2::<f32>::zeros((rows, OUTPUT_DIM));
  for i in 0..rows {
    let theta: Vec<f32> = (0..INPUT_DIM)
      .map(|j| {
        let u = ((i * 7 + j * 13) % 97) as f32 / 96.0;
        PARAM_LB[j] + u * (PARAM_UB[j] - PARAM_LB[j])
      })
      .collect();
    params
      .row_mut(i)
      .assign(&ndarray::Array1::from_vec(theta.clone()));
    surfaces
      .row_mut(i)
      .assign(&ndarray::Array1::from_vec(synthetic_surface(&theta)));
  }
  let mut model = HestonNn::new(&Device::Cpu).unwrap();
  let cfg = TrainConfig {
    epochs: 30,
    ..TrainConfig::default()
  };
  model.train(&params, &surfaces, &cfg).unwrap();

  // 2. Calibrate to a market surface: LM on the network's exact Jacobian.
  let market: Vec<f64> = synthetic_surface(&[0.02, -0.6, 0.4, 0.05, 3.0])
    .iter()
    .map(|&v| v as f64)
    .collect();
  let calibrator = HestonSurrogateCalibrator::new(&model, market).unwrap();
  let result = calibrator.calibrate(None).unwrap();
  assert!(result.converged(), "{}", result.message().unwrap_or(""));
  assert!(result.fit.in_bounds);
  assert!(result.rmse() < 0.05, "rmse {}", result.rmse());
  let fitted: HestonParams = result.params();
  assert!(fitted.kappa > 0.0 && fitted.rho < 0.0);

  // 3. The result is a pricer: price a vanilla off the calibrated parameters.
  let pricer = result.to_model(0.01, 0.0);
  let call = pricer.price_call(100.0, 100.0, 0.01, 0.0, 1.0);
  assert!(call > 0.0 && call < 100.0);
}
