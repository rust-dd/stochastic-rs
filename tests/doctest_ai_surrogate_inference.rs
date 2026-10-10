// docs: ai#example--heston-surrogate-inference
//! Backs the surrogate-inference example on the AI page.
#![cfg(feature = "ai")]

use ndarray::Array2;
use stochastic_rs::ai::Device;
use stochastic_rs::ai::volatility::common::TrainConfig;
use stochastic_rs::ai::volatility::grid::MATURITIES;
use stochastic_rs::ai::volatility::heston;
use stochastic_rs::ai::volatility::heston::HestonNn;

#[test]
fn predict_a_heston_surface_on_the_training_grid() {
  // A throwaway network on a smooth synthetic target: no pretrained weights ship, and
  // the repository's training sets are too large for an example.
  let rows = 64;
  let params = Array2::<f32>::from_shape_fn((rows, heston::INPUT_DIM), |(i, j)| {
    let u = ((i * 7 + j * 13) % 97) as f32 / 96.0;
    heston::PARAM_LB[j] + u * (heston::PARAM_UB[j] - heston::PARAM_LB[j])
  });
  let surfaces = Array2::<f32>::from_shape_fn((rows, heston::OUTPUT_DIM), |(i, k)| {
    0.2 + 0.1 * params[[i, 0]] + 0.02 * (k as f32 / heston::OUTPUT_DIM as f32)
  });
  let device = Device::Cpu;
  let mut model = HestonNn::new(&device).unwrap();
  let cfg = TrainConfig {
    epochs: 2,
    ..TrainConfig::default()
  };
  model.train(&params, &surfaces, &cfg).unwrap();

  // Save and reload the way a trained model travels: a directory, a device.
  let dir = std::env::temp_dir().join(format!(
    "stochastic_rs_heston_nn_doc_{}",
    std::process::id()
  ));
  model.save(&dir).unwrap();
  let nn = HestonNn::load(&dir, &device).unwrap();
  std::fs::remove_dir_all(&dir).unwrap();

  // The surface is flat, maturity-major, on the training grid at zero rates:
  // scale the relative strikes by the spot and give the forward at every maturity.
  let spot = 100.0;
  let strikes = heston::STRIKES
    .iter()
    .map(|k| k * spot)
    .collect::<Vec<f64>>();
  let forwards = vec![spot; MATURITIES.len()];
  let theta = [0.03_f32, -0.5, 0.3, 0.04, 2.0];
  let surface = nn
    .predict_implied_vol_surface(&theta, strikes, MATURITIES.to_vec(), forwards)
    .unwrap();

  assert_eq!(surface.ivs.dim(), (MATURITIES.len(), heston::STRIKES.len()));
  assert!(surface.strikes.windows(2).all(|w| w[0] < w[1]));
  let smile = surface.smile_slice(4);
  assert_eq!(smile.tau, MATURITIES[4]);
  assert_eq!(smile.implied_vols.len(), heston::STRIKES.len());
}
