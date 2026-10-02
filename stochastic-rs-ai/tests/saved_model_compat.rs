//! A model directory saved under candle 0.9.2 (`tests/data/heston_nn_candle_0_9_2`) must keep loading.

use std::path::Path;

use stochastic_rs_ai::Device;
use stochastic_rs_ai::volatility::heston::HestonNn;

#[test]
fn a_model_saved_with_candle_0_9_2_predicts_the_same_surface() {
  let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/heston_nn_candle_0_9_2");
  let model = HestonNn::load(&dir, &Device::Cpu).unwrap();
  let surface = model
    .predict_surface(&[0.02, -0.5, 0.4, 0.05, 3.0])
    .unwrap();
  let expected = std::fs::read_to_string(dir.join("expected_surface.txt"))
    .unwrap()
    .lines()
    .map(|line| line.parse().unwrap())
    .collect::<Vec<f32>>();
  assert_eq!(surface.len(), expected.len());
  let max_diff = surface
    .iter()
    .zip(&expected)
    .map(|(a, b)| (a - b).abs())
    .fold(0.0_f32, f32::max);
  assert!(max_diff < 1e-6, "max |diff| {max_diff}");
}
