//! Two trainings with one seed on the CPU agree bit for bit; the init is the crate's, not candle's
//! unseedable RNG.

use ndarray::Array2;
use stochastic_rs_ai::Device;
use stochastic_rs_ai::volatility::common::TrainConfig;
use stochastic_rs_ai::volatility::heston;
use stochastic_rs_ai::volatility::heston::HestonNn;

fn data(rows: usize) -> (Array2<f32>, Array2<f32>) {
  let params = Array2::<f32>::from_shape_fn((rows, heston::INPUT_DIM), |(i, j)| {
    let u = ((i * 7 + j * 13) % 97) as f32 / 96.0;
    heston::PARAM_LB[j] + u * (heston::PARAM_UB[j] - heston::PARAM_LB[j])
  });
  let surfaces = Array2::<f32>::from_shape_fn((rows, heston::OUTPUT_DIM), |(i, k)| {
    0.2 + 0.1 * params[[i, 0]] + 0.02 * (k as f32 / heston::OUTPUT_DIM as f32)
  });
  (params, surfaces)
}

fn train(init_seed: u64) -> Vec<f32> {
  let (params, surfaces) = data(64);
  let mut model = HestonNn::new(&Device::Cpu, init_seed).unwrap();
  let cfg = TrainConfig {
    epochs: 3,
    random_seed: 11,
    ..TrainConfig::default()
  };
  model.train(&params, &surfaces, &cfg).unwrap();
  model
    .predict_surface(&[0.02, -0.5, 0.4, 0.05, 3.0])
    .unwrap()
}

#[test]
fn the_same_seed_gives_the_same_network_on_the_cpu() {
  let a = train(5);
  let b = train(5);
  let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
  assert_eq!(bits(&a), bits(&b));
}

#[test]
fn a_different_seed_gives_a_different_network() {
  let (a, b) = (train(5), train(6));
  assert!(a.iter().chain(&b).all(|v| v.is_finite()));
  assert_ne!(a, b);
}
