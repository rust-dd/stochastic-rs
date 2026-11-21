use std::collections::HashMap;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use candle_core::Device;
use candle_core::Tensor;
use candle_nn::VarMap;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::uniform::SimdUniform;

/// candle-nn's `linear` laws from the crate's own streams: weights `N(0, 2/in)`, biases
/// `U(±1/√in)`, in name order.
pub(super) fn seed_weights(varmap: &VarMap, device: &Device, seed: u64) -> Result<()> {
  let mut layout = {
    let vars = varmap
      .data()
      .lock()
      .map_err(|_| anyhow!("the variable map is poisoned"))?;
    vars
      .iter()
      .map(|(name, var)| (name.clone(), var.dims().to_vec()))
      .collect::<Vec<(String, Vec<usize>)>>()
  };
  layout.sort_by(|a, b| a.0.cmp(&b.0));
  let weight_fan_in = layout
    .iter()
    .filter(|(name, dims)| name.ends_with(".weight") && dims.len() == 2)
    .map(|(name, dims)| (name.clone(), dims[1]))
    .collect::<HashMap<String, usize>>();

  let source = Deterministic::new(seed);
  let mut normal = SimdNormal::<f32>::new(0.0, 1.0).seeded(&source);
  let mut uniform = SimdUniform::<f32>::new(-1.0, 1.0).seeded(&source);
  let mut values = Vec::with_capacity(layout.len());
  for (name, dims) in &layout {
    let count = dims.iter().product::<usize>();
    let draws = if let Some(stem) = name.strip_suffix(".weight") {
      let fan_in = *weight_fan_in
        .get(name)
        .ok_or_else(|| anyhow!("{stem} is not a dense weight"))?;
      let std = (2.0 / fan_in as f32).sqrt();
      (0..count)
        .map(|_| std * normal.sample())
        .collect::<Vec<f32>>()
    } else if let Some(stem) = name.strip_suffix(".bias") {
      let fan_in = *weight_fan_in
        .get(&format!("{stem}.weight"))
        .ok_or_else(|| anyhow!("no weight beside {name}"))?;
      let bound = 1.0 / (fan_in as f32).sqrt();
      (0..count)
        .map(|_| bound * uniform.sample())
        .collect::<Vec<f32>>()
    } else {
      bail!("unexpected variable {name}");
    };
    values.push(Tensor::from_vec(draws, dims.clone(), device)?);
  }

  let vars = varmap
    .data()
    .lock()
    .map_err(|_| anyhow!("the variable map is poisoned"))?;
  for ((name, _), tensor) in layout.iter().zip(&values) {
    vars[name].set(tensor)?;
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use candle_core::DType;
  use candle_nn::VarBuilder;

  use super::super::network::FeedForwardNet;
  use super::*;

  const WIDTH: usize = 512;

  fn seeded(seed: u64) -> VarMap {
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    FeedForwardNet::new(vs, 5, WIDTH, 8).unwrap();
    seed_weights(&varmap, &Device::Cpu, seed).unwrap();
    varmap
  }

  fn read(varmap: &VarMap, name: &str) -> Vec<f32> {
    let vars = varmap.data().lock().unwrap();
    vars[name].flatten_all().unwrap().to_vec1::<f32>().unwrap()
  }

  #[test]
  fn the_weights_follow_the_kaiming_normal_and_the_biases_the_uniform_law() {
    let varmap = seeded(3);
    let weight = read(&varmap, "dense_2.weight");
    let n = weight.len() as f64;
    let mean = weight.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
    let var = weight
      .iter()
      .map(|v| (f64::from(*v) - mean).powi(2))
      .sum::<f64>()
      / n;
    let expected_std = (2.0 / WIDTH as f64).sqrt();
    assert!(mean.abs() < 0.01, "mean {mean}");
    assert!(
      (var.sqrt() - expected_std).abs() < 0.1 * expected_std,
      "std {}",
      var.sqrt()
    );

    let bias = read(&varmap, "dense_2.bias");
    let bound = 1.0 / (WIDTH as f32).sqrt();
    let (lo, hi) = bias
      .iter()
      .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    assert!(
      lo >= -bound && hi <= bound && hi - lo > bound,
      "range {lo}..{hi}"
    );
  }

  #[test]
  fn the_first_layer_takes_its_fan_in_from_the_input_width() {
    let weight = read(&seeded(3), "dense_1.weight");
    let bound = 1.0 / 5.0_f32.sqrt();
    assert!(weight.iter().any(|v| v.abs() > bound));
    let bias = read(&seeded(3), "dense_1.bias");
    let (lo, hi) = bias
      .iter()
      .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    assert!(
      lo >= -bound && hi <= bound && hi - lo > bound,
      "range {lo}..{hi}"
    );
  }

  #[test]
  fn a_seed_fixes_the_init_and_another_seed_moves_it() {
    let a = read(&seeded(7), "dense_3.weight");
    assert_eq!(a, read(&seeded(7), "dense_3.weight"));
    assert_ne!(a, read(&seeded(8), "dense_3.weight"));
  }
}
