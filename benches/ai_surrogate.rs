use std::hint::black_box;

use criterion::BatchSize;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use ndarray::Array2;
use stochastic_rs::ai::Device;
use stochastic_rs::ai::volatility::common::StochVolModelSpec;
use stochastic_rs::ai::volatility::common::StochVolNn;
use stochastic_rs::ai::volatility::common::TrainConfig;
use stochastic_rs::ai::volatility::heston;

fn spec() -> StochVolModelSpec {
  StochVolModelSpec::new(
    heston::MODEL_ID,
    heston::INPUT_DIM,
    heston::OUTPUT_DIM,
    heston::DEFAULT_HIDDEN_DIM,
    heston::PARAM_LB.to_vec(),
    heston::PARAM_UB.to_vec(),
  )
  .unwrap()
}

fn dataset(rows: usize) -> (Array2<f32>, Array2<f32>) {
  let params = Array2::<f32>::from_shape_fn((rows, heston::INPUT_DIM), |(i, j)| {
    let u = ((i * 7 + j * 13) % 97) as f32 / 96.0;
    heston::PARAM_LB[j] + u * (heston::PARAM_UB[j] - heston::PARAM_LB[j])
  });
  let surfaces = Array2::<f32>::from_shape_fn((rows, heston::OUTPUT_DIM), |(i, k)| {
    0.2 + 0.1 * params[[i, 0]] + 0.05 * params[[i, 2]] * (k as f32 * 0.11).sin()
  });
  (params, surfaces)
}

fn bench_ai_surrogate(c: &mut Criterion) {
  let device = Device::Cpu;
  let (params, surfaces) = dataset(2_048);
  let cfg = TrainConfig {
    epochs: 1,
    batch_size: 64,
    ..TrainConfig::default()
  };
  let mut trained = StochVolNn::new(spec(), &device).unwrap();
  trained.train(&params, &surfaces, &cfg).unwrap();
  let theta = (0..heston::INPUT_DIM)
    .map(|j| 0.5 * (heston::PARAM_LB[j] + heston::PARAM_UB[j]))
    .collect::<Vec<f32>>();
  let batch = dataset(1_024).0;

  let mut group = c.benchmark_group("ai_surrogate");
  group.bench_function("train_epoch/rows_2048", |b| {
    b.iter_batched_ref(
      || StochVolNn::new(spec(), &device).unwrap(),
      |nn| black_box(nn.train(&params, &surfaces, &cfg).unwrap()),
      BatchSize::PerIteration,
    )
  });
  group.bench_function("jacobian", |b| {
    b.iter(|| black_box(trained.predict_surface_with_jacobian(&theta).unwrap()))
  });
  group.bench_function("predict_surfaces/rows_1024", |b| {
    b.iter(|| black_box(trained.predict_surfaces(&batch).unwrap()))
  });
  group.finish();
}

criterion_group!(benches, bench_ai_surrogate);
criterion_main!(benches);
