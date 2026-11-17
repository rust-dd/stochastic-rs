//! The Euler kernels on Metal: `sample_par` of 100 000 `f32` paths over 252 points, one row per
//! family shape (a diffusion, a two-factor system, normal and exponential jumps, a live series).

#[cfg(metal_backend)]
mod gated {
  use std::hint::black_box;

  use criterion::BenchmarkGroup;
  use criterion::Criterion;
  use criterion::criterion_group;
  use criterion::measurement::WallTime;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::exp::SimdExp;
  use stochastic_rs_distributions::normal::SimdNormal;
  use stochastic_rs_stochastic::device::Metal;
  use stochastic_rs_stochastic::diffusion::gbm::Gbm;
  use stochastic_rs_stochastic::jump::kou::Kou;
  use stochastic_rs_stochastic::jump::merton::Merton;
  use stochastic_rs_stochastic::params::VarianceBoundary;
  use stochastic_rs_stochastic::traits::ProcessExt;
  use stochastic_rs_stochastic::volatility::HestonPow;
  use stochastic_rs_stochastic::volatility::heston::Heston;
  use stochastic_rs_stochastic::volatility::svcgmy::Svcgmy;

  const PATHS: usize = 100_000;

  const POINTS: usize = 252;

  fn row<P: ProcessExt<f32>>(group: &mut BenchmarkGroup<'_, WallTime>, name: &str, process: P) {
    black_box(process.sample_par(PATHS));
    group.bench_function(format!("{name}.sample_par/{PATHS}"), |b| {
      b.iter(|| black_box(process.sample_par(PATHS)))
    });
  }

  fn euler_metal(c: &mut Criterion) {
    let mut group = c.benchmark_group("EulerMetal");
    group.sample_size(20);
    let seed = || Deterministic::new(1);
    row(
      &mut group,
      "Gbm",
      Gbm::<f32, _>::new(0.05, 0.2, POINTS, 100.0, Some(1.0), seed()).on::<Metal>(),
    );
    row(
      &mut group,
      "Heston",
      Heston::<f32, _, _>::new(
        100.0,
        0.04,
        1.5,
        0.04,
        0.3,
        -0.7,
        0.05,
        POINTS,
        Some(1.0),
        HestonPow::Sqrt,
        VarianceBoundary::Floor,
        seed(),
      )
      .on::<Metal>(),
    );
    row(
      &mut group,
      "Merton",
      Merton::<f32, _, _>::new(
        0.05,
        0.2,
        3.0,
        -0.02,
        SimdNormal::<f32>::new(-0.02, 0.1),
        POINTS,
        Some(1.0),
        Some(1.0),
        seed(),
      )
      .on::<Metal>(),
    );
    row(
      &mut group,
      "Kou",
      Kou::<f32, _, _>::new(
        0.05,
        0.2,
        3.0,
        0.01,
        SimdExp::<f32>::new(8.0),
        POINTS,
        Some(1.0),
        Some(1.0),
        seed(),
      )
      .on::<Metal>(),
    );
    row(
      &mut group,
      "Svcgmy",
      Svcgmy::<f32, _>::new(
        2.0,
        6.0,
        0.5,
        2.0,
        0.04,
        0.2,
        0.3,
        POINTS,
        256,
        Some(0.0),
        0.04,
        Some(1.0),
        seed(),
      )
      .on::<Metal>(),
    );
    group.finish();
  }

  criterion_group!(benches, euler_metal);
}

#[cfg(metal_backend)]
criterion::criterion_main!(gated::benches);

#[cfg(not(metal_backend))]
fn main() {}
