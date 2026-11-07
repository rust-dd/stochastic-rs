use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use stochastic_rs::distributions::exp::SimdExp;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::jump::kou::Kou;
use stochastic_rs::stochastic::jump::levy_diffusion::LevyDiffusion;
use stochastic_rs::stochastic::jump::merton::Merton;
use stochastic_rs::stochastic::jump::mjd_log::MjdLog;
use stochastic_rs::stochastic::volatility::bates_svj::BatesSvj;
use stochastic_rs::traits::ProcessExt;

const LAMBDA: f64 = 10.0;
const SIZES: [usize; 2] = [256, 4_096];

fn bench_jump_processes(c: &mut Criterion) {
  let mut group = c.benchmark_group("JumpProcesses");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(500));

  for &n in &SIZES {
    let merton = Merton::new(
      0.03,
      0.2,
      LAMBDA,
      0.0,
      SimdNormal::new(0.0, 0.1),
      n,
      Some(0.0),
      Some(1.0),
      Unseeded,
    );
    group.bench_with_input(BenchmarkId::new("merton", n), &n, |b, _| {
      b.iter(|| black_box(merton.sample()))
    });

    let kou = Kou::new(
      0.03,
      0.2,
      LAMBDA,
      0.0,
      SimdNormal::new(0.0, 0.12),
      n,
      Some(0.0),
      Some(1.0),
      Unseeded,
    );
    group.bench_with_input(BenchmarkId::new("kou", n), &n, |b, _| {
      b.iter(|| black_box(kou.sample()))
    });

    let kou_exp = Kou::new(
      0.03,
      0.2,
      LAMBDA,
      0.0,
      SimdExp::new(10.0),
      n,
      Some(0.0),
      Some(1.0),
      Unseeded,
    );
    group.bench_with_input(BenchmarkId::new("kou_exp", n), &n, |b, _| {
      b.iter(|| black_box(kou_exp.sample()))
    });

    let levy = LevyDiffusion::new(
      0.01,
      0.2,
      LAMBDA,
      SimdNormal::new(0.0, 0.08),
      n,
      Some(0.0),
      Some(1.0),
      Unseeded,
    );
    group.bench_with_input(BenchmarkId::new("levy_diffusion", n), &n, |b, _| {
      b.iter(|| black_box(levy.sample()))
    });

    let mjd = MjdLog::new(
      Some(0.05),
      None,
      None,
      None,
      0.2,
      LAMBDA,
      0.0,
      0.1,
      n,
      Some(100.0),
      Some(1.0),
      Unseeded,
    );
    group.bench_with_input(BenchmarkId::new("mjd_log", n), &n, |b, _| {
      b.iter(|| black_box(mjd.sample()))
    });

    let bates = BatesSvj::new(
      Some(0.05),
      None,
      None,
      None,
      LAMBDA,
      -0.1,
      0.2,
      0.04,
      1.5,
      0.3,
      -0.6,
      n,
      Some(100.0),
      Some(0.04),
      Some(1.0),
      Some(false),
      Unseeded,
    );
    group.bench_with_input(BenchmarkId::new("bates_svj", n), &n, |b, _| {
      b.iter(|| black_box(bates.sample()))
    });
  }

  group.finish();
}

criterion_group!(benches, bench_jump_processes);
criterion_main!(benches);
