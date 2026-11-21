use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use stochastic_rs::distributions::exp::SimdExp;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::jump::bates::Bates;
use stochastic_rs::stochastic::jump::bates::LogNormalJump;
use stochastic_rs::stochastic::jump::kou::Kou;
use stochastic_rs::stochastic::jump::levy_diffusion::LevyDiffusion;
use stochastic_rs::stochastic::jump::merton::Merton;
use stochastic_rs::stochastic::jump::mjd_log::MjdLog;
use stochastic_rs::stochastic::params::Drift;
use stochastic_rs::stochastic::params::VarianceBoundary;
use stochastic_rs::stochastic::process::subordinator::ctrw::Ctrw;
use stochastic_rs::stochastic::process::subordinator::ctrw::CtrwJumpLaw;
use stochastic_rs::stochastic::process::subordinator::ctrw::CtrwWaitingLaw;
use stochastic_rs::stochastic::volatility::bns::Bns;
use stochastic_rs::traits::ProcessExt;

const LAMBDA: f64 = 10.0;
const SIZES: [usize; 2] = [256, 4_096];

fn bench_jump_processes(c: &mut Criterion) {
  let mut group = c.benchmark_group("JumpProcesses");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(500));

  let bns = Bns::<f64>::new(100.0, 0.04, 1.0, 0.05, 2.0, 1.0, 1024, Some(1.0), Unseeded);
  group.bench_function("bns", |b| b.iter(|| black_box(bns.sample())));

  let ctrw = Ctrw::<f64>::new(
    CtrwWaitingLaw::Exponential { rate: 50.0 },
    CtrwJumpLaw::Normal {
      mean: 0.0,
      std: 0.1,
    },
    1024,
    None,
    Some(1.0),
    Unseeded,
  );
  group.bench_function("ctrw", |b| b.iter(|| black_box(ctrw.sample())));

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
      Drift::Mu(0.05),
      0.2,
      LAMBDA,
      0.0,
      0.1,
      n,
      100.0,
      Some(1.0),
      Unseeded,
    );
    group.bench_with_input(BenchmarkId::new("mjd_log", n), &n, |b, _| {
      b.iter(|| black_box(mjd.sample()))
    });

    let law = LogNormalJump::new(-0.1, 0.2);
    let bates = Bates::new(
      Drift::Mu(0.05),
      LAMBDA,
      law.mean(),
      0.04,
      1.5,
      0.3,
      -0.6,
      law,
      n,
      100.0,
      0.04,
      Some(1.0),
      VarianceBoundary::Floor,
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
