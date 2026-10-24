use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkGroup;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::measurement::WallTime;
use ndarray::Array1;
use stochastic_rs::stochastic::mc::antithetic;

const PATHS: usize = 50_000;

/// One normal and a `max`: the cheapest payoff there is, so the cost of the
/// estimator's own accumulation shows.
fn payoff_1d(z: &Array1<f64>) -> f64 {
  z[0].max(0.0)
}

/// Call on the terminal value of a log-Euler geometric Brownian motion driven
/// by the 64 normals.
fn payoff_64_steps(z: &Array1<f64>) -> f64 {
  let dt = 1.0 / z.len() as f64;
  let drift = (0.05 - 0.5 * 0.2 * 0.2) * dt;
  let vol = 0.2 * dt.sqrt();
  let terminal = z.iter().fold(100.0, |s, &zj| s * (drift + vol * zj).exp());
  (terminal - 100.0).max(0.0)
}

fn bench_payoff<F>(group: &mut BenchmarkGroup<'_, WallTime>, name: &str, dim: usize, payoff: F)
where
  F: Fn(&Array1<f64>) -> f64 + Sync + Copy,
{
  group.bench_function(BenchmarkId::new("estimate", name), |b| {
    b.iter(|| black_box(antithetic::estimate(PATHS, dim, payoff)));
  });
  group.bench_function(BenchmarkId::new("estimate_par", name), |b| {
    b.iter(|| black_box(antithetic::estimate_par(PATHS, dim, payoff)));
  });
}

fn bench_antithetic(c: &mut Criterion) {
  let mut group = c.benchmark_group("mc_antithetic");
  group.throughput(Throughput::Elements(PATHS as u64));
  group
    .sample_size(20)
    .measurement_time(Duration::from_secs(4));
  bench_payoff(&mut group, "payoff_1d", 1, payoff_1d);
  bench_payoff(&mut group, "payoff_64_steps", 64, payoff_64_steps);
  group.finish();
}

criterion_group!(benches, bench_antithetic);
criterion_main!(benches);
