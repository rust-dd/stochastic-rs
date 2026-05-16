use std::hint::black_box;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::rough::rl_bs::RlBlackScholes;
use stochastic_rs::stochastic::rough::rl_fbm::RlFBm;
use stochastic_rs::stochastic::rough::rl_fou::RlFOU;
use stochastic_rs::stochastic::rough::rl_heston::RlHeston;
use stochastic_rs::traits::ProcessExt;

fn bench_rl_fbm_by_size(c: &mut Criterion) {
  let mut group = c.benchmark_group("RL_fBM_by_size");
  let hurst = 0.1f64;

  for &n in &[256_usize, 1024, 4096] {
    group.bench_with_input(BenchmarkId::new("bilokon_wong", n), &n, |b, &n| {
      let p = RlFBm::new(hurst, n, Some(1.0), None, Unseeded);
      b.iter(|| black_box(p.sample()));
    });
  }

  group.finish();
}

fn bench_rl_fbm_by_hurst(c: &mut Criterion) {
  let mut group = c.benchmark_group("RL_fBM_by_hurst");
  let n = 1024_usize;

  for &h in &[0.05_f64, 0.1, 0.2, 0.3, 0.45] {
    let label = format!("H={h:.2}");
    group.bench_with_input(BenchmarkId::new("bilokon_wong", &label), &h, |b, &h| {
      let p = RlFBm::new(h, n, Some(1.0), None, Unseeded);
      b.iter(|| black_box(p.sample()));
    });
  }

  group.finish();
}

fn bench_rl_fou_and_bs(c: &mut Criterion) {
  let n = 1024_usize;

  c.bench_function("RL_FOU/H=0.1", |b| {
    let p = RlFOU::new(0.1_f64, 2.0, 0.0, 0.25, n, Some(0.0), Some(1.0), None, Unseeded);
    b.iter(|| black_box(p.sample()));
  });

  c.bench_function("RL_BlackScholes/H=0.3", |b| {
    let p = RlBlackScholes::new(0.3_f64, 100.0, 0.05, 0.2, n, Some(0.5), None, Unseeded);
    b.iter(|| black_box(p.sample()));
  });
}

fn bench_rl_heston(c: &mut Criterion) {
  let mut group = c.benchmark_group("RL_Heston");

  for &n in &[256_usize, 512, 1024] {
    group.bench_with_input(BenchmarkId::new("H=0.12", n), &n, |b, &n| {
      let p = RlHeston::new(
        0.12_f64,
        Some(100.0),
        Some(0.04),
        0.1,
        0.3156,
        0.0331,
        -0.681,
        0.0,
        n,
        Some(1.0),
        None,
        Unseeded,
      );
      b.iter(|| black_box(p.sample()));
    });
  }

  group.finish();
}

/// Compares three ways of generating m paths:
/// - sample()×m: m sequential single-path calls
/// - sample_par(m): rayon-parallel (thread-per-path)
/// - sample_batch(m): path-parallel SIMD (new, matches Python RoughHestonFast)
fn bench_rl_fbm_throughput(c: &mut Criterion) {
  use criterion::Throughput;
  let mut group = c.benchmark_group("RL_fBM_throughput");
  let n = 1024_usize;
  let hurst = 0.1_f64;

  for &m in &[16_usize, 64, 256, 1024] {
    group.throughput(Throughput::Elements(m as u64));

    group.bench_with_input(BenchmarkId::new("sample_loop", m), &m, |b, &m| {
      let p = RlFBm::new(hurst, n, Some(1.0), None, Unseeded);
      b.iter(|| {
        for _ in 0..m {
          black_box(p.sample());
        }
      });
    });

    group.bench_with_input(BenchmarkId::new("sample_par", m), &m, |b, &m| {
      let p = RlFBm::new(hurst, n, Some(1.0), None, Unseeded);
      b.iter(|| black_box(p.sample_par(m)));
    });

    group.bench_with_input(BenchmarkId::new("sample_batch", m), &m, |b, &m| {
      let p = RlFBm::new(hurst, n, Some(1.0), None, Unseeded);
      b.iter(|| black_box(p.sample_batch(m)));
    });

    group.bench_with_input(BenchmarkId::new("sample_batch_par", m), &m, |b, &m| {
      let p = RlFBm::new(hurst, n, Some(1.0), None, Unseeded);
      b.iter(|| black_box(p.sample_batch_par(m)));
    });
  }

  group.finish();
}

fn bench_rl_heston_throughput(c: &mut Criterion) {
  use criterion::Throughput;
  let mut group = c.benchmark_group("RL_Heston_throughput");
  let n = 256_usize;

  for &m in &[16_usize, 256, 1024] {
    group.throughput(Throughput::Elements(m as u64));

    group.bench_with_input(BenchmarkId::new("sample_par", m), &m, |b, &m| {
      let p = RlHeston::new(
        0.12_f64,
        Some(100.0),
        Some(0.04),
        0.1,
        0.3156,
        0.0331,
        -0.681,
        0.0,
        n,
        Some(1.0),
        None,
        Unseeded,
      );
      b.iter(|| black_box(p.sample_par(m)));
    });

    group.bench_with_input(BenchmarkId::new("sample_batch", m), &m, |b, &m| {
      let p = RlHeston::new(
        0.12_f64,
        Some(100.0),
        Some(0.04),
        0.1,
        0.3156,
        0.0331,
        -0.681,
        0.0,
        n,
        Some(1.0),
        None,
        Unseeded,
      );
      b.iter(|| black_box(p.sample_batch(m)));
    });
  }

  group.finish();
}

criterion_group!(
  benches,
  bench_rl_fbm_by_size,
  bench_rl_fbm_by_hurst,
  bench_rl_fou_and_bs,
  bench_rl_heston,
  bench_rl_fbm_throughput,
  bench_rl_heston_throughput
);
criterion_main!(benches);
