use criterion::black_box;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
use stochastic_rs::stochastic::noise::fgn::FGN;
use stochastic_rs::stochastic::process::fbm::FBM;
use stochastic_rs::traits::ProcessExt;

fn bench_fgn_by_size(c: &mut Criterion) {
  let mut group = c.benchmark_group("FGN_by_size");
  let hurst = 0.7f64;

  for &n in &[256, 1024, 4096, 16384, 65536] {
    group.bench_with_input(BenchmarkId::new("ndrustfft", n), &n, |b, &n| {
      let fgn = FGN::new(hurst, n, None);
      b.iter(|| black_box(fgn.sample()));
    });
  }

  group.finish();
}

fn bench_fgn_by_hurst(c: &mut Criterion) {
  let mut group = c.benchmark_group("FGN_by_hurst");
  let n = 4096usize;

  for &h in &[0.1, 0.3, 0.5, 0.7, 0.9] {
    let label = format!("H={:.1}", h);

    group.bench_with_input(BenchmarkId::new("ndrustfft", &label), &h, |b, &h| {
      let fgn = FGN::new(h, n, None);
      b.iter(|| black_box(fgn.sample()));
    });
  }

  group.finish();
}

fn bench_fgn_f32_vs_f64(c: &mut Criterion) {
  let mut group = c.benchmark_group("FGN_f32_vs_f64");
  let n = 4096usize;
  let hurst_f64 = 0.7f64;
  let hurst_f32 = 0.7f32;

  group.bench_function("f64/ndrustfft", |b| {
    let fgn = FGN::new(hurst_f64, n, None);
    b.iter(|| black_box(fgn.sample()));
  });

  group.bench_function("f32/ndrustfft", |b| {
    let fgn = FGN::new(hurst_f32, n, None);
    b.iter(|| black_box(fgn.sample()));
  });

  group.finish();
}

fn bench_fgn_sample_par(c: &mut Criterion) {
  let mut group = c.benchmark_group("FGN_sample_par");
  let hurst = 0.7f64;
  let n = 4096usize;

  for &m in &[10, 100, 1000] {
    group.bench_with_input(BenchmarkId::new("sample_par", m), &m, |b, &m| {
      let fgn = FGN::new(hurst, n, None);
      b.iter(|| black_box(fgn.sample_par(m)));
    });

    group.bench_with_input(BenchmarkId::new("sample_sequential", m), &m, |b, &m| {
      let fgn = FGN::new(hurst, n, None);
      b.iter(|| {
        let v: Vec<_> = (0..m).map(|_| fgn.sample()).collect();
        black_box(v)
      });
    });
  }

  group.finish();
}

fn bench_fbm_by_size(c: &mut Criterion) {
  let mut group = c.benchmark_group("FBM_by_size");
  let hurst = 0.7f64;

  for &n in &[256, 1024, 4096, 16384, 65536] {
    group.bench_with_input(BenchmarkId::new("sample", n), &n, |b, &n| {
      let fbm = FBM::new(hurst, n, None);
      b.iter(|| black_box(fbm.sample()));
    });
  }

  group.finish();
}

fn bench_fbm_sample_par(c: &mut Criterion) {
  let mut group = c.benchmark_group("FBM_sample_par");
  let hurst = 0.7f64;
  let n = 4096usize;

  for &m in &[10, 100, 1000] {
    group.bench_with_input(BenchmarkId::new("sample_par", m), &m, |b, &m| {
      let fbm = FBM::new(hurst, n, None);
      b.iter(|| black_box(fbm.sample_par(m)));
    });

    group.bench_with_input(BenchmarkId::new("sample_sequential", m), &m, |b, &m| {
      let fbm = FBM::new(hurst, n, None);
      b.iter(|| {
        let v: Vec<_> = (0..m).map(|_| fbm.sample()).collect();
        black_box(v)
      });
    });
  }

  group.finish();
}

criterion_group!(
  benches,
  bench_fgn_by_size,
  bench_fgn_by_hurst,
  bench_fgn_f32_vs_f64,
  bench_fgn_sample_par,
  bench_fbm_by_size,
  bench_fbm_sample_par,
);

criterion_main!(benches);
