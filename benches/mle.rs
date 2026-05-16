use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stats::mle::DensityApprox;
use stochastic_rs::stats::mle::fit_mle;
use stochastic_rs::stochastic::diffusion::cir::Cir;
use stochastic_rs::stochastic::diffusion::ou::Ou;
use stochastic_rs::traits::ProcessExt;

fn bench_density_eval(c: &mut Criterion) {
  let mut group = c.benchmark_group("density_eval");
  group.measurement_time(Duration::from_secs(3));

  let cir = Cir::new(
    3.0,
    0.3,
    0.2,
    100,
    Some(0.4),
    Some(1.0),
    None,
    Deterministic::new(0),
  );
  let ou = Ou::new(
    2.0,
    1.0,
    0.3,
    100,
    Some(1.0),
    Some(1.0),
    Deterministic::new(0),
  );
  let dt = 1.0 / 250.0;

  for (name, density) in [
    ("Euler", DensityApprox::Euler),
    ("Kessler", DensityApprox::Kessler),
    ("Ozaki", DensityApprox::Ozaki),
    ("ShojiOzaki", DensityApprox::ShojiOzaki),
    ("Elerian", DensityApprox::Elerian),
  ] {
    group.bench_function(BenchmarkId::new("Cir", name), |b| {
      b.iter(|| black_box(density.density(&cir, 0.4, 0.41, 0.0, dt)))
    });
    group.bench_function(BenchmarkId::new("Ou", name), |b| {
      b.iter(|| black_box(density.density(&ou, 0.5, 0.55, 0.0, 0.01)))
    });
  }

  // Ou Exact
  group.bench_function(BenchmarkId::new("Ou", "Exact"), |b| {
    b.iter(|| black_box(DensityApprox::Exact.density(&ou, 0.5, 0.55, 0.0, 0.01)))
  });

  group.finish();
}

fn bench_log_likelihood(c: &mut Criterion) {
  let mut group = c.benchmark_group("log_likelihood");
  group.measurement_time(Duration::from_secs(3));

  for &n in &[1_000usize, 5_000, 10_000] {
    let ou = Ou::new(
      2.0,
      1.0,
      0.3,
      n + 1,
      Some(1.0),
      Some(10.0),
      Deterministic::new(42),
    );
    let path = ou.sample();
    let dt = 10.0 / n as f64;

    group.bench_function(BenchmarkId::new("Ou/Euler", n), |b| {
      b.iter(|| {
        let mut sum = 0.0f64;
        for i in 0..path.len() - 1 {
          let d = DensityApprox::Euler.density(&ou, path[i], path[i + 1], 0.0, dt);
          sum += d.max(1e-30).ln();
        }
        black_box(sum)
      })
    });

    group.bench_function(BenchmarkId::new("Ou/Kessler", n), |b| {
      b.iter(|| {
        let mut sum = 0.0f64;
        for i in 0..path.len() - 1 {
          let d = DensityApprox::Kessler.density(&ou, path[i], path[i + 1], 0.0, dt);
          sum += d.max(1e-30).ln();
        }
        black_box(sum)
      })
    });
  }

  group.finish();
}

fn bench_mle_fit(c: &mut Criterion) {
  let mut group = c.benchmark_group("mle_fit");
  group.measurement_time(Duration::from_secs(10));
  group.sample_size(10);

  for &(n, label) in &[(1_000usize, "1k"), (5_000, "5k")] {
    // Ou Euler
    let ou = Ou::new(
      2.0,
      1.0,
      0.3,
      n + 1,
      Some(1.0),
      Some(10.0),
      Deterministic::new(42),
    );
    let path = ou.sample();
    let dt = 10.0 / n as f64;

    group.bench_function(BenchmarkId::new("Ou/Euler", label), |b| {
      b.iter(|| {
        let mut ou_fit = Ou::new(
          1.0,
          0.5,
          0.5,
          100,
          Some(1.0),
          Some(1.0),
          Deterministic::new(0),
        );
        black_box(fit_mle(
          &mut ou_fit,
          path.view(),
          dt,
          DensityApprox::Euler,
          None,
        ))
      })
    });

    group.bench_function(BenchmarkId::new("Ou/Kessler", label), |b| {
      b.iter(|| {
        let mut ou_fit = Ou::new(
          1.0,
          0.5,
          0.5,
          100,
          Some(1.0),
          Some(1.0),
          Deterministic::new(0),
        );
        black_box(fit_mle(
          &mut ou_fit,
          path.view(),
          dt,
          DensityApprox::Kessler,
          None,
        ))
      })
    });

    // Cir Kessler
    let cir = Cir::new(
      3.0,
      0.3,
      0.2,
      n + 1,
      Some(0.4),
      Some(10.0),
      None,
      Deterministic::new(42),
    );
    let cir_path = cir.sample();
    let cir_dt = 10.0 / n as f64;

    group.bench_function(BenchmarkId::new("Cir/Kessler", label), |b| {
      b.iter(|| {
        let mut cir_fit = Cir::new(
          1.0,
          0.5,
          0.3,
          100,
          Some(0.4),
          Some(1.0),
          None,
          Deterministic::new(0),
        );
        black_box(fit_mle(
          &mut cir_fit,
          cir_path.view(),
          cir_dt,
          DensityApprox::Kessler,
          None,
        ))
      })
    });
  }

  group.finish();
}

criterion_group!(
  benches,
  bench_density_eval,
  bench_log_likelihood,
  bench_mle_fit
);
criterion_main!(benches);
