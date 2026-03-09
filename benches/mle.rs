use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use stochastic_rs::stats::mle::DensityApprox;
use stochastic_rs::stats::mle::fit_mle;
use stochastic_rs::stochastic::diffusion::cir::CIR;
use stochastic_rs::stochastic::diffusion::ou::OU;
use stochastic_rs::traits::ProcessExt;

fn bench_density_eval(c: &mut Criterion) {
  let mut group = c.benchmark_group("density_eval");
  group.measurement_time(Duration::from_secs(3));

  let cir = CIR::seeded(3.0, 0.3, 0.2, 100, Some(0.4), Some(1.0), None, 0);
  let ou = OU::seeded(2.0, 1.0, 0.3, 100, Some(1.0), Some(1.0), 0);
  let dt = 1.0 / 250.0;

  for (name, density) in [
    ("Euler", DensityApprox::Euler),
    ("Kessler", DensityApprox::Kessler),
    ("Ozaki", DensityApprox::Ozaki),
    ("ShojiOzaki", DensityApprox::ShojiOzaki),
    ("Elerian", DensityApprox::Elerian),
  ] {
    group.bench_function(BenchmarkId::new("CIR", name), |b| {
      b.iter(|| black_box(density.density(&cir, 0.4, 0.41, 0.0, dt)))
    });
    group.bench_function(BenchmarkId::new("OU", name), |b| {
      b.iter(|| black_box(density.density(&ou, 0.5, 0.55, 0.0, 0.01)))
    });
  }

  // OU Exact
  group.bench_function(BenchmarkId::new("OU", "Exact"), |b| {
    b.iter(|| black_box(DensityApprox::Exact.density(&ou, 0.5, 0.55, 0.0, 0.01)))
  });

  group.finish();
}

fn bench_log_likelihood(c: &mut Criterion) {
  let mut group = c.benchmark_group("log_likelihood");
  group.measurement_time(Duration::from_secs(3));

  for &n in &[1_000usize, 5_000, 10_000] {
    let ou = OU::seeded(2.0, 1.0, 0.3, n + 1, Some(1.0), Some(10.0), 42);
    let path = ou.sample();
    let dt = 10.0 / n as f64;

    group.bench_function(BenchmarkId::new("OU/Euler", n), |b| {
      b.iter(|| {
        let mut sum = 0.0f64;
        for i in 0..path.len() - 1 {
          let d = DensityApprox::Euler.density(&ou, path[i], path[i + 1], 0.0, dt);
          sum += d.max(1e-30).ln();
        }
        black_box(sum)
      })
    });

    group.bench_function(BenchmarkId::new("OU/Kessler", n), |b| {
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
    // OU Euler
    let ou = OU::seeded(2.0, 1.0, 0.3, n + 1, Some(1.0), Some(10.0), 42);
    let path = ou.sample();
    let dt = 10.0 / n as f64;

    group.bench_function(BenchmarkId::new("OU/Euler", label), |b| {
      b.iter(|| {
        let mut ou_fit = OU::seeded(1.0, 0.5, 0.5, 100, Some(1.0), Some(1.0), 0);
        black_box(fit_mle(&mut ou_fit, &path, dt, DensityApprox::Euler, None))
      })
    });

    group.bench_function(BenchmarkId::new("OU/Kessler", label), |b| {
      b.iter(|| {
        let mut ou_fit = OU::seeded(1.0, 0.5, 0.5, 100, Some(1.0), Some(1.0), 0);
        black_box(fit_mle(
          &mut ou_fit,
          &path,
          dt,
          DensityApprox::Kessler,
          None,
        ))
      })
    });

    // CIR Kessler
    let cir = CIR::seeded(3.0, 0.3, 0.2, n + 1, Some(0.4), Some(10.0), None, 42);
    let cir_path = cir.sample();
    let cir_dt = 10.0 / n as f64;

    group.bench_function(BenchmarkId::new("CIR/Kessler", label), |b| {
      b.iter(|| {
        let mut cir_fit = CIR::seeded(1.0, 0.5, 0.3, 100, Some(0.4), Some(1.0), None, 0);
        black_box(fit_mle(
          &mut cir_fit,
          &cir_path,
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
