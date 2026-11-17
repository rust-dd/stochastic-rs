use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use stochastic_rs::distributions::DistributionSampler;
use stochastic_rs::distributions::SimdDistribution;
use stochastic_rs::distributions::pareto::SimdPareto;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stats::evt::block_maxima;
use stochastic_rs::stats::evt::fit_gev;
use stochastic_rs::stats::evt::fit_hill;
use stochastic_rs::stats::evt::fit_pot;
use stochastic_rs::stats::garch::GarchSpec;
use stochastic_rs::stats::garch::fit_garch;
use stochastic_rs::stats::mle::DensityApprox;
use stochastic_rs::stats::mle::fit_mle;
use stochastic_rs::stochastic::autoregressive::garch::Garch;
use stochastic_rs::stochastic::diffusion::cir::Cir;
use stochastic_rs::stochastic::diffusion::ou::Ou;
use stochastic_rs::stochastic::params::VarianceBoundary;
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
    VarianceBoundary::Floor,
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

  group.bench_function(BenchmarkId::new("Ou", "Exact"), |b| {
    b.iter(|| black_box(DensityApprox::Exact.density(&ou, 0.5, 0.55, 0.0, 0.01)))
  });

  group.bench_function(BenchmarkId::new("Cir", "Exact"), |b| {
    b.iter(|| black_box(DensityApprox::Exact.density(&cir, 0.4, 0.41, 0.0, dt)))
  });

  group.bench_function(BenchmarkId::new("Cir", "transition_pdf"), |b| {
    b.iter(|| black_box(cir.transition_pdf(dt, 0.4, 0.41)))
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
          sum += d.ln();
        }
        black_box(sum)
      })
    });

    group.bench_function(BenchmarkId::new("Ou/Kessler", n), |b| {
      b.iter(|| {
        let mut sum = 0.0f64;
        for i in 0..path.len() - 1 {
          let d = DensityApprox::Kessler.density(&ou, path[i], path[i + 1], 0.0, dt);
          sum += d.ln();
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
        black_box(fit_mle(&mut ou_fit, path.view(), dt, DensityApprox::Euler, None).unwrap())
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
        black_box(fit_mle(&mut ou_fit, path.view(), dt, DensityApprox::Kessler, None).unwrap())
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
      VarianceBoundary::Floor,
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
          VarianceBoundary::Floor,
          Deterministic::new(0),
        );
        black_box(
          fit_mle(
            &mut cir_fit,
            cir_path.view(),
            cir_dt,
            DensityApprox::Kessler,
            None,
          )
          .unwrap(),
        )
      })
    });
  }

  group.finish();
}

fn bench_garch_fit(c: &mut Criterion) {
  let mut group = c.benchmark_group("garch_fit");
  group.measurement_time(Duration::from_secs(10));
  group.sample_size(10);
  for &(n, label) in &[(1_000usize, "1k"), (5_000, "5k")] {
    let process = Garch::<f64, _>::new(
      0.05,
      ndarray::Array1::from(vec![0.10]),
      ndarray::Array1::from(vec![0.85]),
      n,
      Deterministic::new(42),
    );
    let returns = process.sample();
    for (name, spec) in [
      ("Garch11", GarchSpec::garch(1, 1)),
      ("Gjr11", GarchSpec::gjr(1, 1)),
      ("Egarch11", GarchSpec::egarch(1, 1)),
    ] {
      group.bench_function(BenchmarkId::new(name, label), |b| {
        b.iter(|| black_box(fit_garch(returns.view(), spec).unwrap()))
      });
    }
  }
  group.finish();
}

fn bench_evt_fit(c: &mut Criterion) {
  let mut group = c.benchmark_group("evt_fit");
  group.measurement_time(Duration::from_secs(5));
  let mut dist = SimdPareto::<f64>::new(1.0, 3.0).seeded(&Deterministic::new(7));
  let mut losses = vec![0.0; 20_000];
  dist.fill_slice(&mut losses);
  let losses = ndarray::Array1::from(losses);
  group.bench_function("hill_20k_k500", |b| {
    b.iter(|| black_box(fit_hill(losses.view(), 500).unwrap()))
  });
  group.bench_function("pot_20k_u3", |b| {
    b.iter(|| black_box(fit_pot(losses.view(), 3.0).unwrap()))
  });
  let maxima = block_maxima(losses.view(), 100);
  group.bench_function("gev_200_maxima", |b| {
    b.iter(|| black_box(fit_gev(maxima.view()).unwrap()))
  });
  group.finish();
}

criterion_group!(
  benches,
  bench_density_eval,
  bench_log_likelihood,
  bench_mle_fit,
  bench_garch_fit,
  bench_evt_fit
);
criterion_main!(benches);
