use std::hint::black_box;
use std::time::Duration;

use criterion::BatchSize;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use rand_distr::Distribution;
use stochastic_rs::distributions::DistributionSampler;
use stochastic_rs::distributions::SimdDistribution;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::ndarray::array;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::autoregressive::ar::Ar;
use stochastic_rs::stochastic::autoregressive::arima::Arima;
use stochastic_rs::stochastic::autoregressive::ma::Ma;
use stochastic_rs::stochastic::diffusion::ou::Ou;
use stochastic_rs::stochastic::jump::cgmy::Cgmy;
use stochastic_rs::stochastic::jump::cts::Cts;
use stochastic_rs::stochastic::noise::gn::Gn;
use stochastic_rs::stochastic::process::bm::Bm;
use stochastic_rs::stochastic::volatility::heston::Euler;
use stochastic_rs::stochastic::volatility::heston_hull_white_2f::HestonHullWhite2F;
use stochastic_rs::traits::FloatExt;
use stochastic_rs::traits::ProcessExt;
use stochastic_rs_core::simd_rng::Deterministic;

fn bench_process_generation(c: &mut Criterion) {
  let mut group = c.benchmark_group("ProcessGeneration");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(500));

  for &n in &[1_000usize, 100_000usize] {
    let std_dev = (1.0f64 / (n.saturating_sub(1).max(1) as f64)).sqrt();

    group.bench_with_input(BenchmarkId::new("normal/fill_slice", n), &n, |b, &n| {
      let mut dist = SimdNormal::<f64>::new(0.0, std_dev).seeded(&Unseeded);
      let mut out = vec![0.0f64; n.saturating_sub(1)];
      b.iter(|| {
        dist.fill_slice(&mut out);
        black_box(out[0])
      });
    });

    group.bench_with_input(
      BenchmarkId::new("normal/rand_distr_loop", n),
      &n,
      |b, &n| {
        let mut rng = rand::rng();
        let dist = rand_distr::Normal::<f64>::new(0.0, std_dev).unwrap();
        let mut out = vec![0.0f64; n.saturating_sub(1)];
        b.iter(|| {
          for x in &mut out {
            *x = dist.sample(&mut rng);
          }
          black_box(out[0])
        });
      },
    );

    group.bench_with_input(BenchmarkId::new("process/Gn.sample", n), &n, |b, &n| {
      let gn = Gn::<f64, _>::new(n.saturating_sub(1), Some(1.0), Unseeded);
      b.iter(|| black_box(gn.sample()));
    });

    group.bench_with_input(BenchmarkId::new("process/Bm.sample", n), &n, |b, &n| {
      let bm = Bm::<f64, _>::new(n, Some(1.0), Unseeded);
      b.iter(|| black_box(bm.sample()));
    });

    group.bench_with_input(BenchmarkId::new("process/Ou.sample", n), &n, |b, &n| {
      let ou = Ou::<f64, _>::new(2.0, 0.0, 0.2, n, Some(0.0), Some(1.0), Unseeded);
      b.iter(|| black_box(ou.sample()));
    });

    group.bench_with_input(BenchmarkId::new("process/Bm.old_style", n), &n, |b, &n| {
      let gn = Gn::<f64, _>::new(n.saturating_sub(1), Some(1.0), Unseeded);
      b.iter(|| {
        let inc = gn.sample();
        let mut bm = vec![0.0f64; n];
        for i in 1..n {
          bm[i] = bm[i - 1] + inc[i - 1];
        }
        black_box(bm[n - 1])
      });
    });

    group.bench_with_input(BenchmarkId::new("process/Ou.old_style", n), &n, |b, &n| {
      let gn = Gn::<f64, _>::new(n.saturating_sub(1), Some(1.0), Unseeded);
      let dt = gn.dt();
      b.iter(|| {
        let noise = gn.sample();
        let mut ou = vec![0.0f64; n];
        for i in 1..n {
          ou[i] = ou[i - 1] + 2.0 * (0.0 - ou[i - 1]) * dt + 0.2 * noise[i - 1];
        }
        black_box(ou[n - 1])
      });
    });

    group.bench_with_input(
      BenchmarkId::new("process/Bm.rand_baseline", n),
      &n,
      |b, &n| {
        let mut rng = rand::rng();
        let dist = rand_distr::Normal::<f64>::new(0.0, std_dev).unwrap();
        b.iter_batched_ref(
          || vec![0.0f64; n],
          |out| {
            let mut acc = 0.0f64;
            for item in out.iter_mut().take(n).skip(1) {
              acc += dist.sample(&mut rng);
              *item = acc;
            }
            black_box(*out.last().expect("n must be > 0"))
          },
          BatchSize::SmallInput,
        );
      },
    );

    let mut dist = SimdNormal::<f64>::new(0.0, std_dev).seeded(&Unseeded);
    let mut increments = vec![0.0f64; n.saturating_sub(1)];
    dist.fill_slice(&mut increments);

    group.bench_with_input(BenchmarkId::new("bm/cumsum_only", n), &n, |b, &n| {
      b.iter_batched_ref(
        || vec![0.0f64; n],
        |out| {
          let mut acc = 0.0f64;
          for (item, inc) in out.iter_mut().take(n).skip(1).zip(increments.iter()) {
            acc += *inc;
            *item = acc;
          }
          black_box(*out.last().expect("n must be > 0"))
        },
        BatchSize::SmallInput,
      );
    });
  }

  group.bench_function("process/Ar.sample", |b| {
    let ar = Ar::<f64, _>::new(array![0.5, -0.2], 1.0, 4096, None, Unseeded);
    b.iter(|| black_box(ar.sample()));
  });

  group.bench_function("process/Ma.sample", |b| {
    let ma = Ma::<f64, _>::new(array![0.4, -0.2], 1.0, 4096, Unseeded);
    b.iter(|| black_box(ma.sample()));
  });

  group.bench_function("process/Arima.sample", |b| {
    let arima = Arima::<f64, _>::new(array![0.5], array![0.3], 1, 1.0, 4096, Unseeded);
    b.iter(|| black_box(arima.sample()));
  });

  group.bench_function("process/Cts.sample", |b| {
    let cts = Cts::<f64, _>::new(4.0, 7.0, 0.6, 1_000, 1_024, Some(0.0), Some(1.0), Unseeded);
    b.iter(|| black_box(cts.sample()));
  });

  group.bench_function("process/Cgmy.sample", |b| {
    let cgmy = Cgmy::<f64, _>::new(
      0.5,
      4.0,
      7.0,
      0.6,
      1_000,
      1_024,
      Some(0.0),
      Some(1.0),
      Unseeded,
    );
    b.iter(|| black_box(cgmy.sample()));
  });

  group.finish();
}

/// The paper's experiment 4.2 parameters (the type's `Default`) under a pinned seed.
fn seeded_heston_hull_white_2f<T: FloatExt>() -> HestonHullWhite2F<T, Euler, Deterministic> {
  let d = HestonHullWhite2F::<T>::default();
  let rho = d.rho().clone();
  HestonHullWhite2F::new(
    d.s0,
    d.v0,
    d.kappa,
    d.theta,
    d.sigma,
    d.theta_r,
    d.a,
    d.sigma1,
    d.sigma2,
    d.b,
    rho,
    d.r0,
    d.n,
    d.t,
    Deterministic::new(42),
  )
}

fn bench_heston_hull_white_2f(c: &mut Criterion) {
  let mut group = c.benchmark_group("HestonHullWhite2F");
  for &n in &[256usize, 4_096] {
    group.throughput(Throughput::Elements(n as u64));
    // Built outside `b.iter`: construction and the Cholesky factorisation are not measured.
    let euler = seeded_heston_hull_white_2f::<f64>().with_points(n);
    group.bench_with_input(BenchmarkId::new("euler/sample", n), &n, |b, _| {
      b.iter(|| black_box(euler.sample()));
    });
    let qe = euler.clone().qe();
    group.bench_with_input(BenchmarkId::new("qe/sample", n), &n, |b, _| {
      b.iter(|| black_box(qe.sample()));
    });
  }
  let euler = seeded_heston_hull_white_2f::<f64>();
  let qe = euler.clone().qe();
  group.throughput(Throughput::Elements(1_000 * 252));
  group.bench_function("euler/sample_par/1000", |b| {
    b.iter(|| black_box(euler.sample_par(1_000)));
  });
  group.bench_function("qe/sample_par/1000", |b| {
    b.iter(|| black_box(qe.sample_par(1_000)));
  });
  #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
  {
    use stochastic_rs::stochastic::device::Metal;
    let euler = seeded_heston_hull_white_2f::<f32>().on::<Metal>();
    let qe = seeded_heston_hull_white_2f::<f32>().qe().on::<Metal>();
    group.throughput(Throughput::Elements(100_000 * 252));
    group.bench_function("euler/metal/sample_par/100000", |b| {
      b.iter(|| black_box(euler.sample_par(100_000)));
    });
    group.bench_function("qe/metal/sample_par/100000", |b| {
      b.iter(|| black_box(qe.sample_par(100_000)));
    });
  }
  group.finish();
}

criterion_group!(
  benches,
  bench_process_generation,
  bench_heston_hull_white_2f
);
criterion_main!(benches);
