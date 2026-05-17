//! Benchmarks for the `stats::filtering` module.

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use ndarray::Array1;
#[cfg(feature = "openblas")]
use ndarray::Array2;
use ndarray::ArrayView1;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::simd_rng::SimdRng;
use stochastic_rs::stats::filtering::ParticleFilter;
#[cfg(feature = "openblas")]
use stochastic_rs::stats::filtering::UkfState;
use stochastic_rs::stats::filtering::random_walk_metropolis;
#[cfg(feature = "openblas")]
use stochastic_rs::stats::filtering::unscented_kalman_step;

fn bench_particle(c: &mut Criterion) {
  c.bench_function("particle_filter_n500_step", |b| {
    let init_dist = SimdNormal::<f64>::new(0.0, 1.0, &stochastic_rs_core::simd_rng::Deterministic::new(1));
    let init_fn = move |_rng: &mut SimdRng| {
      let mut a = [0.0_f64];
      init_dist.fill_slice_fast(&mut a);
      Array1::from(vec![a[0]])
    };
    let trans_dist = SimdNormal::<f64>::new(0.0, 1.0, &stochastic_rs_core::simd_rng::Deterministic::new(2));
    let transition = move |prev: ArrayView1<f64>, _rng: &mut SimdRng| {
      let mut a = [0.0_f64];
      trans_dist.fill_slice_fast(&mut a);
      Array1::from(vec![prev[0] + a[0]])
    };
    let log_obs = |x: ArrayView1<f64>, y: ArrayView1<f64>| {
      let z = (y[0] - x[0]) / 0.3;
      -0.5 * z * z
    };
    let mut pf = ParticleFilter::new(500, init_fn, transition, log_obs, 11);
    let y = Array1::from(vec![0.0_f64]);
    b.iter(|| pf.step(y.view()));
  });
}

fn bench_mcmc(c: &mut Criterion) {
  let init = Array1::from(vec![0.0_f64]);
  let scale = Array1::from(vec![1.0_f64]);
  let log_target = |x: ArrayView1<f64>| -0.5 * x[0] * x[0];
  c.bench_function("metropolis_5k_after_burn1k", |b| {
    b.iter(|| {
      std::hint::black_box(random_walk_metropolis(
        init.view(),
        log_target,
        scale.view(),
        5_000,
        1_000,
        17,
      ))
    });
  });
}

#[cfg(feature = "openblas")]
fn bench_ukf(c: &mut Criterion) {
  let q = Array2::from_shape_vec((2, 2), vec![0.05_f64, 0.0, 0.0, 0.05]).unwrap();
  let r = Array2::from_shape_vec((1, 1), vec![0.5_f64]).unwrap();
  let transition = |x: ArrayView1<f64>| Array1::from(vec![x[0] + x[1] * 0.1, x[1]]);
  let measurement = |x: ArrayView1<f64>| Array1::from(vec![x[0]]);
  c.bench_function("ukf_step_state2_obs1", |b| {
    let mut state = UkfState {
      mean: Array1::from(vec![0.0_f64, 0.0]),
      covariance: Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap(),
    };
    let y = Array1::from(vec![0.0_f64]);
    b.iter(|| {
      state = unscented_kalman_step(
        &state,
        transition,
        measurement,
        q.view(),
        r.view(),
        y.view(),
        0.001,
        2.0,
        0.0,
      );
    });
  });
}

#[cfg(feature = "openblas")]
criterion_group!(benches, bench_particle, bench_mcmc, bench_ukf);
#[cfg(not(feature = "openblas"))]
criterion_group!(benches, bench_particle, bench_mcmc);
criterion_main!(benches);
