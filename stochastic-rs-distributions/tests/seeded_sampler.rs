//! The bulk `DistributionSampler` paths of a seeded stream: shape, replay, thread-count independence.

use num_complex::Complex;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::complex::ComplexDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::poisson::SimdPoisson;

#[test]
fn sample_n_returns_requested_length() {
  let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Unseeded);
  let out = dist.sample_n(1024);
  assert_eq!(out.len(), 1024);
}

#[test]
fn sample_matrix_float_has_expected_shape() {
  let mut dist = SimdNormal::<f32>::new(0.0, 1.0).seeded(&Unseeded);
  let out = dist.sample_matrix(32, 64);
  assert_eq!(out.shape(), &[32, 64]);
}

#[test]
fn sample_matrix_int_has_expected_shape() {
  let mut dist = SimdPoisson::<i64>::new(1.5, &Unseeded);
  let out = dist.sample_matrix(16, 8);
  assert_eq!(out.shape(), &[16, 8]);
}

/// Two identically seeded samplers agree on every public bulk path.
#[test]
fn sample_n_deterministic_all_paths() {
  let mut poisson_a = SimdPoisson::<u64>::new(4.5, &Deterministic::new(42));
  let mut poisson_b = SimdPoisson::<u64>::new(4.5, &Deterministic::new(42));
  assert_eq!(poisson_a.sample_n(64), poisson_b.sample_n(64));

  let mut normal_a = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(42));
  let mut normal_b = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(42));
  assert_eq!(normal_a.sample_n(64), normal_b.sample_n(64));
}

/// The parallel fan-out stays bit-identical across two identically seeded samplers on a 4-thread pool.
#[test]
fn sample_matrix_parallel_deterministic() {
  let pool = rayon::ThreadPoolBuilder::new()
    .num_threads(4)
    .build()
    .expect("failed to build 4-thread pool");
  let (a, b) = pool.install(|| {
    let mut dist_a = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(42));
    let mut dist_b = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(42));
    (
      dist_a.sample_matrix(200, 2000),
      dist_b.sample_matrix(200, 2000),
    )
  });
  assert_eq!(a, b);
}

/// Each parallel call advances the fork basis, so one stream never replays a matrix, seeded or not.
#[test]
fn sample_matrix_repeat_calls_advance() {
  let pool = rayon::ThreadPoolBuilder::new()
    .num_threads(4)
    .build()
    .expect("failed to build 4-thread pool");
  pool.install(|| {
    let mut det = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(42));
    let call1 = det.sample_matrix(200, 2000);
    let call2 = det.sample_matrix(200, 2000);
    assert_ne!(
      call1, call2,
      "Deterministic sample_matrix replayed the same matrix on a repeat call"
    );

    let mut unseeded = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Unseeded);
    let call1 = unseeded.sample_matrix(200, 2000);
    let call2 = unseeded.sample_matrix(200, 2000);
    assert_ne!(
      call1, call2,
      "Unseeded sample_matrix replayed the same matrix on a repeat call"
    );
  });
}

/// Identically seeded streams agree call for call, and a serial call in between leaves the fork basis alone.
#[test]
fn sample_matrix_call_sequence_deterministic() {
  let pool = rayon::ThreadPoolBuilder::new()
    .num_threads(4)
    .build()
    .expect("failed to build 4-thread pool");
  pool.install(|| {
    let mut dist_a = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(42));
    let mut dist_b = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(42));

    let a_call1 = dist_a.sample_matrix(200, 2000);
    let b_call1 = dist_b.sample_matrix(200, 2000);
    assert_eq!(a_call1, b_call1, "first parallel call diverged");

    let a_serial = dist_a.sample_matrix(2, 8);
    let b_serial = dist_b.sample_matrix(2, 8);
    assert_eq!(a_serial, b_serial, "serial call diverged");

    let a_call2 = dist_a.sample_matrix(200, 2000);
    let b_call2 = dist_b.sample_matrix(200, 2000);
    assert_eq!(a_call2, b_call2, "second parallel call diverged");
    assert_ne!(a_call1, a_call2, "second parallel call replayed the first");
  });
}

/// The worker count depends on the matrix size alone, never on the pool size.
#[test]
fn sample_matrix_is_thread_count_independent() {
  let sample_under = |threads: usize| {
    rayon::ThreadPoolBuilder::new()
      .num_threads(threads)
      .build()
      .expect("failed to build pool")
      .install(|| {
        let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(42));
        dist.sample_matrix(200, 2000)
      })
  };
  let under_1 = sample_under(1);
  let under_4 = sample_under(4);
  let under_8 = sample_under(8);
  assert_eq!(under_1, under_4, "1-thread and 4-thread pools diverged");
  assert_eq!(under_4, under_8, "4-thread and 8-thread pools diverged");
}

#[test]
fn distribution_sample_draws_from_the_caller_rng() {
  use rand::distr::Distribution;

  let d = SimdNormal::<f64>::new(0.0, 1.0);
  let draws = |seed: u64| {
    let mut rng = SimdRng::from_seed(seed);
    (0..32)
      .map(|_| d.sample(&mut rng).to_bits())
      .collect::<Vec<_>>()
  };
  assert_eq!(draws(1), draws(1));
  assert_ne!(draws(1), draws(999_999));
}

/// One 64-wide bulk chunk equals 64 pops: each part refills its 64-buffer with the same kernel call.
#[test]
fn complex_fill_matches_pops_over_one_chunk() {
  let law = ComplexDistribution::new(
    SimdNormal::<f64>::new(0.0, 1.0),
    SimdNormal::<f64>::new(0.5, 2.0),
  );
  let bits = |zs: &[Complex<f64>]| {
    zs.iter()
      .map(|z| (z.re.to_bits(), z.im.to_bits()))
      .collect::<Vec<_>>()
  };
  let fill = |seed: u64| {
    let mut out = vec![Complex::new(0.0, 0.0); 64];
    law.seeded(&Deterministic::new(seed)).fill_slice(&mut out);
    bits(&out)
  };
  let mut twin = law.seeded(&Deterministic::new(9));
  let pops = (0..64).map(|_| twin.sample()).collect::<Vec<_>>();
  assert_eq!(fill(9), bits(&pops));
  assert_eq!(fill(9), fill(9));
  assert_ne!(fill(9), fill(10));
}
