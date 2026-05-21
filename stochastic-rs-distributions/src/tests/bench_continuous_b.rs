use std::time::Instant;

use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::beta::SimdBeta;
use crate::chi_square::SimdChiSquared;
use crate::gamma::SimdGamma;
use crate::poisson::SimdPoisson;
use crate::studentt::SimdStudentT;
use crate::weibull::SimdWeibull;

#[test]
#[ignore = "perf benchmark (5-10M sample loop): run with --ignored or via cargo bench"]
fn bench_gamma_simd_vs_rand() {
  let n = 10_000_000usize;
  let warmup = 1_000_000usize;

  {
    let mut rng = rand::rng();
    let d = SimdGamma::<f32>::new(2.0f32, 2.0, &Unseeded);
    let rd = rand_distr::Gamma::<f32>::new(2.0, 2.0).unwrap();
    let mut s = 0.0f32;
    for _ in 0..warmup {
      s += d.sample(&mut rng);
      s += rd.sample(&mut rng);
    }
    std::hint::black_box(s);
  }

  let mut rng = rand::rng();
  let simd = SimdGamma::<f32>::new(2.0, 2.0, &Unseeded);
  let mut s_sum = 0.0f32;
  let t0 = Instant::now();
  for _ in 0..n {
    s_sum += simd.sample(&mut rng);
  }
  let dt_s = t0.elapsed();

  let mut rng = rand::rng();
  let rd = rand_distr::Gamma::<f32>::new(2.0, 2.0).unwrap();
  let mut r_sum = 0.0f32;
  let t1 = Instant::now();
  for _ in 0..n {
    r_sum += rd.sample(&mut rng);
  }
  let dt_r = t1.elapsed();

  println!(
    "Gamma single: simd {:?}, sum={:.3} | rand_distr {:?}, sum={:.3}",
    dt_s, s_sum, dt_r, r_sum
  );
  assert!(!s_sum.is_nan() && !r_sum.is_nan());
}

#[test]
#[ignore = "perf benchmark (5-10M sample loop): run with --ignored or via cargo bench"]
fn bench_weibull_simd_vs_rand() {
  let n = 10_000_000usize;
  let warmup = 1_000_000usize;

  {
    let mut rng = rand::rng();
    let d = SimdWeibull::<f32>::new(1.0f32, 1.5, &Unseeded);
    let rd = rand_distr::Weibull::<f32>::new(1.0, 1.5).unwrap();
    let mut s = 0.0f32;
    for _ in 0..warmup {
      s += d.sample(&mut rng);
      s += rd.sample(&mut rng);
    }
    std::hint::black_box(s);
  }

  let mut rng = rand::rng();
  let simd = SimdWeibull::<f32>::new(1.0, 1.5, &Unseeded);
  let mut s_sum = 0.0f32;
  let t0 = Instant::now();
  for _ in 0..n {
    s_sum += simd.sample(&mut rng);
  }
  let dt_s = t0.elapsed();

  let mut rng = rand::rng();
  let rd = rand_distr::Weibull::<f32>::new(1.0, 1.5).unwrap();
  let mut r_sum = 0.0f32;
  let t1 = Instant::now();
  for _ in 0..n {
    r_sum += rd.sample(&mut rng);
  }
  let dt_r = t1.elapsed();

  println!(
    "Weibull single: simd {:?}, sum={:.3} | rand_distr {:?}, sum={:.3}",
    dt_s, s_sum, dt_r, r_sum
  );
  assert!(!s_sum.is_nan() && !r_sum.is_nan());
}

#[test]
#[ignore = "perf benchmark (5-10M sample loop): run with --ignored or via cargo bench"]
fn bench_beta_simd_vs_rand() {
  let n = 10_000_000usize;
  let warmup = 1_000_000usize;

  {
    let mut rng = rand::rng();
    let d = SimdBeta::<f32>::new(2.0f32, 2.0, &Unseeded);
    let rd = rand_distr::Beta::<f32>::new(2.0, 2.0).unwrap();
    let mut s = 0.0f32;
    for _ in 0..warmup {
      s += d.sample(&mut rng);
      s += rd.sample(&mut rng);
    }
    std::hint::black_box(s);
  }

  let mut rng = rand::rng();
  let simd = SimdBeta::<f32>::new(2.0, 2.0, &Unseeded);
  let mut s_sum = 0.0f32;
  let t0 = Instant::now();
  for _ in 0..n {
    s_sum += simd.sample(&mut rng);
  }
  let dt_s = t0.elapsed();

  let mut rng = rand::rng();
  let rd = rand_distr::Beta::<f32>::new(2.0, 2.0).unwrap();
  let mut r_sum = 0.0f32;
  let t1 = Instant::now();
  for _ in 0..n {
    r_sum += rd.sample(&mut rng);
  }
  let dt_r = t1.elapsed();

  println!(
    "Beta single: simd {:?}, sum={:.3} | rand_distr {:?}, sum={:.3}",
    dt_s, s_sum, dt_r, r_sum
  );
  assert!(!s_sum.is_nan() && !r_sum.is_nan());
}

#[test]
#[ignore = "perf benchmark (5-10M sample loop): run with --ignored or via cargo bench"]
fn bench_chisq_simd_vs_rand() {
  let n = 10_000_000usize;
  let warmup = 1_000_000usize;

  {
    let mut rng = rand::rng();
    let d = SimdChiSquared::<f32>::new(5.0f32, &Unseeded);
    let rd = rand_distr::ChiSquared::<f32>::new(5.0).unwrap();
    let mut s = 0.0f32;
    for _ in 0..warmup {
      s += d.sample(&mut rng);
      s += rd.sample(&mut rng);
    }
    std::hint::black_box(s);
  }

  let mut rng = rand::rng();
  let simd = SimdChiSquared::<f32>::new(5.0, &Unseeded);
  let mut s_sum = 0.0f32;
  let t0 = Instant::now();
  for _ in 0..n {
    s_sum += simd.sample(&mut rng);
  }
  let dt_s = t0.elapsed();

  let mut rng = rand::rng();
  let rd = rand_distr::ChiSquared::<f32>::new(5.0).unwrap();
  let mut r_sum = 0.0f32;
  let t1 = Instant::now();
  for _ in 0..n {
    r_sum += rd.sample(&mut rng);
  }
  let dt_r = t1.elapsed();

  println!(
    "ChiSq single: simd {:?}, sum={:.3} | rand_distr {:?}, sum={:.3}",
    dt_s, s_sum, dt_r, r_sum
  );
  assert!(!s_sum.is_nan() && !r_sum.is_nan());
}

#[test]
#[ignore = "perf benchmark (5-10M sample loop): run with --ignored or via cargo bench"]
fn bench_studentt_simd_vs_rand() {
  let n = 10_000_000usize;
  let warmup = 1_000_000usize;

  {
    let mut rng = rand::rng();
    let d = SimdStudentT::<f32>::new(5.0f32, &Unseeded);
    let rd = rand_distr::StudentT::<f32>::new(5.0).unwrap();
    let mut s = 0.0f32;
    for _ in 0..warmup {
      s += d.sample(&mut rng);
      s += rd.sample(&mut rng);
    }
    std::hint::black_box(s);
  }

  let mut rng = rand::rng();
  let simd = SimdStudentT::<f32>::new(5.0, &Unseeded);
  let mut s_sum = 0.0f32;
  let t0 = Instant::now();
  for _ in 0..n {
    s_sum += simd.sample(&mut rng);
  }
  let dt_s = t0.elapsed();

  let mut rng = rand::rng();
  let rd = rand_distr::StudentT::<f32>::new(5.0).unwrap();
  let mut r_sum = 0.0f32;
  let t1 = Instant::now();
  for _ in 0..n {
    r_sum += rd.sample(&mut rng);
  }
  let dt_r = t1.elapsed();

  println!(
    "StudentT single: simd {:?}, sum={:.3} | rand_distr {:?}, sum={:.3}",
    dt_s, s_sum, dt_r, r_sum
  );
  assert!(!s_sum.is_nan() && !r_sum.is_nan());
}

#[test]
#[ignore = "perf benchmark (5-10M sample loop): run with --ignored or via cargo bench"]
fn bench_poisson_simd_vs_rand() {
  let n = 5_000_000usize;
  let warmup = 500_000usize;

  {
    let mut rng = rand::rng();
    let d = SimdPoisson::<u32>::new(4.0, &Unseeded);
    let rd = rand_distr::Poisson::<f64>::new(4.0).unwrap();
    let mut s: u64 = 0;
    for _ in 0..warmup {
      s += d.sample(&mut rng) as u64;
      s += rd.sample(&mut rng) as u64;
    }
    std::hint::black_box(s);
  }

  let mut rng = rand::rng();
  let simd = SimdPoisson::<u32>::new(4.0, &Unseeded);
  let mut s_sum: u64 = 0;
  let t0 = Instant::now();
  for _ in 0..n {
    s_sum += simd.sample(&mut rng) as u64;
  }
  let dt_s = t0.elapsed();

  let mut rng = rand::rng();
  let rd = rand_distr::Poisson::<f64>::new(4.0).unwrap();
  let mut r_sum: u64 = 0;
  let t1 = Instant::now();
  for _ in 0..n {
    r_sum += rd.sample(&mut rng) as u64;
  }
  let dt_r = t1.elapsed();

  println!(
    "Poisson single: simd {:?}, sum={} | rand_distr {:?}, sum={}",
    dt_s, s_sum, dt_r, r_sum
  );
  assert!(s_sum > 0 && r_sum > 0);
}
