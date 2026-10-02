use std::time::Instant;

use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::cauchy::SimdCauchy;
use crate::exp::SimdExp;
use crate::lognormal::SimdLogNormal;
use crate::normal::SimdNormal;
use crate::traits::SimdDistribution;

#[test]
#[ignore = "perf benchmark (5-10M sample loop): run with --ignored or via cargo bench"]
fn bench_normal_simd_vs_rand() {
  let n = 10_000_000usize;
  let warmup = 1_000_000usize;

  {
    let mut rng = rand::rng();
    let mut d = SimdNormal::<f32>::new(0.0, 1.0).seeded(&Unseeded);
    let rd = rand_distr::Normal::<f32>::new(0.0, 1.0).unwrap();
    let mut s = 0.0f32;
    for _ in 0..warmup {
      s += d.sample();
      s += rd.sample(&mut rng);
    }
    std::hint::black_box(s);
  }

  let mut simd = SimdNormal::<f32>::new(0.0, 1.0).seeded(&Unseeded);
  let mut s_sum = 0.0f32;
  let t0 = Instant::now();
  for _ in 0..n {
    s_sum += simd.sample();
  }
  let dt_s = t0.elapsed();

  let mut rng = rand::rng();
  let rd = rand_distr::Normal::<f32>::new(0.0, 1.0).unwrap();
  let mut r_sum = 0.0f32;
  let t1 = Instant::now();
  for _ in 0..n {
    r_sum += rd.sample(&mut rng);
  }
  let dt_r = t1.elapsed();

  println!(
    "Normal single: simd {:?}, sum={:.3} | rand_distr {:?}, sum={:.3}",
    dt_s, s_sum, dt_r, r_sum
  );
  assert!(!s_sum.is_nan() && !r_sum.is_nan());
}

#[test]
#[ignore = "perf benchmark (5-10M sample loop): run with --ignored or via cargo bench"]
fn bench_lognormal_simd_vs_rand() {
  let n = 10_000_000usize;
  let warmup = 1_000_000usize;

  {
    let mut rng = rand::rng();
    let mut d = SimdLogNormal::<f32>::new(0.2f32, 0.8).seeded(&Unseeded);
    let rd = rand_distr::LogNormal::<f32>::new(0.2, 0.8).unwrap();
    let mut s = 0.0f32;
    for _ in 0..warmup {
      s += d.sample();
      s += rd.sample(&mut rng);
    }
    std::hint::black_box(s);
  }

  let mut simd = SimdLogNormal::<f32>::new(0.2, 0.8).seeded(&Unseeded);
  let mut s_sum = 0.0f32;
  let t0 = Instant::now();
  for _ in 0..n {
    s_sum += simd.sample();
  }
  let dt_s = t0.elapsed();

  let mut rng = rand::rng();
  let rd = rand_distr::LogNormal::<f32>::new(0.2, 0.8).unwrap();
  let mut r_sum = 0.0f32;
  let t1 = Instant::now();
  for _ in 0..n {
    r_sum += rd.sample(&mut rng);
  }
  let dt_r = t1.elapsed();

  println!(
    "LogNormal single: simd {:?}, sum={:.3} | rand_distr {:?}, sum={:.3}",
    dt_s, s_sum, dt_r, r_sum
  );
  assert!(!s_sum.is_nan() && !r_sum.is_nan());
}

#[test]
#[ignore = "perf benchmark (5-10M sample loop): run with --ignored or via cargo bench"]
fn bench_exp_simd_vs_rand() {
  let n = 10_000_000usize;
  let lambda = 1.5f32;
  let warmup = 1_000_000usize;

  {
    let mut rng = rand::rng();
    let mut d = SimdExp::<f32>::new(lambda).seeded(&Unseeded);
    let rd = rand_distr::Exp::<f32>::new(lambda).unwrap();
    let mut s = 0.0f32;
    for _ in 0..warmup {
      s += d.sample();
      s += rd.sample(&mut rng);
    }
    std::hint::black_box(s);
  }

  let mut simd = SimdExp::<f32>::new(lambda).seeded(&Unseeded);
  let mut s_sum = 0.0f32;
  let t0 = Instant::now();
  for _ in 0..n {
    s_sum += simd.sample();
  }
  let dt_s = t0.elapsed();

  let mut rng = rand::rng();
  let rd = rand_distr::Exp::<f32>::new(lambda).unwrap();
  let mut r_sum = 0.0f32;
  let t1 = Instant::now();
  for _ in 0..n {
    r_sum += rd.sample(&mut rng);
  }
  let dt_r = t1.elapsed();

  println!(
    "Exp single: simd {:?}, sum={:.3} | rand_distr {:?}, sum={:.3}",
    dt_s, s_sum, dt_r, r_sum
  );
  assert!(!s_sum.is_nan() && !r_sum.is_nan());
}

#[test]
#[ignore = "perf benchmark (5-10M sample loop): run with --ignored or via cargo bench"]
fn bench_cauchy_simd_vs_rand() {
  let n = 10_000_000usize;
  let warmup = 1_000_000usize;

  {
    let mut rng = rand::rng();
    let mut d = SimdCauchy::<f32>::new(0.0f32, 1.0).seeded(&Unseeded);
    let rd = rand_distr::Cauchy::<f32>::new(0.0, 1.0).unwrap();
    let mut s = 0.0f32;
    for _ in 0..warmup {
      s += d.sample();
      s += rd.sample(&mut rng);
    }
    std::hint::black_box(s);
  }

  let mut simd = SimdCauchy::<f32>::new(0.0, 1.0).seeded(&Unseeded);
  let mut s_sum = 0.0f32;
  let t0 = Instant::now();
  for _ in 0..n {
    s_sum += simd.sample();
  }
  let dt_s = t0.elapsed();

  let mut rng = rand::rng();
  let rd = rand_distr::Cauchy::<f32>::new(0.0, 1.0).unwrap();
  let mut r_sum = 0.0f32;
  let t1 = Instant::now();
  for _ in 0..n {
    r_sum += rd.sample(&mut rng);
  }
  let dt_r = t1.elapsed();

  println!(
    "Cauchy single: simd {:?}, sum={:.3} | rand_distr {:?}, sum={:.3}",
    dt_s, s_sum, dt_r, r_sum
  );
  assert!(!s_sum.is_nan() && !r_sum.is_nan());
}
