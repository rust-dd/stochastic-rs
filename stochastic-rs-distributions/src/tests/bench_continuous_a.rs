use std::time::Instant;

use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::cauchy::SimdCauchy;
use crate::exp::SimdExp;
use crate::exp::SimdExpZig;
use crate::lognormal::SimdLogNormal;
use crate::normal::SimdNormal;

#[test]
#[ignore = "perf benchmark (5-10M sample loop): run with --ignored or via cargo bench"]
fn bench_normal_simd_vs_rand() {
  let n = 10_000_000usize;
  let warmup = 1_000_000usize;

  {
    let mut rng = rand::rng();
    let d: SimdNormal<f32> = SimdNormal::new(0.0, 1.0, &Unseeded);
    let rd = rand_distr::Normal::<f32>::new(0.0, 1.0).unwrap();
    let mut s = 0.0f32;
    for _ in 0..warmup {
      s += d.sample(&mut rng);
      s += rd.sample(&mut rng);
    }
    std::hint::black_box(s);
  }

  let mut rng = rand::rng();
  let simd: SimdNormal<f32> = SimdNormal::new(0.0, 1.0, &Unseeded);
  let mut s_sum = 0.0f32;
  let t0 = Instant::now();
  for _ in 0..n {
    s_sum += simd.sample(&mut rng);
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
    let d = SimdLogNormal::<f32>::new(0.2f32, 0.8, &Unseeded);
    let rd = rand_distr::LogNormal::<f32>::new(0.2, 0.8).unwrap();
    let mut s = 0.0f32;
    for _ in 0..warmup {
      s += d.sample(&mut rng);
      s += rd.sample(&mut rng);
    }
    std::hint::black_box(s);
  }

  let mut rng = rand::rng();
  let simd = SimdLogNormal::<f32>::new(0.2, 0.8, &Unseeded);
  let mut s_sum = 0.0f32;
  let t0 = Instant::now();
  for _ in 0..n {
    s_sum += simd.sample(&mut rng);
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
    let d = SimdExp::<f32>::new(lambda, &Unseeded);
    let rd = rand_distr::Exp::<f32>::new(lambda).unwrap();
    let mut s = 0.0f32;
    for _ in 0..warmup {
      s += d.sample(&mut rng);
      s += rd.sample(&mut rng);
    }
    std::hint::black_box(s);
  }

  let mut rng = rand::rng();
  let simd = SimdExp::<f32>::new(lambda, &Unseeded);
  let mut s_sum = 0.0f32;
  let t0 = Instant::now();
  for _ in 0..n {
    s_sum += simd.sample(&mut rng);
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
fn bench_exp_zig_simd_vs_rand() {
  let n = 10_000_000usize;
  let lambda = 1.5f32;
  let warmup = 1_000_000usize;

  {
    let mut rng = rand::rng();
    let d: SimdExpZig<f32> = SimdExpZig::new(lambda, &Unseeded);
    let d2 = SimdExp::<f32>::new(lambda, &Unseeded);
    let rd = rand_distr::Exp::<f32>::new(lambda).unwrap();
    let mut s = 0.0f32;
    for _ in 0..warmup {
      s += d.sample(&mut rng);
      s += d2.sample(&mut rng);
      s += rd.sample(&mut rng);
    }
    std::hint::black_box(s);
  }

  let mut rng = rand::rng();
  let zig: SimdExpZig<f32> = SimdExpZig::new(lambda, &Unseeded);
  let mut z_sum = 0.0f32;
  let t0 = Instant::now();
  for _ in 0..n {
    z_sum += zig.sample(&mut rng);
  }
  let dt_z = t0.elapsed();

  let mut rng = rand::rng();
  let old = SimdExp::<f32>::new(lambda, &Unseeded);
  let mut o_sum = 0.0f32;
  let t1 = Instant::now();
  for _ in 0..n {
    o_sum += old.sample(&mut rng);
  }
  let dt_o = t1.elapsed();

  let mut rng = rand::rng();
  let rd = rand_distr::Exp::<f32>::new(lambda).unwrap();
  let mut r_sum = 0.0f32;
  let t2 = Instant::now();
  for _ in 0..n {
    r_sum += rd.sample(&mut rng);
  }
  let dt_r = t2.elapsed();

  println!(
    "Exp Ziggurat: {:?}, sum={:.3} | Exp ICDF: {:?}, sum={:.3} | rand_distr: {:?}, sum={:.3}",
    dt_z, z_sum, dt_o, o_sum, dt_r, r_sum
  );
  assert!(!z_sum.is_nan() && !o_sum.is_nan() && !r_sum.is_nan());
}

#[test]
#[ignore = "perf benchmark (5-10M sample loop): run with --ignored or via cargo bench"]
fn bench_cauchy_simd_vs_rand() {
  let n = 10_000_000usize;
  let warmup = 1_000_000usize;

  {
    let mut rng = rand::rng();
    let d = SimdCauchy::<f32>::new(0.0f32, 1.0, &Unseeded);
    let rd = rand_distr::Cauchy::<f32>::new(0.0, 1.0).unwrap();
    let mut s = 0.0f32;
    for _ in 0..warmup {
      s += d.sample(&mut rng);
      s += rd.sample(&mut rng);
    }
    std::hint::black_box(s);
  }

  let mut rng = rand::rng();
  let simd = SimdCauchy::<f32>::new(0.0, 1.0, &Unseeded);
  let mut s_sum = 0.0f32;
  let t0 = Instant::now();
  for _ in 0..n {
    s_sum += simd.sample(&mut rng);
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
