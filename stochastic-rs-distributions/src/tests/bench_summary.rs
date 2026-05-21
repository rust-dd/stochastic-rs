use std::time::Instant;

use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::beta::SimdBeta;
use crate::cauchy::SimdCauchy;
use crate::chi_square::SimdChiSquared;
use crate::exp::SimdExp;
use crate::gamma::SimdGamma;
use crate::lognormal::SimdLogNormal;
use crate::normal::SimdNormal;
use crate::pareto::SimdPareto;
use crate::poisson::SimdPoisson;
use crate::studentt::SimdStudentT;
use crate::weibull::SimdWeibull;

struct Row {
  name: &'static str,
  simd_ms: f64,
  rand_ms: f64,
}

fn time_f32<F1, F2>(
  rows: &mut Vec<Row>,
  n: usize,
  name: &'static str,
  mut simd_fn: F1,
  mut rand_fn: F2,
) where
  F1: FnMut() -> f32,
  F2: FnMut() -> f32,
{
  use std::hint::black_box;
  let warmup = n / 5;
  let mut w = 0.0f32;
  for _ in 0..warmup {
    w += simd_fn();
    w += rand_fn();
  }
  black_box(w);

  let t0 = Instant::now();
  let mut s_sum = 0.0f32;
  for _ in 0..n {
    s_sum += simd_fn();
  }
  let dt_simd = t0.elapsed().as_secs_f64() * 1000.0;

  let t1 = Instant::now();
  let mut r_sum = 0.0f32;
  for _ in 0..n {
    r_sum += rand_fn();
  }
  let dt_rand = t1.elapsed().as_secs_f64() * 1000.0;

  black_box(s_sum);
  black_box(r_sum);
  rows.push(Row {
    name,
    simd_ms: dt_simd,
    rand_ms: dt_rand,
  });
}

fn time_u32<F1, F2>(
  rows: &mut Vec<Row>,
  n: usize,
  name: &'static str,
  mut simd_fn: F1,
  mut rand_fn: F2,
) where
  F1: FnMut() -> u32,
  F2: FnMut() -> u32,
{
  use std::hint::black_box;
  let warmup = n / 5;
  let mut w: u64 = 0;
  for _ in 0..warmup {
    w += simd_fn() as u64;
    w += rand_fn() as u64;
  }
  black_box(w);

  let t0 = Instant::now();
  let mut s_sum: u64 = 0;
  for _ in 0..n {
    s_sum += simd_fn() as u64;
  }
  let dt_simd = t0.elapsed().as_secs_f64() * 1000.0;

  let t1 = Instant::now();
  let mut r_sum: u64 = 0;
  for _ in 0..n {
    r_sum += rand_fn() as u64;
  }
  let dt_rand = t1.elapsed().as_secs_f64() * 1000.0;

  black_box(s_sum);
  black_box(r_sum);
  rows.push(Row {
    name,
    simd_ms: dt_simd,
    rand_ms: dt_rand,
  });
}

#[test]
#[ignore = "perf benchmark summary (~119s): run with --ignored or via cargo bench"]
fn bench_summary_table() {
  let n_f = 5_000_000usize;
  let n_i = 5_000_000usize;

  let mut rows: Vec<Row> = Vec::new();

  {
    let mut rng = rand::rng();
    let simd: SimdNormal<f32> = SimdNormal::new(0.0, 1.0, &Unseeded);
    let mut rng2 = rand::rng();
    let rd = rand_distr::Normal::<f32>::new(0.0, 1.0).unwrap();
    time_f32(
      &mut rows,
      n_f,
      "Normal",
      || simd.sample(&mut rng),
      || rd.sample(&mut rng2),
    );
  }

  {
    let mut rng = rand::rng();
    let simd = SimdLogNormal::<f32>::new(0.2, 0.8, &Unseeded);
    let mut rng2 = rand::rng();
    let rd = rand_distr::LogNormal::<f32>::new(0.2, 0.8).unwrap();
    time_f32(
      &mut rows,
      n_f,
      "LogNormal",
      || simd.sample(&mut rng),
      || rd.sample(&mut rng2),
    );
  }

  {
    let mut rng = rand::rng();
    let simd = SimdExp::<f32>::new(1.5, &Unseeded);
    let mut rng2 = rand::rng();
    let rd = rand_distr::Exp::<f32>::new(1.5).unwrap();
    time_f32(
      &mut rows,
      n_f,
      "Exp",
      || simd.sample(&mut rng),
      || rd.sample(&mut rng2),
    );
  }

  {
    let mut rng = rand::rng();
    let simd = SimdCauchy::<f32>::new(0.0, 1.0, &Unseeded);
    let mut rng2 = rand::rng();
    let rd = rand_distr::Cauchy::<f32>::new(0.0, 1.0).unwrap();
    time_f32(
      &mut rows,
      n_f,
      "Cauchy",
      || simd.sample(&mut rng),
      || rd.sample(&mut rng2),
    );
  }

  {
    let mut rng = rand::rng();
    let simd = SimdGamma::<f32>::new(2.0, 2.0, &Unseeded);
    let mut rng2 = rand::rng();
    let rd = rand_distr::Gamma::<f32>::new(2.0, 2.0).unwrap();
    time_f32(
      &mut rows,
      n_f,
      "Gamma",
      || simd.sample(&mut rng),
      || rd.sample(&mut rng2),
    );
  }

  {
    let mut rng = rand::rng();
    let simd = SimdWeibull::<f32>::new(1.0, 1.5, &Unseeded);
    let mut rng2 = rand::rng();
    let rd = rand_distr::Weibull::<f32>::new(1.0, 1.5).unwrap();
    time_f32(
      &mut rows,
      n_f,
      "Weibull",
      || simd.sample(&mut rng),
      || rd.sample(&mut rng2),
    );
  }

  {
    let mut rng = rand::rng();
    let simd = SimdBeta::<f32>::new(2.0, 2.0, &Unseeded);
    let mut rng2 = rand::rng();
    let rd = rand_distr::Beta::<f32>::new(2.0, 2.0).unwrap();
    time_f32(
      &mut rows,
      n_f,
      "Beta",
      || simd.sample(&mut rng),
      || rd.sample(&mut rng2),
    );
  }

  {
    let mut rng = rand::rng();
    let simd = SimdChiSquared::<f32>::new(5.0, &Unseeded);
    let mut rng2 = rand::rng();
    let rd = rand_distr::ChiSquared::<f32>::new(5.0).unwrap();
    time_f32(
      &mut rows,
      n_f,
      "ChiSquared",
      || simd.sample(&mut rng),
      || rd.sample(&mut rng2),
    );
  }

  {
    let mut rng = rand::rng();
    let simd = SimdStudentT::<f32>::new(5.0, &Unseeded);
    let mut rng2 = rand::rng();
    let rd = rand_distr::StudentT::<f32>::new(5.0).unwrap();
    time_f32(
      &mut rows,
      n_f,
      "StudentT",
      || simd.sample(&mut rng),
      || rd.sample(&mut rng2),
    );
  }

  {
    let mut rng = rand::rng();
    let simd = SimdPoisson::<u32>::new(4.0, &Unseeded);
    let mut rng2 = rand::rng();
    let rd = rand_distr::Poisson::<f64>::new(4.0).unwrap();
    time_u32(
      &mut rows,
      n_i,
      "Poisson",
      || simd.sample(&mut rng),
      || rd.sample(&mut rng2) as u32,
    );
  }

  #[allow(unused)]
  {
    let _ = SimdPareto::<f64>::new(1.0, 1.5, &Unseeded);
  }

  println!(
    "{:<14} {:>12} {:>14}",
    "Distribution", "simd (ms)", "rand_distr (ms)"
  );
  println!("{:-<14} {:-<12} {:-<14}", "", "", "");
  for r in &rows {
    println!("{:<14} {:>12.2} {:>14.2}", r.name, r.simd_ms, r.rand_ms);
  }

  println!();
  println!(
    "{:<24} {:>12} {:>14} {:>8}",
    "Normal fill_slice", "simd (ms)", "rand_distr (ms)", "speedup"
  );
  println!("{:-<24} {:-<12} {:-<14} {:-<8}", "", "", "", "");
  let total = 5_000_000usize;
  for &size in &[8, 16, 64, 256, 1024, 10_000, 100_000] {
    let iters = total / size;
    let simd = SimdNormal::<f32>::new(0.0, 1.0, &Unseeded);
    let rd = rand_distr::Normal::<f32>::new(0.0, 1.0).unwrap();
    let mut buf = vec![0.0f32; size];

    let mut rng = rand::rng();
    let t0 = Instant::now();
    for _ in 0..iters {
      simd.fill_slice(&mut rng, &mut buf);
      std::hint::black_box(&buf);
    }
    let dt_simd = t0.elapsed().as_secs_f64() * 1000.0;

    let mut rng2 = rand::rng();
    let t1 = Instant::now();
    for _ in 0..iters {
      for x in buf.iter_mut() {
        *x = rd.sample(&mut rng2);
      }
      std::hint::black_box(&buf);
    }
    let dt_rand = t1.elapsed().as_secs_f64() * 1000.0;

    let speedup = dt_rand / dt_simd;
    println!(
      "  n={:<20} {:>10.2} {:>14.2} {:>7.2}x",
      size, dt_simd, dt_rand, speedup
    );
  }
}
