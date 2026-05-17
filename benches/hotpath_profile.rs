//! Hotpath profiler for distributions, simd_rng, and Fgn.
//!
//! ```sh
//! cargo bench --bench hotpath_profile --features='hotpath-alloc'
//! ```
use std::hint::black_box;

use stochastic_rs::distributions::alpha_stable::SimdAlphaStable;
use stochastic_rs::distributions::exp::SimdExpZig;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::noise::fgn::Fgn;
use stochastic_rs::traits::ProcessExt;

const N_SAMPLES: usize = 1_000_000;
const FGN_STEPS: usize = 4096;
const FGN_PATHS: usize = 64;

#[hotpath::measure]
fn profile_simd_rng_f64(n: usize) {
  let mut rng = stochastic_rs::simd_rng::rng();
  let mut acc = 0.0_f64;
  for _ in 0..n {
    acc += rng.next_f64();
  }
  black_box(acc);
}

#[hotpath::measure]
fn profile_simd_rng_f32(n: usize) {
  let mut rng = stochastic_rs::simd_rng::rng();
  let mut acc = 0.0_f32;
  for _ in 0..n {
    acc += rng.next_f32();
  }
  black_box(acc);
}

#[hotpath::measure]
fn profile_normal_scalar_f64(n: usize) {
  let dist: SimdNormal<f64> = SimdNormal::new(0.0, 1.0, &stochastic_rs_core::simd_rng::Unseeded);
  let mut rng = rand::rng();
  let mut acc = 0.0_f64;
  for _ in 0..n {
    acc += rand_distr::Distribution::sample(&dist, &mut rng);
  }
  black_box(acc);
}

#[hotpath::measure]
fn profile_normal_bulk_f64(n: usize) {
  let dist: SimdNormal<f64> = SimdNormal::new(0.0, 1.0, &stochastic_rs_core::simd_rng::Unseeded);
  let mut buf = vec![0.0_f64; n];
  dist.fill_slice_fast(&mut buf);
  black_box(&buf);
}

#[hotpath::measure]
fn profile_normal_scalar_f32(n: usize) {
  let dist: SimdNormal<f32> = SimdNormal::new(0.0, 1.0, &stochastic_rs_core::simd_rng::Unseeded);
  let mut rng = rand::rng();
  let mut acc = 0.0_f32;
  for _ in 0..n {
    acc += rand_distr::Distribution::sample(&dist, &mut rng);
  }
  black_box(acc);
}

#[hotpath::measure]
fn profile_normal_bulk_f32(n: usize) {
  let dist: SimdNormal<f32> = SimdNormal::new(0.0, 1.0, &stochastic_rs_core::simd_rng::Unseeded);
  let mut buf = vec![0.0_f32; n];
  dist.fill_slice_fast(&mut buf);
  black_box(&buf);
}

#[hotpath::measure]
fn profile_exp_bulk_f64(n: usize) {
  let dist: SimdExpZig<f64> = SimdExpZig::new(1.5, &stochastic_rs_core::simd_rng::Unseeded);
  let mut buf = vec![0.0_f64; n];
  let mut rng = rand::rng();
  dist.fill_slice(&mut rng, &mut buf);
  black_box(&buf);
}

#[hotpath::measure]
fn profile_alpha_stable_general_f64(n: usize) {
  let dist = SimdAlphaStable::<f64>::new(1.7, 0.3, 1.0, 0.0, &stochastic_rs_core::simd_rng::Unseeded);
  let mut buf = vec![0.0_f64; n];
  dist.fill_slice_fast(&mut buf);
  black_box(&buf);
}

#[hotpath::measure]
fn profile_alpha_stable_gaussian_f64(n: usize) {
  let dist = SimdAlphaStable::<f64>::new(2.0, 0.0, 1.0, 0.0, &stochastic_rs_core::simd_rng::Unseeded);
  let mut buf = vec![0.0_f64; n];
  dist.fill_slice_fast(&mut buf);
  black_box(&buf);
}

#[hotpath::measure]
fn profile_alpha_stable_cauchy_f64(n: usize) {
  let dist = SimdAlphaStable::<f64>::new(1.0, 0.5, 1.0, 0.0, &stochastic_rs_core::simd_rng::Unseeded);
  let mut buf = vec![0.0_f64; n];
  dist.fill_slice_fast(&mut buf);
  black_box(&buf);
}

#[hotpath::measure]
fn profile_fgn_construction(n: usize, hurst: f64) {
  for _ in 0..64 {
    let fgn = Fgn::<f64>::new(hurst, n, Some(1.0), Unseeded);
    black_box(&fgn);
  }
}

#[hotpath::measure]
fn profile_fgn_sample(fgn: &Fgn<f64>, m: usize) {
  for _ in 0..m {
    let path = fgn.sample();
    black_box(&path);
  }
}

#[hotpath::main]
fn main() {
  println!("=== hotpath profiler: distributions / simd_rng / fgn ===\n");

  println!("[1/7] simd_rng f64 ({N_SAMPLES})...");
  profile_simd_rng_f64(N_SAMPLES);

  println!("[2/7] simd_rng f32 ({N_SAMPLES})...");
  profile_simd_rng_f32(N_SAMPLES);

  println!("[3/7] Normal f64/f32 ({N_SAMPLES})...");
  profile_normal_scalar_f64(N_SAMPLES);
  profile_normal_bulk_f64(N_SAMPLES);
  profile_normal_scalar_f32(N_SAMPLES);
  profile_normal_bulk_f32(N_SAMPLES);

  println!("[4/7] Exponential f64 ({N_SAMPLES})...");
  profile_exp_bulk_f64(N_SAMPLES);

  println!("[5/7] Alpha-Stable f64 ({N_SAMPLES})...");
  profile_alpha_stable_general_f64(N_SAMPLES);
  profile_alpha_stable_gaussian_f64(N_SAMPLES);
  profile_alpha_stable_cauchy_f64(N_SAMPLES);

  println!("[6/7] Fgn construction (n={FGN_STEPS}, 64x)...");
  profile_fgn_construction(FGN_STEPS, 0.7);

  println!("[7/7] Fgn sampling (n={FGN_STEPS}, {FGN_PATHS} paths)...");
  let fgn = Fgn::<f64>::new(0.7, FGN_STEPS, Some(1.0), Unseeded);
  profile_fgn_sample(&fgn, FGN_PATHS);

  println!("\nDone.\n");
}
