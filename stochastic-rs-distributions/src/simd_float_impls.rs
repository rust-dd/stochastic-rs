//! `SimdFloatExt` implementations for `f32` (lane width 8) and `f64`
//! (lane width 8, decomposed into two AVX2-friendly `f64x4` halves).

use rand::Rng;
use wide::f32x8;
use wide::f64x4;
use wide::f64x8;
use wide::i32x4;
use wide::i32x8;

use crate::traits::SimdFloatExt;

fn fill_f32_zero_one<R: Rng + ?Sized>(rng: &mut R, out: &mut [f32]) {
  for x in out.iter_mut() {
    *x = rng.random();
  }
}

fn fill_f64_zero_one<R: Rng + ?Sized>(rng: &mut R, out: &mut [f64]) {
  for x in out.iter_mut() {
    *x = rng.random();
  }
}

impl SimdFloatExt for f32 {
  type Simd = f32x8;

  fn splat(val: f32) -> f32x8 {
    f32x8::splat(val)
  }

  fn simd_from_array(arr: [f32; 8]) -> f32x8 {
    f32x8::from(arr)
  }

  fn simd_to_array(v: f32x8) -> [f32; 8] {
    v.to_array()
  }

  fn simd_ln(v: f32x8) -> f32x8 {
    v.ln()
  }

  fn simd_sqrt(v: f32x8) -> f32x8 {
    v.sqrt()
  }

  fn simd_cos(v: f32x8) -> f32x8 {
    v.cos()
  }

  fn simd_sin(v: f32x8) -> f32x8 {
    v.sin()
  }

  fn simd_exp(v: f32x8) -> f32x8 {
    v.exp()
  }

  fn simd_tan(v: f32x8) -> f32x8 {
    v.tan()
  }

  fn simd_max(a: f32x8, b: f32x8) -> f32x8 {
    a.max(b)
  }

  fn simd_powf(v: f32x8, exp: f32) -> f32x8 {
    v.powf(exp)
  }

  fn fill_uniform<R: Rng + ?Sized>(rng: &mut R, out: &mut [f32]) {
    fill_f32_zero_one(rng, out)
  }

  fn fill_uniform_simd<R: crate::simd_rng::SimdRngExt>(rng: &mut R, out: &mut [f32]) {
    rng.fill_uniform_f32(out);
  }

  fn sample_uniform<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    rng.random()
  }

  #[inline(always)]
  fn sample_uniform_simd<R: crate::simd_rng::SimdRngExt>(rng: &mut R) -> f32 {
    rng.next_f32()
  }

  fn simd_from_i32x8(v: wide::i32x8) -> f32x8 {
    v.round_float()
  }

  const PREFERS_F32_WN: bool = true;

  #[inline(always)]
  fn from_f64_fast(v: f64) -> f32 {
    v as f32
  }

  #[inline(always)]
  fn from_f32_fast(v: f32) -> f32 {
    v
  }

  fn pi() -> f32 {
    std::f32::consts::PI
  }

  fn two_pi() -> f32 {
    2.0 * std::f32::consts::PI
  }

  fn min_positive_val() -> f32 {
    f32::MIN_POSITIVE
  }
}

impl SimdFloatExt for f64 {
  type Simd = f64x8;

  fn splat(val: f64) -> f64x8 {
    f64x8::splat(val)
  }

  fn simd_from_array(arr: [f64; 8]) -> f64x8 {
    f64x8::from(arr)
  }

  fn simd_to_array(v: f64x8) -> [f64; 8] {
    v.to_array()
  }

  fn simd_ln(v: f64x8) -> f64x8 {
    v.ln()
  }

  fn simd_sqrt(v: f64x8) -> f64x8 {
    v.sqrt()
  }

  fn simd_cos(v: f64x8) -> f64x8 {
    v.cos()
  }

  fn simd_sin(v: f64x8) -> f64x8 {
    v.sin()
  }

  fn simd_exp(v: f64x8) -> f64x8 {
    v.exp()
  }

  fn simd_tan(v: f64x8) -> f64x8 {
    v.tan()
  }

  fn simd_max(a: f64x8, b: f64x8) -> f64x8 {
    a.max(b)
  }

  fn simd_powf(v: f64x8, exp: f64) -> f64x8 {
    v.powf(exp)
  }

  fn fill_uniform<R: Rng + ?Sized>(rng: &mut R, out: &mut [f64]) {
    fill_f64_zero_one(rng, out)
  }

  fn fill_uniform_simd<R: crate::simd_rng::SimdRngExt>(rng: &mut R, out: &mut [f64]) {
    rng.fill_uniform_f64(out);
  }

  fn sample_uniform<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    rng.random()
  }

  #[inline(always)]
  fn sample_uniform_simd<R: crate::simd_rng::SimdRngExt>(rng: &mut R) -> f64 {
    rng.next_f64()
  }

  fn simd_from_i32x8(v: wide::i32x8) -> f64x8 {
    // `wide::f64x8::from_i32x8` falls back to 8 scalar `as f64` casts on
    // AVX2 (no AVX-512). Going through `f64x4::from_i32x4` uses a single
    // `vcvtdq2pd` per half, replacing 8 scalar conversions with 2 SIMD
    // ones in the Ziggurat hot path.
    let halves: [i32x4; 2] = unsafe { core::mem::transmute::<i32x8, [i32x4; 2]>(v) };
    let fa = f64x4::from_i32x4(halves[0]).to_array();
    let fb = f64x4::from_i32x4(halves[1]).to_array();
    f64x8::new([fa[0], fa[1], fa[2], fa[3], fb[0], fb[1], fb[2], fb[3]])
  }

  const PREFERS_F32_WN: bool = false;

  #[inline(always)]
  fn from_f64_fast(v: f64) -> f64 {
    v
  }

  fn pi() -> f64 {
    std::f64::consts::PI
  }

  fn two_pi() -> f64 {
    2.0 * std::f64::consts::PI
  }

  fn min_positive_val() -> f64 {
    f64::MIN_POSITIVE
  }
}

