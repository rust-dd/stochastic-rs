//! `SimdFloatExt` implementations for `f32` (lane width 8) and `f64`
//! (lane width 8, decomposed into two AVX2-friendly `f64x4` halves).

use rand::Rng;
use rand::RngExt;
use wide::f32x8;
use wide::f64x4;
use wide::f64x8;
use wide::i32x4;
use wide::i32x8;

use crate::traits::SimdFloatExt;

mod markov_lift;

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

  #[inline]
  fn splat(val: f32) -> f32x8 {
    f32x8::splat(val)
  }

  #[inline]
  fn simd_from_array(arr: [f32; 8]) -> f32x8 {
    f32x8::from(arr)
  }

  #[inline]
  fn simd_to_array(v: f32x8) -> [f32; 8] {
    v.to_array()
  }

  #[inline]
  fn simd_ln(v: f32x8) -> f32x8 {
    v.ln()
  }

  #[inline]
  fn simd_sqrt(v: f32x8) -> f32x8 {
    v.sqrt()
  }

  #[inline]
  fn simd_cos(v: f32x8) -> f32x8 {
    v.cos()
  }

  #[inline]
  fn simd_sin(v: f32x8) -> f32x8 {
    v.sin()
  }

  #[inline]
  fn simd_exp(v: f32x8) -> f32x8 {
    v.exp()
  }

  #[inline]
  fn simd_exp_m1(v: f32x8) -> f32x8 {
    v.exp_m1()
  }

  #[inline]
  fn simd_tan(v: f32x8) -> f32x8 {
    v.tan()
  }

  #[inline]
  fn simd_max(a: f32x8, b: f32x8) -> f32x8 {
    a.max(b)
  }

  #[inline]
  fn simd_powf(v: f32x8, exp: f32) -> f32x8 {
    v.powf_simd(f32x8::splat(exp))
  }

  #[inline]
  fn fill_uniform<R: Rng + ?Sized>(rng: &mut R, out: &mut [f32]) {
    fill_f32_zero_one(rng, out)
  }

  #[inline]
  fn fill_uniform_simd<R: crate::simd_rng::SimdRngExt>(rng: &mut R, out: &mut [f32]) {
    rng.fill_uniform_f32(out);
  }

  #[inline]
  fn sample_uniform<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    rng.random()
  }

  #[inline(always)]
  fn sample_uniform_simd<R: crate::simd_rng::SimdRngExt>(rng: &mut R) -> f32 {
    rng.next_f32()
  }

  #[inline]
  fn simd_from_i32x8(v: wide::i32x8) -> f32x8 {
    v.round_float()
  }

  const PREFERS_F32_WN: bool = true;

  #[inline(always)]
  fn history_sum_fused(we: &[Self], h_state: &[Self], j_state: &[Self]) -> Self {
    markov_lift::f32_lanes::history_sum_fused(we, h_state, j_state)
  }

  #[inline(always)]
  fn update_state_fused(
    h_state: &mut [Self],
    j_state: &mut [Self],
    exp_neg: &[Self],
    omx: &[Self],
    f_prev: Self,
    g_dw: Self,
  ) {
    markov_lift::f32_lanes::update_state_fused(h_state, j_state, exp_neg, omx, f_prev, g_dw)
  }

  #[inline(always)]
  fn batch_history_accumulate(we_l: Self, h_row: &[Self], j_row: &[Self], history: &mut [Self]) {
    markov_lift::f32_lanes::batch_history_accumulate(we_l, h_row, j_row, history)
  }

  #[inline(always)]
  fn batch_update_state(
    e_l: Self,
    omx_l: Self,
    h_row: &mut [Self],
    j_row: &mut [Self],
    f_prev: &[Self],
    g_dw: &[Self],
  ) {
    markov_lift::f32_lanes::batch_update_state(e_l, omx_l, h_row, j_row, f_prev, g_dw)
  }
}

impl SimdFloatExt for f64 {
  type Simd = f64x8;

  #[inline]
  fn splat(val: f64) -> f64x8 {
    f64x8::splat(val)
  }

  #[inline]
  fn simd_from_array(arr: [f64; 8]) -> f64x8 {
    f64x8::from(arr)
  }

  #[inline]
  fn simd_to_array(v: f64x8) -> [f64; 8] {
    v.to_array()
  }

  #[inline]
  fn simd_ln(v: f64x8) -> f64x8 {
    v.ln()
  }

  #[inline]
  fn simd_sqrt(v: f64x8) -> f64x8 {
    v.sqrt()
  }

  #[inline]
  fn simd_cos(v: f64x8) -> f64x8 {
    v.cos()
  }

  #[inline]
  fn simd_sin(v: f64x8) -> f64x8 {
    v.sin()
  }

  #[inline]
  fn simd_exp(v: f64x8) -> f64x8 {
    v.exp()
  }

  #[inline]
  fn simd_exp_m1(v: f64x8) -> f64x8 {
    v.exp_m1()
  }

  #[inline]
  fn simd_tan(v: f64x8) -> f64x8 {
    v.tan()
  }

  #[inline]
  fn simd_max(a: f64x8, b: f64x8) -> f64x8 {
    a.max(b)
  }

  #[inline]
  fn simd_powf(v: f64x8, exp: f64) -> f64x8 {
    v.powf_simd(f64x8::splat(exp))
  }

  #[inline]
  fn fill_uniform<R: Rng + ?Sized>(rng: &mut R, out: &mut [f64]) {
    fill_f64_zero_one(rng, out)
  }

  #[inline]
  fn fill_uniform_simd<R: crate::simd_rng::SimdRngExt>(rng: &mut R, out: &mut [f64]) {
    rng.fill_uniform_f64(out);
  }

  #[inline]
  fn sample_uniform<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    rng.random()
  }

  #[inline(always)]
  fn sample_uniform_simd<R: crate::simd_rng::SimdRngExt>(rng: &mut R) -> f64 {
    rng.next_f64()
  }

  #[inline]
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
  fn history_sum_fused(we: &[Self], h_state: &[Self], j_state: &[Self]) -> Self {
    markov_lift::f64_lanes::history_sum_fused(we, h_state, j_state)
  }

  #[inline(always)]
  fn update_state_fused(
    h_state: &mut [Self],
    j_state: &mut [Self],
    exp_neg: &[Self],
    omx: &[Self],
    f_prev: Self,
    g_dw: Self,
  ) {
    markov_lift::f64_lanes::update_state_fused(h_state, j_state, exp_neg, omx, f_prev, g_dw)
  }

  #[inline(always)]
  fn batch_history_accumulate(we_l: Self, h_row: &[Self], j_row: &[Self], history: &mut [Self]) {
    markov_lift::f64_lanes::batch_history_accumulate(we_l, h_row, j_row, history)
  }

  #[inline(always)]
  fn batch_update_state(
    e_l: Self,
    omx_l: Self,
    h_row: &mut [Self],
    j_row: &mut [Self],
    f_prev: &[Self],
    g_dw: &[Self],
  ) {
    markov_lift::f64_lanes::batch_update_state(e_l, omx_l, h_row, j_row, f_prev, g_dw)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Lanes against the scalar `exp_m1` at both ends of the range, where a plain `e^v − 1` cancels or overflows.
  #[test]
  fn simd_exp_m1_matches_the_scalar_form() {
    for z in [
      0.0, 1e-300, -1e-300, 1e-12, -1e-12, 1e-8, -1e-8, 1e-4, -1e-4, 0.1, -0.1, 1.0, -1.0, 5.0,
      -5.0, 20.0, -20.0, 40.0, -40.0,
    ] {
      let lanes = f64::simd_to_array(f64::simd_exp_m1(f64::splat(z)));
      let want = z.exp_m1();
      assert!(
        (lanes[0] - want).abs() <= 4.0 * f64::EPSILON * want.abs().max(f64::MIN_POSITIVE),
        "f64 exp_m1({z}) = {} vs {want}",
        lanes[0]
      );
      let lanes32 = f32::simd_to_array(f32::simd_exp_m1(f32::splat(z as f32)));
      let want32 = (z as f32).exp_m1();
      assert!(
        (lanes32[0] - want32).abs() <= 4.0 * f32::EPSILON * want32.abs().max(f32::MIN_POSITIVE),
        "f32 exp_m1({z}) = {} vs {want32}",
        lanes32[0]
      );
    }
  }
}
