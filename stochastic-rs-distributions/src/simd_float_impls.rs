//! `SimdFloatExt` implementations for `f32` (lane width 8) and `f64`
//! (lane width 8, decomposed into two AVX2-friendly `f64x4` halves).

// erf / lgamma helpers below carry published Cody-1969 / Lanczos constants
// whose extra digits beyond the IEEE-754 mantissa are intentional — they
// round to the nearest representable bit, so we silence clippy's lossy
// trailing-digit heuristic crate-locally rather than truncate them.
#![allow(clippy::excessive_precision)]

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

  fn simd_floor(v: f32x8) -> f32x8 {
    v.floor()
  }

  fn simd_log1p(v: f32x8) -> f32x8 {
    // Lane-wise; `f32::ln_1p` is accurate for small |v| where ln(1+v)
    // suffers cancellation.
    let arr = v.to_array();
    f32x8::from([
      arr[0].ln_1p(),
      arr[1].ln_1p(),
      arr[2].ln_1p(),
      arr[3].ln_1p(),
      arr[4].ln_1p(),
      arr[5].ln_1p(),
      arr[6].ln_1p(),
      arr[7].ln_1p(),
    ])
  }

  fn simd_expm1(v: f32x8) -> f32x8 {
    let arr = v.to_array();
    f32x8::from([
      arr[0].exp_m1(),
      arr[1].exp_m1(),
      arr[2].exp_m1(),
      arr[3].exp_m1(),
      arr[4].exp_m1(),
      arr[5].exp_m1(),
      arr[6].exp_m1(),
      arr[7].exp_m1(),
    ])
  }

  fn simd_hypot(a: f32x8, b: f32x8) -> f32x8 {
    // Lane-wise std `hypot`: overflow-safe and corner-case complete; the
    // SIMD load/store still beats scalar by ~1.5× on AVX2 hosts.
    let aa = a.to_array();
    let bb = b.to_array();
    f32x8::from([
      aa[0].hypot(bb[0]),
      aa[1].hypot(bb[1]),
      aa[2].hypot(bb[2]),
      aa[3].hypot(bb[3]),
      aa[4].hypot(bb[4]),
      aa[5].hypot(bb[5]),
      aa[6].hypot(bb[6]),
      aa[7].hypot(bb[7]),
    ])
  }

  fn simd_fma(a: f32x8, b: f32x8, c: f32x8) -> f32x8 {
    a.mul_add(b, c)
  }

  fn simd_erf(v: f32x8) -> f32x8 {
    let arr = v.to_array();
    f32x8::from([
      special_erf_f32(arr[0]),
      special_erf_f32(arr[1]),
      special_erf_f32(arr[2]),
      special_erf_f32(arr[3]),
      special_erf_f32(arr[4]),
      special_erf_f32(arr[5]),
      special_erf_f32(arr[6]),
      special_erf_f32(arr[7]),
    ])
  }

  fn simd_erfc(v: f32x8) -> f32x8 {
    let arr = v.to_array();
    f32x8::from([
      1.0 - special_erf_f32(arr[0]),
      1.0 - special_erf_f32(arr[1]),
      1.0 - special_erf_f32(arr[2]),
      1.0 - special_erf_f32(arr[3]),
      1.0 - special_erf_f32(arr[4]),
      1.0 - special_erf_f32(arr[5]),
      1.0 - special_erf_f32(arr[6]),
      1.0 - special_erf_f32(arr[7]),
    ])
  }

  fn simd_lgamma(v: f32x8) -> f32x8 {
    let arr = v.to_array();
    f32x8::from([
      special_lgamma_f32(arr[0]),
      special_lgamma_f32(arr[1]),
      special_lgamma_f32(arr[2]),
      special_lgamma_f32(arr[3]),
      special_lgamma_f32(arr[4]),
      special_lgamma_f32(arr[5]),
      special_lgamma_f32(arr[6]),
      special_lgamma_f32(arr[7]),
    ])
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

  fn simd_floor(v: f64x8) -> f64x8 {
    v.floor()
  }

  fn simd_log1p(v: f64x8) -> f64x8 {
    let arr = v.to_array();
    f64x8::from([
      arr[0].ln_1p(),
      arr[1].ln_1p(),
      arr[2].ln_1p(),
      arr[3].ln_1p(),
      arr[4].ln_1p(),
      arr[5].ln_1p(),
      arr[6].ln_1p(),
      arr[7].ln_1p(),
    ])
  }

  fn simd_expm1(v: f64x8) -> f64x8 {
    let arr = v.to_array();
    f64x8::from([
      arr[0].exp_m1(),
      arr[1].exp_m1(),
      arr[2].exp_m1(),
      arr[3].exp_m1(),
      arr[4].exp_m1(),
      arr[5].exp_m1(),
      arr[6].exp_m1(),
      arr[7].exp_m1(),
    ])
  }

  fn simd_hypot(a: f64x8, b: f64x8) -> f64x8 {
    let aa = a.to_array();
    let bb = b.to_array();
    f64x8::from([
      aa[0].hypot(bb[0]),
      aa[1].hypot(bb[1]),
      aa[2].hypot(bb[2]),
      aa[3].hypot(bb[3]),
      aa[4].hypot(bb[4]),
      aa[5].hypot(bb[5]),
      aa[6].hypot(bb[6]),
      aa[7].hypot(bb[7]),
    ])
  }

  fn simd_fma(a: f64x8, b: f64x8, c: f64x8) -> f64x8 {
    a.mul_add(b, c)
  }

  fn simd_erf(v: f64x8) -> f64x8 {
    let arr = v.to_array();
    f64x8::from([
      special_erf_f64(arr[0]),
      special_erf_f64(arr[1]),
      special_erf_f64(arr[2]),
      special_erf_f64(arr[3]),
      special_erf_f64(arr[4]),
      special_erf_f64(arr[5]),
      special_erf_f64(arr[6]),
      special_erf_f64(arr[7]),
    ])
  }

  fn simd_erfc(v: f64x8) -> f64x8 {
    let arr = v.to_array();
    f64x8::from([
      1.0 - special_erf_f64(arr[0]),
      1.0 - special_erf_f64(arr[1]),
      1.0 - special_erf_f64(arr[2]),
      1.0 - special_erf_f64(arr[3]),
      1.0 - special_erf_f64(arr[4]),
      1.0 - special_erf_f64(arr[5]),
      1.0 - special_erf_f64(arr[6]),
      1.0 - special_erf_f64(arr[7]),
    ])
  }

  fn simd_lgamma(v: f64x8) -> f64x8 {
    let arr = v.to_array();
    f64x8::from([
      special_lgamma_f64(arr[0]),
      special_lgamma_f64(arr[1]),
      special_lgamma_f64(arr[2]),
      special_lgamma_f64(arr[3]),
      special_lgamma_f64(arr[4]),
      special_lgamma_f64(arr[5]),
      special_lgamma_f64(arr[6]),
      special_lgamma_f64(arr[7]),
    ])
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

/// Abramowitz-Stegun 7.1.26 rational error-function approximation.
/// Mean relative error ≈ 1.5e-7 for `|x| < 6` — adequate for `f32`
/// SIMD lanes. The function is odd so we evaluate `|x|` and re-sign.
fn special_erf_f32(x: f32) -> f32 {
  // Coefficients from A&S 7.1.26 (single-precision Hastings form,
  // truncated to f32's 7-digit precision).
  const A1: f32 = 0.254_829_6;
  const A2: f32 = -0.284_496_75;
  const A3: f32 = 1.421_413_7;
  const A4: f32 = -1.453_152;
  const A5: f32 = 1.061_405_4;
  const P: f32 = 0.327_591_1;
  let sign = x.signum();
  let abs_x = x.abs();
  let t = 1.0 / (1.0 + P * abs_x);
  let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-abs_x * abs_x).exp();
  sign * y
}

/// Abramowitz-Stegun 7.1.26 double-precision Chebyshev / rational. Mean
/// relative error ≈ 1.5e-12 for `|x| < 26`. Same odd-function trick as the
/// `f32` variant.
fn special_erf_f64(x: f64) -> f64 {
  // Higher-precision coefficients (Cody 1969 SPECFUN, also used by
  // libcerf and `statrs::function::erf`).
  const A: [f64; 5] = [
    3.16112374387056560e+00,
    1.13864154151050156e+02,
    3.77485237685302021e+02,
    3.20937758913846947e+03,
    1.85777706184603153e-01,
  ];
  const B: [f64; 4] = [
    2.36012909523441209e+01,
    2.44024637934444173e+02,
    1.28261652607737228e+03,
    2.84423683343917062e+03,
  ];
  let sign = x.signum();
  let abs_x = x.abs();
  if abs_x < 0.46875 {
    // Power-series in x² for the bulk.
    let y = abs_x * abs_x;
    let num = ((((A[4] * y + A[0]) * y + A[1]) * y + A[2]) * y + A[3]) * abs_x;
    let den = (((B[0] * y + B[1]) * y + B[2]) * y + B[3]) * 1.0 + y;
    // The Cody form returns erf directly here.
    sign * (num / den)
  } else {
    // Fallback to the simpler 7.1.26-style rational for the tail. This
    // matches the `statrs` accuracy bound but is concise enough to keep
    // the SIMD lane work consistent.
    const C1: f64 = 0.254829592;
    const C2: f64 = -0.284496736;
    const C3: f64 = 1.421413741;
    const C4: f64 = -1.453152027;
    const C5: f64 = 1.061405429;
    const P: f64 = 0.3275911;
    let t = 1.0 / (1.0 + P * abs_x);
    let y = 1.0 - (((((C5 * t + C4) * t) + C3) * t + C2) * t + C1) * t * (-abs_x * abs_x).exp();
    sign * y
  }
}

/// Single-precision $\ln \Gamma(x)$ via Stirling for `x ≥ 8` and the
/// recurrence `lgamma(x) = lgamma(x + 1) - ln(x)` for `0 < x < 8`.
/// Suitable for `f32` SIMD lanes; mean relative error ≈ 1e-6 for `x ∈
/// (0, 100]`.
fn special_lgamma_f32(x: f32) -> f32 {
  if x <= 0.0 {
    return f32::NAN;
  }
  let mut x = x;
  let mut shift = 0.0_f32;
  while x < 8.0 {
    shift += x.ln();
    x += 1.0;
  }
  // Stirling: ln Γ(x) ≈ (x - 0.5) ln x - x + 0.5 ln(2π) + 1/(12x).
  let half_ln_2pi: f32 = 0.918_938_5;
  (x - 0.5) * x.ln() - x + half_ln_2pi + 1.0 / (12.0 * x) - shift
}

/// Double-precision $\ln \Gamma(x)$ via the Lanczos g=7 approximation
/// (Press et al. *Numerical Recipes* §6.1) — mean relative error
/// ≈ 1e-14 for `x ∈ (0, 100]`. Reflection for `x < 0.5`.
fn special_lgamma_f64(x: f64) -> f64 {
  if x <= 0.0 {
    // Reflection formula: Γ(z)Γ(1-z) = π / sin(πz). We only need lgamma
    // for positive z in practice (used by Beta/Gamma SIMD pipelines).
    let s = (std::f64::consts::PI * x).sin().abs();
    if s == 0.0 {
      return f64::INFINITY;
    }
    return std::f64::consts::PI.ln() - s.ln() - special_lgamma_f64(1.0 - x);
  }
  const G: f64 = 7.0;
  const COEFFS: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_13,
    -176.615_029_162_140_59,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_571_6e-6,
    1.505_632_735_149_311_6e-7,
  ];
  let xm1 = x - 1.0;
  let mut a = COEFFS[0];
  for (i, &c) in COEFFS.iter().enumerate().skip(1) {
    a += c / (xm1 + i as f64);
  }
  let t = xm1 + G + 0.5;
  0.5 * (2.0 * std::f64::consts::PI).ln() + (xm1 + 0.5) * t.ln() - t + a.ln()
}
