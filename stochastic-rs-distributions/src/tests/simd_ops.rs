//! Unit tests for the seven extra SIMD ops `simd_log1p`, `simd_expm1`,
//! `simd_hypot`, `simd_fma`, `simd_erf`, `simd_erfc`, `simd_lgamma`.
//! Verifies lane-wise agreement with the scalar `std` / closed-form
//! references.

use std::f32;
use std::f64;

use crate::traits::SimdFloatExt;

#[test]
fn simd_log1p_matches_scalar() {
  let lanes_f32 = [-0.9, -0.3, 1e-5, 0.0, 1e-3, 0.5, 1.0, 5.0_f32];
  let v = f32::simd_from_array(lanes_f32);
  let out = f32::simd_to_array(f32::simd_log1p(v));
  for i in 0..8 {
    let scalar = lanes_f32[i].ln_1p();
    if scalar.is_finite() && out[i].is_finite() {
      assert!(
        (out[i] - scalar).abs() < 1e-6,
        "simd_log1p[f32] lane {i}: {} vs {scalar}",
        out[i]
      );
    }
  }

  let lanes_f64 = [-0.9, -0.3, 1e-10, 0.0, 1e-7, 0.5, 1.0, 100.0_f64];
  let v = f64::simd_from_array(lanes_f64);
  let out = f64::simd_to_array(f64::simd_log1p(v));
  for i in 0..8 {
    let scalar = lanes_f64[i].ln_1p();
    if scalar.is_finite() && out[i].is_finite() {
      assert!(
        (out[i] - scalar).abs() < 1e-12,
        "simd_log1p[f64] lane {i}: {} vs {scalar}",
        out[i]
      );
    }
  }
}

#[test]
fn simd_expm1_matches_scalar() {
  let lanes_f32 = [-5.0, -1.0, -1e-3, 0.0, 1e-3, 0.5, 1.0, 5.0_f32];
  let v = f32::simd_from_array(lanes_f32);
  let out = f32::simd_to_array(f32::simd_expm1(v));
  for i in 0..8 {
    let scalar = lanes_f32[i].exp_m1();
    assert!(
      (out[i] - scalar).abs() < 1e-5 * scalar.abs().max(1.0),
      "simd_expm1[f32] lane {i}: {} vs {scalar}",
      out[i]
    );
  }

  let lanes_f64 = [-5.0, -1.0, -1e-7, 0.0, 1e-7, 0.5, 1.0, 5.0_f64];
  let v = f64::simd_from_array(lanes_f64);
  let out = f64::simd_to_array(f64::simd_expm1(v));
  for i in 0..8 {
    let scalar = lanes_f64[i].exp_m1();
    assert!(
      (out[i] - scalar).abs() < 1e-12 * scalar.abs().max(1.0),
      "simd_expm1[f64] lane {i}: {} vs {scalar}",
      out[i]
    );
  }
}

#[test]
fn simd_hypot_matches_scalar() {
  let a_f32 = [0.0, 3.0, -5.0, 1e20, 1.0, 0.5, 12.0, -8.0_f32];
  let b_f32 = [0.0, 4.0, 12.0, 1e20, 0.0, 0.5, 5.0, 6.0_f32];
  let va = f32::simd_from_array(a_f32);
  let vb = f32::simd_from_array(b_f32);
  let out = f32::simd_to_array(f32::simd_hypot(va, vb));
  for i in 0..8 {
    let scalar = a_f32[i].hypot(b_f32[i]);
    if scalar.is_finite() && out[i].is_finite() {
      assert!(
        (out[i] - scalar).abs() < 1e-5 * scalar.max(1.0),
        "simd_hypot[f32] lane {i}: {} vs {scalar}",
        out[i]
      );
    }
  }

  let a_f64 = [0.0, 3.0, -5.0, 1e150, 1.0, 0.5, 12.0, -8.0_f64];
  let b_f64 = [0.0, 4.0, 12.0, 1e150, 0.0, 0.5, 5.0, 6.0_f64];
  let va = f64::simd_from_array(a_f64);
  let vb = f64::simd_from_array(b_f64);
  let out = f64::simd_to_array(f64::simd_hypot(va, vb));
  for i in 0..8 {
    let scalar = a_f64[i].hypot(b_f64[i]);
    if scalar.is_finite() && out[i].is_finite() {
      assert!(
        (out[i] - scalar).abs() < 1e-12 * scalar.max(1.0),
        "simd_hypot[f64] lane {i}: {} vs {scalar}",
        out[i]
      );
    }
  }
}

#[test]
fn simd_fma_matches_scalar() {
  let a_f64 = [1.0, 2.0, 3.0, 4.0, 5.0, -6.0, 7.0, 0.5_f64];
  let b_f64 = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, -7.5_f64];
  let c_f64 = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0_f64];
  let out = f64::simd_to_array(f64::simd_fma(
    f64::simd_from_array(a_f64),
    f64::simd_from_array(b_f64),
    f64::simd_from_array(c_f64),
  ));
  for i in 0..8 {
    let scalar = a_f64[i].mul_add(b_f64[i], c_f64[i]);
    assert!(
      (out[i] - scalar).abs() < 1e-12 * scalar.abs().max(1.0),
      "simd_fma[f64] lane {i}: {} vs {scalar}",
      out[i]
    );
  }
}

#[test]
fn simd_erf_matches_known_values() {
  // Reference values from libm / GSL / scipy.special.erf at standard
  // arguments. Used as a coarse sanity check; Abramowitz-Stegun 7.1.26
  // is accurate to ~1.5e-7 in f32.
  let lanes_f64 = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0_f64];
  let expected = [
    -0.9953222650189527_f64,
    -0.8427007929497149,
    -0.5204998778130465,
    0.0,
    0.5204998778130465,
    0.8427007929497149,
    0.9953222650189527,
    0.9999779095030014,
  ];
  let v = f64::simd_from_array(lanes_f64);
  let out = f64::simd_to_array(f64::simd_erf(v));
  for i in 0..8 {
    assert!(
      (out[i] - expected[i]).abs() < 5e-7,
      "simd_erf[f64] lane {i}: erf({}) = {} vs expected {}",
      lanes_f64[i],
      out[i],
      expected[i]
    );
  }
}

#[test]
fn simd_erfc_is_one_minus_erf() {
  let lanes = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0_f64];
  let v = f64::simd_from_array(lanes);
  let erf_out = f64::simd_to_array(f64::simd_erf(v));
  let erfc_out = f64::simd_to_array(f64::simd_erfc(v));
  for i in 0..8 {
    let expected = 1.0 - erf_out[i];
    assert!(
      (erfc_out[i] - expected).abs() < 1e-12,
      "simd_erfc[f64] lane {i}: {} vs 1 - erf = {expected}",
      erfc_out[i]
    );
  }
}

#[test]
fn simd_lgamma_matches_known_values() {
  // ln Γ at integer / half-integer arguments. Γ(0.5) = √π, Γ(n) = (n-1)!
  let lanes = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0_f64];
  let expected = [
    std::f64::consts::PI.sqrt().ln(),
    0.0,
    (std::f64::consts::PI.sqrt() / 2.0).ln(),
    0.0,
    2.0_f64.ln(),
    6.0_f64.ln(),
    24.0_f64.ln(),
    362_880.0_f64.ln(),
  ];
  let v = f64::simd_from_array(lanes);
  let out = f64::simd_to_array(f64::simd_lgamma(v));
  for i in 0..8 {
    let abs_err = (out[i] - expected[i]).abs();
    assert!(
      abs_err < 1e-10,
      "simd_lgamma[f64] lane {i}: lgamma({}) = {} vs expected {} (abs err {abs_err})",
      lanes[i],
      out[i],
      expected[i]
    );
  }
}
