//! The Philox port against Random123's known answers, and the host forms of the device draws.

use std::f32::consts::LN_2;
use std::f32::consts::TAU;

use crate::euler::philox::LN_2_23;
use crate::euler::philox::key;
use crate::euler::philox::philox4x32_10;
use crate::euler::philox::u01_f32;
use crate::euler::philox::u01_f32_zero_cell;
use crate::euler::philox::u01_f64;
use crate::traits::RealExt;

/// `Exp(1)` from two words as the `f32` kernels draw it, with an accurate logarithm.
pub(crate) fn exp1_f32(a: u32, b: u32) -> f32 {
  let e = if a >> 9 == 0 {
    LN_2_23 - u01_f32_zero_cell(a, b).ln()
  } else {
    -u01_f32(a).ln()
  };
  e.max_or_nan(0.0)
}

/// A standard normal from one call's words as the `f32` kernels draw it; the fourth is unused.
pub(crate) fn normal_f32(w: [u32; 4]) -> f32 {
  (2.0 * exp1_f32(w[0], w[1])).sqrt() * (TAU * u01_f32(w[2])).cos()
}

/// A standard normal from one call's words as the `f64` kernels draw it.
pub(crate) fn normal_f64(w: [u32; 4]) -> f64 {
  let e = (-u01_f64(w[0], w[1]).ln()).max_or_nan(0.0);
  (2.0 * e).sqrt() * (std::f64::consts::TAU * u01_f64(w[2], w[3])).cos()
}

#[test]
fn the_random123_known_answers() {
  assert_eq!(
    philox4x32_10([0; 4], [0; 2]),
    [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
  );
  assert_eq!(
    philox4x32_10([u32::MAX; 4], [u32::MAX; 2]),
    [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
  );
  assert_eq!(
    philox4x32_10(
      [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
      [0xa409_3822, 0x299f_31d0]
    ),
    [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
  );
}

#[test]
fn the_key_is_the_whole_seed() {
  assert_eq!(key(0x0123_4567_89ab_cdef), [0x89ab_cdef, 0x0123_4567]);
  assert_ne!(key(0), key(0x1_0000_0001));
}

#[test]
fn the_uniforms_are_strictly_inside_the_unit_interval() {
  let step = f32::EPSILON;
  assert_eq!(u01_f32(0), 0.5 * step);
  assert_eq!(u01_f32(u32::MAX), 1.0 - 0.5 * step);
  assert!(u01_f32(0) > 0.0 && u01_f32(u32::MAX) < 1.0);
  for j in [1u32, 2, 3, 1 << 22, (1 << 23) - 2] {
    assert_eq!(u01_f32(j << 9), (2 * j + 1) as f32 * 0.5 * step, "j = {j}");
    assert_eq!(
      u01_f32((j << 9) | 0x1ff),
      u01_f32(j << 9),
      "the low bits are dropped"
    );
  }
  assert_eq!(u01_f64(0, 0), 0.5 * f64::EPSILON);
  assert_eq!(u01_f64(u32::MAX, u32::MAX), 1.0 - 0.5 * f64::EPSILON);
  assert_eq!(
    u01_f64(3, 5 << 12),
    ((3u64 << 20 | 5) * 2 + 1) as f64 * 0.5 * f64::EPSILON
  );
}

#[test]
fn the_exponential_extension_continues_the_tail() {
  let e_top = exp1_f32(0x1ff, u32::MAX);
  let e_first = exp1_f32(1 << 9, 0);
  assert!(e_top > e_first && e_top - 23.0 * LN_2 < 1e-6);
  assert!(
    (exp1_f32(0, 0) - 47.0 * LN_2).abs() < 1e-5,
    "two zero words reach 47 ln 2"
  );
  assert!(exp1_f32(u32::MAX, 0) >= 0.0 && exp1_f32(u32::MAX, u32::MAX) >= 0.0);
}

/// Over four million word pairs, half in the exponential's zero cell, an `f32` uniform and the
/// uniform under an `f32` exponential are the `f64` uniform cut to their resolution.
#[test]
fn an_f32_draw_is_the_truncation_of_the_f64_draw() {
  let cell = |u: f64, width: f64| ((u / width).floor() + 0.5) * width;
  let width = f64::from(f32::EPSILON);
  for i in 0..1u32 << 20 {
    let w = philox4x32_10([i, 0, 0, 0], key(0x5eed));
    for (a, b) in [
      (w[0], w[1]),
      (w[2], w[3]),
      (w[0] & 0x1ff, w[1]),
      (w[2] & 0x1ff, w[3]),
    ] {
      let u = u01_f64(a, b);
      assert_eq!(f64::from(u01_f32(a)), cell(u, width), "{a:#x} {b:#x}");
      let under = if u < width {
        cell(u, width * width)
      } else {
        cell(u, width)
      };
      let e = f64::from(exp1_f32(a, b));
      assert!(
        (e + under.ln()).abs() <= 4e-6 * (1.0 + e),
        "{a:#x} {b:#x}: {e} against {}",
        -under.ln()
      );
    }
  }
}

#[test]
fn a_call_gives_the_normal_its_three_words() {
  let w = philox4x32_10([7, 0, 3, 0], key(0x0123_4567_89ab_cdef));
  assert_eq!(w, [0xfd6c_aef0, 0x6758_1926, 0x7e30_6f1a, 0x0c01_d0c0]);
  let z = normal_f32(w);
  let r = (-2.0 * u01_f32(w[0]).ln()).sqrt();
  assert!((z - r * (TAU * u01_f32(w[2])).cos()).abs() < 1e-6);
  assert!(normal_f64(w).is_finite() && (normal_f64(w) - z as f64).abs() < 1e-5);
}
