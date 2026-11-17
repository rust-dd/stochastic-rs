//! The frame's draws on Philox: a normal from one call, two uniforms or exponentials from another.

use metal_oxide_device::math;

use super::philox::{
  GAMMA_BOOST, LN_2_23, SITE_COUNT, SITE_GAMMA, SITE_NOISE, philox4x32_10, u01_f32,
  u01_f32_zero_cell,
};
use super::vocabulary;

/// `π/2` in units of 2⁻⁵⁶, so `π/2 − |x|` is exact in integers, which fast math cannot reorder.
const HALF_PI_2_56: u64 = 0x0192_1FB5_4442_D184;

/// `step` as the compiler's numerical proofs see any value, so they analyze a step's draws once,
/// not once per step; `path ^ path` is zero, which the GPU compiler folds away.
#[allow(clippy::eq_op, reason = "a zero the proofs cannot see")]
pub(crate) fn opaque_step(step: u32, path: u32) -> u32 {
  step ^ (path ^ path)
}

/// The four words at `(path, step, site, index)`.
pub(crate) fn words(path: u32, step: u32, site: u32, index: u32, key: [u32; 2]) -> [u32; 4] {
  philox4x32_10([path, step, site, index], key)
}

/// `Exp(1)` from two words, its zero cell refined in `u01_f64`'s bit order; the fast log's error
/// can take `−ln u` below zero near `u = 1`, where the normal's root would be NaN.
pub(crate) fn exponential(first: u32, second: u32) -> f32 {
  let e = if first >> 9 == 0 {
    LN_2_23 - math::ln(u01_f32_zero_cell(first, second))
  } else {
    -math::ln(u01_f32(first))
  };
  if e < 0.0 { 0.0 } else { e }
}

/// [`exponential`] on the family log, positive at every uniform: the stable law divides by it,
/// where a zero would make the jump infinite.
pub(crate) fn exponential_accurate(first: u32, second: u32) -> f32 {
  let e = if first >> 9 == 0 {
    LN_2_23 - vocabulary::ln(u01_f32_zero_cell(first, second))
  } else {
    -vocabulary::ln(u01_f32(first))
  };
  if e < 0.0 { 0.0 } else { e }
}

/// `cos` to two ulp within 2 of zero, where the fast cosine's absolute error of 2⁻¹³ can outweigh
/// a result near ±π/2; past 2, or for NaN, the fast cosine.
pub(crate) fn cos_accurate(value: f32) -> f32 {
  let magnitude = value.abs();
  if magnitude <= core::f32::consts::FRAC_PI_4 {
    let s = magnitude * magnitude;
    let tail = 0.041_666_668 + s * (-0.001_388_888_9 + s * (2.480_158_8e-5 + s * -2.755_732e-7));
    1.0 + s * (-0.5 + s * tail)
  } else if magnitude < 2.0 {
    let bits = magnitude.to_bits();
    let significand = ((bits & 0x007F_FFFF) | 0x0080_0000) as u64;
    let scaled = if bits >= 0x3F80_0000 {
      significand << 33
    } else {
      significand << 32
    };
    let r = if scaled <= HALF_PI_2_56 {
      (HALF_PI_2_56 - scaled) as f32 * 1.387_778_8e-17
    } else {
      -((scaled - HALF_PI_2_56) as f32 * 1.387_778_8e-17)
    };
    let s = r * r;
    let tail = 0.008_333_334 + s * (-0.000_198_412_7 + s * 2.755_731_9e-6);
    r * (1.0 + s * (-0.166_666_67 + s * tail))
  } else {
    math::cos(value)
  }
}

pub(crate) fn normal(words: [u32; 4]) -> f32 {
  math::sqrt(2.0 * exponential(words[0], words[1]))
    * math::cos(core::f32::consts::TAU * u01_f32(words[2]))
}

/// The step's standard normal for noise component `component`.
pub(crate) fn noise(path: u32, step: u32, component: u32, key: [u32; 2]) -> f32 {
  normal(words(path, step, SITE_NOISE + component, 0, key))
}

pub(crate) fn poisson(path: u32, step: u32, key: [u32; 2], lambda: f32, dt: f32) -> f32 {
  let threshold = math::exp(-lambda * dt);
  let mut product = 1.0;
  let mut count = 0u32;
  let mut pair = [0u32; 4];
  for attempt in 0u32..64 {
    let uniform = if attempt & 1 == 0 {
      pair = words(path, step, SITE_COUNT, attempt >> 1, key);
      u01_f32(pair[0])
    } else {
      u01_f32(pair[2])
    };
    product *= uniform;
    if product <= threshold {
      break;
    }
    count += 1;
  }
  count as f32
}

pub(crate) fn gamma(
  path: u32,
  step: u32,
  key: [u32; 2],
  which: u32,
  shape: f32,
  scale: f32,
) -> f32 {
  let positive_shape = shape > 0.0;
  if !positive_shape {
    return 0.0;
  }
  let site = SITE_GAMMA + which;
  let mut boosted_shape = shape;
  let mut boost = 1.0;
  if shape < 1.0 {
    let value = u01_f32(words(path, step, site, GAMMA_BOOST, key)[0]);
    boost = math::powf(value, 1.0 / shape);
    boosted_shape += 1.0;
  }
  let d = boosted_shape - 1.0 / 3.0;
  let c = 1.0 / math::sqrt(9.0 * d);
  let mut value = d;
  for attempt in 0u32..24 {
    let z = normal(words(path, step, site, 2 * attempt, key));
    let mut v = 1.0 + c * z;
    v = v * v * v;
    if v > 0.0 {
      let u = u01_f32(words(path, step, site, 2 * attempt + 1, key)[0]);
      value = d * v;
      if math::ln(u) < 0.5 * z * z + d - d * v + d * math::ln(v) {
        break;
      }
    }
  }
  scale * boost * value
}
