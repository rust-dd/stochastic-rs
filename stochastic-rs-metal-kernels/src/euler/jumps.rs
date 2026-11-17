//! Euler jump laws of the C frame, including their fixed 64-jump truncation.

use metal_oxide_device::math;

use super::draws::{exponential, normal, words};
use super::philox::{SITE_SIZES, u01_f32};

fn sized(up: f32, value: f32, a: f32, b: f32, c: f32) -> f32 {
  if up < a { value / b } else { -(value / c) }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn draw(
  path: u32,
  step: u32,
  key: [u32; 2],
  law: u32,
  a: f32,
  b: f32,
  c: f32,
  count: f32,
) -> f32 {
  if law == 1 {
    let z = normal(words(path, step, SITE_SIZES, 0, key));
    return a * count + b * math::sqrt(count) * z;
  }
  if law == 6 {
    let z = normal(words(path, step, SITE_SIZES, 0, key));
    return a + b * z;
  }
  if law == 7 {
    let pair = words(path, step, SITE_SIZES, 0, key);
    return sized(u01_f32(pair[0]), exponential(pair[2], pair[3]), a, b, c);
  }
  if law == 9 {
    let pair = words(path, step, SITE_SIZES, 0, key);
    let angle = (u01_f32(pair[0]) - 0.5) * core::f32::consts::PI;
    let exponential = exponential(pair[2], pair[3]);
    let inverse = 1.0 / a;
    let mut ratio = math::cos(angle - a * angle) / exponential;
    if ratio < 1.0e-30 {
      ratio = 1.0e-30;
    }
    let stable = math::sin(a * angle) / math::powf(math::cos(angle), inverse)
      * math::powf(ratio, (1.0 - a) * inverse);
    return b * math::powf(count, inverse) * stable;
  }
  match law {
    2 => compound::<2>(path, step, key, a, b, c, count),
    3 => compound::<3>(path, step, key, a, b, c, count),
    4 => compound::<4>(path, step, key, a, b, c, count),
    5 => compound::<5>(path, step, key, a, b, c, count),
    8 => compound::<8>(path, step, key, a, b, c, count),
    _ => 0.0,
  }
}

#[allow(clippy::too_many_arguments)]
fn compound<const LAW: u32>(
  path: u32,
  step: u32,
  key: [u32; 2],
  a: f32,
  b: f32,
  c: f32,
  count: f32,
) -> f32 {
  let mut sum = 0.0;
  let mut product = 1.0;
  for term in 0u32..64 {
    if term as f32 >= count {
      break;
    }
    let pair = words(path, step, SITE_SIZES, term, key);
    if LAW == 2 {
      sum += sized(u01_f32(pair[0]), exponential(pair[2], pair[3]), a, b, c);
    }
    if LAW == 3 {
      let value = a * math::powf(u01_f32(pair[0]), b);
      if u01_f32(pair[2]) <= math::exp(-c * value) {
        sum += value;
      }
    }
    if LAW == 4 {
      product *= 1.0 + a + b * normal(pair);
    }
    if LAW == 5 {
      product *= 1.0 + sized(u01_f32(pair[0]), exponential(pair[2], pair[3]), a, b, c);
    }
    if LAW == 8 {
      sum += if u01_f32(pair[0]) < 0.5 { a } else { -a };
    }
  }
  if LAW == 4 || LAW == 5 {
    product - 1.0
  } else {
    sum
  }
}
