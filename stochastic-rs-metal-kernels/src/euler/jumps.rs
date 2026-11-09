//! Euler jump laws of the C frame, including their fixed 64-jump truncation.

use metal_oxide_device::math;

use super::draws::{normal, open, uniform};

fn pair(key: u32, term: u32, seed: u32) -> (f32, f32) {
  let offset = term.wrapping_mul(40_503);
  (
    uniform(
      key,
      2_654_435_761u32.wrapping_add(offset),
      seed,
      2_654_435_761,
    ),
    uniform(
      key,
      668_265_263u32.wrapping_add(offset),
      seed,
      2_654_435_761,
    ),
  )
}

fn exponential(up: f32, uniform: f32, a: f32, b: f32, c: f32) -> f32 {
  let value = -math::ln(1.0 - uniform);
  if up < a { value / b } else { -(value / c) }
}

pub(crate) fn draw(key: u32, seed: u32, law: u32, a: f32, b: f32, c: f32, count: f32) -> f32 {
  // Separate return expressions preserve the seeded paths' fast-math rounding.
  if law == 1 {
    let first = open(uniform(key, 1_103_515_245, seed, 2_654_435_761));
    let second = uniform(key, 1_013_904_223, seed, 2_654_435_761);
    let z = normal(first, second);
    return a * count + b * math::sqrt(count) * z;
  }
  if law == 6 {
    let first = open(uniform(key, 1_103_515_245, seed, 2_654_435_761));
    let second = uniform(key, 1_013_904_223, seed, 2_654_435_761);
    let z = normal(first, second);
    return a + b * z;
  }
  if law == 7 {
    let values = pair(key, 0, seed);
    return exponential(values.0, values.1, a, b, c);
  }
  if law == 9 {
    let first = open(uniform(key, 1_103_515_245, seed, 2_654_435_761));
    let second = open(uniform(key, 1_013_904_223, seed, 2_654_435_761));
    let angle = (first - 0.5) * core::f32::consts::PI;
    let exponential = -math::ln(second);
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
    2 => compound::<2>(key, seed, a, b, c, count),
    3 => compound::<3>(key, seed, a, b, c, count),
    4 => compound::<4>(key, seed, a, b, c, count),
    5 => compound::<5>(key, seed, a, b, c, count),
    8 => compound::<8>(key, seed, a, b, c, count),
    _ => 0.0,
  }
}

fn compound<const LAW: u32>(key: u32, seed: u32, a: f32, b: f32, c: f32, count: f32) -> f32 {
  let mut sum = 0.0;
  let mut product = 1.0;
  for term in 0u32..64 {
    if term as f32 >= count {
      break;
    }
    let values = pair(key, term, seed);
    if LAW == 2 {
      sum += exponential(values.0, values.1, a, b, c);
    }
    if LAW == 3 {
      let value = a * math::powf(open(values.0), b);
      if values.1 <= math::exp(-c * value) {
        sum += value;
      }
    }
    if LAW == 4 {
      let z = normal(open(values.0), values.1);
      product *= 1.0 + a + b * z;
    }
    if LAW == 5 {
      product *= 1.0 + exponential(values.0, values.1, a, b, c);
    }
    if LAW == 8 {
      sum += if values.0 < 0.5 { a } else { -a };
    }
  }
  if LAW == 4 || LAW == 5 {
    product - 1.0
  } else {
    sum
  }
}
