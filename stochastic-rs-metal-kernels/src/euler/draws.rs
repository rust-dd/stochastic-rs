//! Counter draws keep the C frame's salts, clamps and attempt limits.

use metal_oxide_device::math;

fn avalanche(mut word: u32) -> u32 {
  word ^= word >> 16;
  word = word.wrapping_mul(2_246_822_519);
  word ^= word >> 13;
  word = word.wrapping_mul(3_266_489_917);
  word ^ (word >> 16)
}

pub(crate) fn cell(path: u32, step: u32) -> u32 {
  avalanche(
    path.wrapping_mul(2_654_435_761) ^ step.wrapping_add(2_246_822_519).wrapping_mul(2_246_822_519),
  )
}

pub(crate) fn uniform(key: u32, salt: u32, seed: u32, multiplier: u32) -> f32 {
  avalanche((key ^ salt) ^ seed.wrapping_mul(multiplier)) as f32 * 2.328_306_4e-10
}

pub(crate) fn open(uniform: f32) -> f32 {
  uniform * 0.999_998 + 1.0e-6
}

pub(crate) fn normal(first: f32, second: f32) -> f32 {
  math::sqrt(-2.0 * math::ln(first)) * math::cos(core::f32::consts::TAU * second)
}

pub(crate) fn noise(key: u32, component: u32, seed: u32) -> f32 {
  let first = uniform(
    key,
    2_654_435_769u32.wrapping_add(component.wrapping_mul(2_654_435_761)),
    seed,
    2_654_435_761,
  );
  let second = uniform(
    key,
    1_442_695_041u32.wrapping_add(component.wrapping_mul(1_013_904_223)),
    seed,
    668_265_263,
  );
  normal(open(first), second)
}

pub(crate) fn poisson(key: u32, seed: u32, lambda: f32, dt: f32) -> f32 {
  let threshold = math::exp(-lambda * dt);
  let mut product = 1.0;
  let mut count = 0u32;
  let mut attempt = 0u32;
  while attempt < 64 {
    product *= uniform(
      key,
      2_166_136_261u32.wrapping_add(attempt.wrapping_mul(16_777_619)),
      seed,
      374_761_393,
    );
    if product <= threshold {
      break;
    }
    count += 1;
    attempt += 1;
  }
  count as f32
}

pub(crate) fn gamma(key: u32, seed: u32, which: u32, shape: f32, scale: f32) -> f32 {
  let positive_shape = shape > 0.0;
  if !positive_shape {
    return 0.0;
  }
  let mut boosted_shape = shape;
  let mut boost = 1.0;
  if shape < 1.0 {
    let value = open(uniform(
      key,
      2_246_822_519u32.wrapping_add(which.wrapping_mul(97)),
      seed,
      2_654_435_761,
    ));
    boost = math::powf(value, 1.0 / shape);
    boosted_shape += 1.0;
  }
  let d = boosted_shape - 1.0 / 3.0;
  let c = 1.0 / math::sqrt(9.0 * d);
  let mut value = d;
  let mut attempt = 0u32;
  while attempt < 24 {
    let offset = which
      .wrapping_mul(7_919)
      .wrapping_add(attempt.wrapping_mul(104_729));
    let first = open(uniform(
      key,
      1_103_515_245u32.wrapping_add(offset),
      seed,
      2_654_435_761,
    ));
    let second = open(uniform(
      key,
      1_013_904_223u32.wrapping_add(offset),
      seed,
      2_654_435_761,
    ));
    let z = normal(first, second);
    let mut v = 1.0 + c * z;
    v = v * v * v;
    if v > 0.0 {
      let u = open(uniform(
        key,
        668_265_263u32.wrapping_add(offset),
        seed,
        2_654_435_761,
      ));
      value = d * v;
      if math::ln(u) < 0.5 * z * z + d - d * v + d * math::ln(v) {
        break;
      }
    }
    attempt += 1;
  }
  scale * boost * value
}
