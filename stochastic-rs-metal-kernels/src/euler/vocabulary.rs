//! Concrete f32 spellings of the authoritative family expression vocabulary.

use metal_oxide_device::math;

pub(crate) use math::{atan, exp, sin, sqrt, tanh};

pub(crate) fn ln(value: f32) -> f32 {
  math::ln(value)
}

pub(crate) fn pow(value: f32, exponent: f32) -> f32 {
  math::powf(value, exponent)
}

pub(crate) fn abs(value: f32) -> f32 {
  value.abs()
}

pub(crate) fn negate(value: f32) -> f32 {
  -value
}

pub(crate) fn recip(value: f32) -> f32 {
  1.0 / value
}

pub(crate) fn positive(value: f32) -> f32 {
  if value > 0.0 { value } else { 0.0 }
}

pub(crate) fn max(left: f32, right: f32) -> f32 {
  if left > right { left } else { right }
}

pub(crate) fn min(left: f32, right: f32) -> f32 {
  if left < right { left } else { right }
}

pub(crate) fn less(left: f32, right: f32) -> f32 {
  if left < right { 1.0 } else { 0.0 }
}

pub(crate) fn leq(left: f32, right: f32) -> f32 {
  if left <= right { 1.0 } else { 0.0 }
}

pub(crate) fn geq(left: f32, right: f32) -> f32 {
  if left >= right { 1.0 } else { 0.0 }
}

pub(crate) fn pick(condition: f32, yes: f32, no: f32) -> f32 {
  if condition != 0.0 { yes } else { no }
}

pub(crate) fn lit(value: f32) -> f32 {
  value
}
