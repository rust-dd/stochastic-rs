//! Per-path series arrivals and inverse-clock tables.

use metal_oxide_device::math;

use super::args::EulerArgs;
use super::draws::{open, uniform};
use super::families::Inputs;
use super::vocabulary::negate;

pub(super) fn key(path: u32, term: u32) -> u32 {
  path.wrapping_mul(2_654_435_761) ^ term.wrapping_mul(40_503) ^ 3_266_489_917
}

/// # Safety
/// `steps` is nonzero, so the scaled unit uniform lies in `[0, steps - 1]`.
pub(super) unsafe fn cell(key: u32, seed: u32, steps: u32) -> u32 {
  let ratio = uniform(key, 374_761_393, seed, 2_654_435_761) * (steps - 1) as f32;
  // SAFETY: the ratio is finite, nonnegative and at most steps - 1.
  let mut cell = unsafe { ratio.to_int_unchecked::<u32>() };
  if (cell as f32) < ratio {
    cell += 1;
  }
  if cell < 1 {
    cell = 1;
  }
  if cell > steps - 1 {
    cell = steps - 1;
  }
  cell
}

/// Exponential gap between consecutive arrivals of the series clock.
pub(super) fn gap(key: u32, seed: u32) -> f32 {
  negate(math::ln(open(uniform(
    key,
    2_654_435_769,
    seed,
    2_654_435_761,
  ))))
}

pub(super) fn term(mut input: Inputs, key: u32, seed: u32, arrival: f32) -> Inputs {
  input.arrival = arrival;
  input.exponential = negate(math::ln(open(uniform(
    key,
    2_246_822_507,
    seed,
    2_654_435_761,
  ))));
  input.series_uniform = [
    uniform(key, 3_266_489_909, seed, 2_654_435_761),
    uniform(key, 668_265_263, seed, 2_654_435_761),
  ];
  input
}

/// Table extent of the first attempt; later attempts double it.
pub(super) fn table_extent(args: EulerArgs, horizon: f32) -> f32 {
  if args.table_u0 <= 0.0 {
    if horizon > 1.0 { horizon } else { 1.0 }
  } else {
    args.table_u0
  }
}

/// Uniforms of one table node in one attempt.
pub(super) fn table_uniforms(path: u32, attempt: u32, node: u32, seed: u32) -> [f32; 2] {
  let key = path.wrapping_mul(2_654_435_761)
    ^ attempt.wrapping_mul(668_265_263)
    ^ node.wrapping_mul(40_503)
    ^ 2_246_822_519;
  [
    uniform(key, 2_654_435_769, seed, 2_654_435_761),
    uniform(key, 3_266_489_909, seed, 2_654_435_761),
  ]
}
