//! Per-path series arrivals and inverse-clock tables.

use super::args::EulerArgs;
use super::draws::{exponential, words};
use super::families::Inputs;
use super::philox::{SITE_SERIES, SITE_SERIES_MARKS, SITE_SERIES_TIME, SITE_TABLE, u01_f32};

/// The term's arrival gap (the first two words) and its exponential (the last two).
pub(super) fn arrival(path: u32, term: u32, key: [u32; 2]) -> [u32; 4] {
  words(path, 0, SITE_SERIES, term, key)
}

pub(super) fn gap(arrival: [u32; 4]) -> f32 {
  exponential(arrival[0], arrival[1])
}

/// # Safety
/// `steps` is nonzero, so the scaled unit uniform lies in `(0, steps - 1)`.
pub(super) unsafe fn cell(path: u32, term: u32, key: [u32; 2], steps: u32) -> u32 {
  let uniform = u01_f32(words(path, 0, SITE_SERIES_TIME, term, key)[0]);
  let ratio = uniform * (steps - 1) as f32;
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

pub(super) fn term(
  mut input: Inputs,
  path: u32,
  term: u32,
  key: [u32; 2],
  arrival: [u32; 4],
  clock: f32,
) -> Inputs {
  input.arrival = clock;
  input.exponential = exponential(arrival[2], arrival[3]);
  let marks = words(path, 0, SITE_SERIES_MARKS, term, key);
  input.series_uniform = [u01_f32(marks[0]), u01_f32(marks[2])];
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
pub(super) fn table_uniforms(path: u32, attempt: u32, node: u32, key: [u32; 2]) -> [f32; 2] {
  let pair = words(path, attempt, SITE_TABLE, node, key);
  [u01_f32(pair[0]), u01_f32(pair[2])]
}
