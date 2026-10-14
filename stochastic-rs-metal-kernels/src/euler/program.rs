//! Postfix coefficient programs encoded as two lengths followed by opcode/value pairs.

use metal_oxide_device::ReadBuffer;

use super::vocabulary;

/// Evaluates up to two programs; absent outputs remain zero.
///
/// # Safety
/// program_count is at most two. The buffer contains the host's validated
/// encoding: two integral lengths and the complete opcode/value pairs for each program.
/// Each program has at most 62 instructions, uses opcodes 0 through 15, needs
/// at most eight stack slots, never underflows, and finishes with one value.
#[allow(
  clippy::manual_range_contains,
  reason = "stack bounds stay explicit for device assertion proofs"
)]
pub(super) unsafe fn evaluate(
  buffer: ReadBuffer<f32>,
  program_count: u32,
  time: f32,
  state: f32,
) -> [f32; 2] {
  let mut values = [0.0; 2];
  for program in 0..program_count.min(2) {
    // SAFETY: the validated encoding contains both lengths.
    let length = unsafe { buffer.load_unchecked(program).to_int_unchecked::<u32>() };
    if length > 62 {
      return values;
    }
    let base = if program == 0 {
      2
    } else {
      // SAFETY: both programs share the first length in the encoding header.
      let first_length = unsafe { buffer.load_unchecked(0).to_int_unchecked::<u32>() };
      if first_length > 62 {
        return values;
      }
      2 + 2 * first_length
    };
    // SAFETY: slots below `depth` were pushed; the result slot starts at zero.
    let mut stack: [f32; 8] = unsafe { metal_oxide_device::uninit_array() };
    stack[0] = 0.0;
    let mut depth = 0_usize;
    for instruction in 0..length.min(62) {
      // SAFETY: base and instruction select a pair in the validated program.
      let (code, value) = unsafe {
        (
          buffer
            .load_unchecked(base + 2 * instruction)
            .to_int_unchecked::<u32>(),
          buffer.load_unchecked(base + 2 * instruction + 1),
        )
      };
      if code <= 2 {
        if depth < 8 {
          stack[depth] = match code {
            0 => time,
            1 => state,
            _ => value,
          };
          depth += 1;
        }
      } else if code <= 9 {
        if depth >= 2 && depth <= 8 {
          let left = stack[depth - 2];
          let right = stack[depth - 1];
          stack[depth - 2] = binary(code, left, right);
          depth -= 1;
        }
      } else if code <= 15 && depth >= 1 && depth <= 8 {
        let top = stack[depth - 1];
        stack[depth - 1] = unary(code, top);
      }
    }
    if program == 0 {
      values[0] = stack[0];
    } else {
      values[1] = stack[0];
    }
  }
  values
}

fn binary(code: u32, left: f32, right: f32) -> f32 {
  match code {
    3 => left + right,
    4 => left - right,
    5 => left * right,
    6 => left / right,
    7 => vocabulary::pow(left, right),
    // The recorded Metal results select the numeric operand when one is NaN.
    8 => left.max(right),
    _ => left.min(right),
  }
}

fn unary(code: u32, value: f32) -> f32 {
  match code {
    10 => vocabulary::negate(value),
    11 => vocabulary::sqrt(value),
    12 => vocabulary::exp(value),
    13 => vocabulary::ln(value),
    14 => vocabulary::abs(value),
    _ => vocabulary::tanh(value),
  }
}
