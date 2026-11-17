//! The generator, the family log and the stable law's cosine and jump as the GPU computes them,
//! for the host to hold against the port, Random123 and `f64`.

use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel};

use super::draws::{cos_accurate, exponential, normal};
use super::jumps::stable;
use super::philox::{philox4x32_10, u01_f32};
use super::vocabulary;

/// # Safety
/// `input` holds `cases` six-word cases (counter, key), `words` four words a case, `draws` three.
#[kernel]
pub unsafe fn philox_kat(
  input: ReadBuffer<u32>,
  words: WriteBuffer<u32>,
  draws: WriteBuffer<f32>,
  cases: u32,
) {
  let case = crate::index();
  if case >= cases {
    return;
  }
  // SAFETY: the caller sized every buffer for `cases` cases, and `case < cases`.
  unsafe {
    let at = 6 * case;
    let counter = [
      input.load_unchecked(at),
      input.load_unchecked(at + 1),
      input.load_unchecked(at + 2),
      input.load_unchecked(at + 3),
    ];
    let key = [input.load_unchecked(at + 4), input.load_unchecked(at + 5)];
    let out = philox4x32_10(counter, key);
    words.store_unchecked(4 * case, out[0]);
    words.store_unchecked(4 * case + 1, out[1]);
    words.store_unchecked(4 * case + 2, out[2]);
    words.store_unchecked(4 * case + 3, out[3]);
    draws.store_unchecked(3 * case, u01_f32(counter[0]));
    draws.store_unchecked(3 * case + 1, exponential(counter[0], counter[1]));
    draws.store_unchecked(3 * case + 2, normal(out));
  }
}

/// # Safety
/// `values` and `logs` hold `cases` values each.
#[kernel]
pub unsafe fn log_kat(values: ReadBuffer<f32>, logs: WriteBuffer<f32>, cases: u32) {
  let case = crate::index();
  if case >= cases {
    return;
  }
  // SAFETY: the caller sized both buffers for `cases` values, and `case < cases`.
  unsafe {
    logs.store_unchecked(case, vocabulary::ln(values.load_unchecked(case)));
  }
}

/// # Safety
/// `values` and `cosines` hold `cases` values each.
#[kernel]
pub unsafe fn cos_kat(values: ReadBuffer<f32>, cosines: WriteBuffer<f32>, cases: u32) {
  let case = crate::index();
  if case >= cases {
    return;
  }
  // SAFETY: the caller sized both buffers for `cases` values, and `case < cases`.
  unsafe {
    cosines.store_unchecked(case, cos_accurate(values.load_unchecked(case)));
  }
}

/// # Safety
/// `words` holds four words a case, `laws` three values a case (`α`, scale, count), `jumps` one.
#[kernel]
pub unsafe fn stable_kat(
  words: ReadBuffer<u32>,
  laws: ReadBuffer<f32>,
  jumps: WriteBuffer<f32>,
  cases: u32,
) {
  let case = crate::index();
  if case >= cases {
    return;
  }
  // SAFETY: the caller sized every buffer for `cases` cases, and `case < cases`.
  unsafe {
    let at = 4 * case;
    let pair = [
      words.load_unchecked(at),
      words.load_unchecked(at + 1),
      words.load_unchecked(at + 2),
      words.load_unchecked(at + 3),
    ];
    let law = 3 * case;
    let (alpha, scale, count) = (
      laws.load_unchecked(law),
      laws.load_unchecked(law + 1),
      laws.load_unchecked(law + 2),
    );
    jumps.store_unchecked(case, stable(pair, alpha, scale, count));
  }
}
