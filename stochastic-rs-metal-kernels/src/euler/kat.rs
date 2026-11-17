//! The generator as the GPU computes it, for the host to hold against the port and Random123.

use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel};

use super::draws::{exponential, normal};
use super::philox::{philox4x32_10, u01_f32};

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
