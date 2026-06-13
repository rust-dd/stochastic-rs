//! Uninitialised-allocation helpers — the single home of the
//! `uninit → write-all → assume_init` unsafe pattern.
//!
//! Samplers allocate their outputs through these instead of `Array1::zeros`:
//! the zeroing pass costs 3.5–6% of a full `sample()` (more around
//! cache-boundary sizes) and every element is overwritten anyway.

use ndarray::Array1;
use ndarray::Array2;

use crate::traits::FloatExt;

/// Allocates an `Array1<T>` without zero-initialisation and fills it.
///
/// `fill` receives the whole buffer and **must write every element**; the
/// closure-based shape makes a partial write a local bug rather than a
/// scattered-callsite audit.
pub fn array1_from_fill<T: FloatExt>(n: usize, fill: impl FnOnce(&mut [T])) -> Array1<T> {
  let mut arr = Array1::<T>::uninit(n);
  // SAFETY: FloatExt elements are plain floats — no invalid bit patterns, no
  // drop glue — and `fill` writes all `n` elements before `assume_init`.
  let slice = unsafe { std::slice::from_raw_parts_mut(arr.as_mut_ptr() as *mut T, n) };
  fill(slice);
  unsafe { arr.assume_init() }
}

/// Row-major `Array2<T>` variant of [`array1_from_fill`].
pub fn array2_from_fill<T: FloatExt>(
  rows: usize,
  cols: usize,
  fill: impl FnOnce(&mut [T]),
) -> Array2<T> {
  let mut arr = Array2::<T>::uninit((rows, cols));
  // SAFETY: as in `array1_from_fill`; a freshly created Array2 is standard
  // row-major contiguous.
  let slice = unsafe { std::slice::from_raw_parts_mut(arr.as_mut_ptr() as *mut T, rows * cols) };
  fill(slice);
  unsafe { arr.assume_init() }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn array1_from_fill_writes_every_element() {
    let a = array1_from_fill(7usize, |out: &mut [f64]| {
      for (i, x) in out.iter_mut().enumerate() {
        *x = i as f64;
      }
    });
    assert_eq!(a.len(), 7);
    for i in 0..7 {
      assert_eq!(a[i], i as f64);
    }
  }

  #[test]
  fn array1_from_fill_zero_len() {
    let a = array1_from_fill(0usize, |_: &mut [f64]| {});
    assert!(a.is_empty());
  }

  #[test]
  fn array2_from_fill_writes_row_major() {
    let a = array2_from_fill(2usize, 3usize, |out: &mut [f64]| {
      for (i, x) in out.iter_mut().enumerate() {
        *x = i as f64;
      }
    });
    assert_eq!(a[[0, 0]], 0.0);
    assert_eq!(a[[1, 2]], 5.0);
  }
}
