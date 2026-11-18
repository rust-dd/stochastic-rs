//! The one home of the `uninit → write-all → assume_init` pattern; zeroing costs 3.5–6% of a call.
//! Crate-private, because `fill` must write every element and only the crate's call sites do:
//!
//! ```compile_fail,E0603
//! use stochastic_rs_stochastic::buffer::array1_from_fill;
//! ```

use ndarray::Array1;

use crate::traits::FloatExt;

/// Allocates an `Array1<T>` without zeroing; `fill` receives the whole buffer and must write
/// every element.
pub(crate) fn array1_from_fill<T: FloatExt>(n: usize, fill: impl FnOnce(&mut [T])) -> Array1<T> {
  let mut arr = Array1::<T>::uninit(n);
  // SAFETY: FloatExt elements are plain floats — no invalid bit patterns, no
  // drop glue — and `fill` writes all `n` elements before `assume_init`.
  let slice = unsafe { std::slice::from_raw_parts_mut(arr.as_mut_ptr() as *mut T, n) };
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
}
