//! The crate's sort order, NaN last whatever its sign: `f64::total_cmp` places a NaN by its sign,
//! and x86's default NaN is negative where aarch64's is positive.

use std::cmp::Ordering;

use crate::traits::RealExt;

/// Ascending with NaN last whatever its sign; branch-free: an `if` chain sorts 100k samples about
/// 4× slower.
pub(crate) fn ascending<T: RealExt>(a: &T, b: &T) -> Ordering {
  let (a_nan, b_nan) = (a.is_nan(), b.is_nan());
  let less = (a < b) | (!a_nan & b_nan);
  let greater = (b < a) | (a_nan & !b_nan);
  greater.cmp(&less)
}

/// Descending with NaN last whatever its sign.
pub(crate) fn descending<T: RealExt>(a: &T, b: &T) -> Ordering {
  ascending(&-*a, &-*b)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn ascending_puts_every_nan_last_whatever_its_sign() {
    let mut xs = [f64::NAN, 2.0, -f64::NAN, -1.0, 0.0];
    xs.sort_by(ascending);
    assert_eq!(xs[..3], [-1.0, 0.0, 2.0]);
    assert!(xs[3..].iter().all(|x| x.is_nan()));
  }

  #[test]
  fn descending_puts_every_nan_last_whatever_its_sign() {
    let mut xs = [f64::NAN, 2.0, -f64::NAN, -1.0, 0.0];
    xs.sort_by(descending);
    assert_eq!(xs[..3], [2.0, 0.0, -1.0]);
    assert!(xs[3..].iter().all(|x| x.is_nan()));
  }

  #[test]
  fn a_signed_zero_ties_with_the_other_in_both_orders() {
    assert_eq!(ascending(&-0.0, &0.0), Ordering::Equal);
    assert_eq!(descending(&0.0, &-0.0), Ordering::Equal);
  }
}
