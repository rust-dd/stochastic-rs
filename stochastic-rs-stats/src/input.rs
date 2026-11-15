//! Input checks shared by the estimators and tests: length, finiteness, contiguous access.

use std::borrow::Cow;

use ndarray::ArrayView1;
use ndarray::ArrayView2;

use crate::StatsError;
use crate::traits::FloatExt;

/// The series as a slice: borrowed when contiguous, copied otherwise.
pub(crate) fn as_slice_or_copy<T: Clone>(x: ArrayView1<'_, T>) -> Cow<'_, [T]> {
  match x.to_slice() {
    Some(slice) => Cow::Borrowed(slice),
    None => Cow::Owned(x.to_vec()),
  }
}

/// At least `min_n` finite observations, else the `Err` the caller returns.
pub(crate) fn validate<T: FloatExt>(y: &[T], min_n: usize) -> Result<(), StatsError> {
  if y.len() < min_n {
    return Err(StatsError::InsufficientData {
      needed: min_n,
      got: y.len(),
    });
  }
  if let Some((index, value)) = y.iter().enumerate().find(|(_, v)| !v.is_finite()) {
    return Err(StatsError::InvalidObservation {
      index,
      value: value.to_f64().unwrap(),
      constraint: "finite",
    });
  }
  Ok(())
}

/// Every entry of `x` finite, else `InvalidObservation` at the row (the observation `t`) of the
/// first that is not.
pub(crate) fn validate_rows<T: FloatExt>(x: ArrayView2<'_, T>) -> Result<(), StatsError> {
  for (index, row) in x.rows().into_iter().enumerate() {
    if let Some(value) = row.iter().find(|v| !v.is_finite()) {
      return Err(StatsError::InvalidObservation {
        index,
        value: value.to_f64().unwrap(),
        constraint: "finite",
      });
    }
  }
  Ok(())
}

/// `Err(DegenerateData(what))` when every value equals the first; exact, as a rounded mean lets a
/// constant series through a variance check.
pub(crate) fn reject_constant<T: PartialEq>(y: &[T], what: &'static str) -> Result<(), StatsError> {
  if y.windows(2).all(|w| w[0] == w[1]) {
    return Err(StatsError::DegenerateData(what));
  }
  Ok(())
}

/// The significance-level precondition every test shares.
#[track_caller]
pub(crate) fn assert_alpha(alpha: f64) {
  assert!(
    alpha > 0.0 && alpha < 1.0,
    "alpha must satisfy `0.0 < alpha < 1.0`, got alpha = {alpha:?}"
  );
}
