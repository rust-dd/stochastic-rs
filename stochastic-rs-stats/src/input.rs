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
  validate_domain(y, min_n, "finite", |_| true)
}

/// [`validate`] on a view, out of line so the pass does not take registers from the caller's loops.
#[inline(never)]
pub(crate) fn validate_view<T: FloatExt>(
  x: ArrayView1<'_, T>,
  min_n: usize,
) -> Result<(), StatsError> {
  validate(&as_slice_or_copy(x), min_n)
}

/// [`validate_domain`] on a view, out of line as [`validate_view`].
#[inline(never)]
pub(crate) fn validate_view_domain<T: FloatExt>(
  x: ArrayView1<'_, T>,
  min_n: usize,
  constraint: &'static str,
  domain: impl Fn(T) -> bool,
) -> Result<(), StatsError> {
  validate_domain(&as_slice_or_copy(x), min_n, constraint, domain)
}

/// As [`validate`], each observation also inside `domain`; `constraint` names both conditions.
pub(crate) fn validate_domain<T: FloatExt>(
  y: &[T],
  min_n: usize,
  constraint: &'static str,
  domain: impl Fn(T) -> bool,
) -> Result<(), StatsError> {
  if y.len() < min_n {
    return Err(StatsError::InsufficientData {
      needed: min_n,
      got: y.len(),
    });
  }
  let inside = |v: T| v.is_finite() & domain(v);
  // A branch-free pass first: the index search runs only on data that fails it.
  if y.iter().fold(true, |ok, &v| ok & inside(v)) {
    return Ok(());
  }
  match y.iter().enumerate().find(|(_, v)| !inside(**v)) {
    Some((index, value)) => Err(StatsError::InvalidObservation {
      index,
      value: value.to_f64().unwrap(),
      constraint,
    }),
    None => Ok(()),
  }
}

/// The mean summed in index order: ndarray's `mean()` regroups contiguous memory, so a strided or
/// reversed view would round apart from its copy.
pub(crate) fn mean_in_order(x: ArrayView1<'_, f64>) -> f64 {
  x.iter().fold(0.0, |acc, &v| acc + v) / x.len() as f64
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

/// A series whose range is within this many `ε` of its largest magnitude is a constant's rounding:
/// neighbouring doubles, as a constant through two or three rounded operations leaves.
const ROUNDING_RANGE: f64 = 4.0;

/// `Err(DegenerateData(what))` below 2 points or when `max − min <= ROUNDING_RANGE ε max|y|`: the
/// variance of such a series is rounding, which no relative rule can tell from a signal.
pub(crate) fn reject_constant<T: FloatExt>(y: &[T], what: &'static str) -> Result<(), StatsError> {
  let low = T::min_or_nan_fold(y.iter().copied());
  let high = T::max_or_nan_fold(y.iter().copied());
  let bound = T::from_f64_fast(ROUNDING_RANGE) * T::epsilon() * low.abs().max(high.abs());
  // A non-finite end is no constant's rounding: `validate` answers for it.
  if y.len() < 2 || (low.is_finite() && high.is_finite() && high - low <= bound) {
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
