//! The crate's precondition form for an array argument.

use std::fmt::Debug;

use ndarray::ArrayBase;
use ndarray::Data;
use ndarray::Dimension;

/// Panics at the first entry of `values` that fails `holds`, naming that entry and its value
/// (``alpha[3] must satisfy `alpha[i].is_finite()`, got alpha[3] = NaN``), never the whole array.
#[track_caller]
pub(crate) fn assert_entries<A, S, D>(
  name: &str,
  predicate: &str,
  values: &ArrayBase<S, D>,
  holds: impl Fn(A) -> bool,
) where
  A: Copy + Debug,
  S: Data<Elem = A>,
  D: Dimension,
  D::Pattern: Debug,
{
  if let Some((i, &v)) = values.indexed_iter().find(|&(_, &v)| !holds(v)) {
    panic!("{name}[{i:?}] must satisfy `{predicate}`, got {name}[{i:?}] = {v:?}");
  }
}
