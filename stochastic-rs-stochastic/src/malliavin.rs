//! The Malliavin derivative of a process along a path it samples.
//!
//! Reference: Nualart, D. (2006), *The Malliavin Calculus and Related Topics*, 2nd ed., §2.2.

use ndarray::Array1;

/// A sampled path and the Malliavin derivative of one of its components along it: `derivative[i]`
/// is `D_{t_i} X_T`, the derivative of the terminal value at the grid point `t_i`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Malliavin<T, O> {
  pub paths: O,
  pub derivative: Array1<T>,
}
