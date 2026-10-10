//! The rules every family's conditional inverse and h-function share: one length check, NaN outside the unit
//! interval, and the stopping rule and budget of the numerical inverse.
//!
//! Reference: Brent, R.P. (1973), "Algorithms for Minimization without Derivatives", Prentice-Hall, ch. 4, eqs. (2.9), (3.3), (3.4).

use ndarray::Array1;
use ndarray::Array2;
use ndarray::Zip;
use roots::Convergency;

use crate::error::CopulaError;

/// Brent's absolute tolerance `t`: positive, as his procedure requires, and below every relative term on the bracket.
const ABSOLUTE_TOLERANCE: f64 = f64::MIN_POSITIVE;

/// Brent's worst case `(k + 1)² − 2` (eq. 3.4) over bisection's `k = ⌈log₂((1 − EPSILON)/δ)⌉ = 104`, `δ = EPSILON² + t`
/// the tolerance at the floor of `[EPSILON, 1]`: staircase-flat h-functions at extreme arguments need more than `k + 1`.
pub(crate) const MAX_ITER: usize = 11_023;

/// Brent's stop: the bracket within `2δ`, `δ = 2ε|u| + t` with `ε = 2⁻⁵³` the rounding unit (`2ε = f64::EPSILON`).
pub(crate) struct BrentTolerance;

impl Convergency<f64> for BrentTolerance {
  fn is_root_found(&mut self, y: f64) -> bool {
    y == 0.0
  }

  fn is_converged(&mut self, x1: f64, x2: f64) -> bool {
    (x1 - x2).abs() <= 2.0 * (f64::EPSILON * x2.abs() + ABSOLUTE_TOLERANCE)
  }

  fn is_iteration_limit_reached(&mut self, iter: usize) -> bool {
    iter >= MAX_ITER
  }
}

/// `DimensionMismatch` unless every quantile level in `y` has its conditioning value in `v`.
pub(crate) fn check_lengths(y: &Array1<f64>, v: &Array1<f64>) -> Result<(), CopulaError> {
  if y.len() == v.len() {
    return Ok(());
  }
  Err(CopulaError::DimensionMismatch {
    expected: y.len(),
    got: v.len(),
  })
}

/// The quantile rule: `u` of `(y, v)` held in `[0, 1]`, which rounding can leave by a few ulps near `y = 1`, NaN where
/// `y` or `v` leaves `[0, 1]`; applied after the inverse, as an early return measured slower in the sampling bench.
pub(crate) fn confine(y: f64, v: f64, u: f64) -> f64 {
  let u = u.clamp(0.0, 1.0);
  if (0.0..=1.0).contains(&y) && (0.0..=1.0).contains(&v) {
    u
  } else {
    f64::NAN
  }
}

/// The h-function rule: `p` where the conditioning `v` lies in `[0, 1]`, NaN elsewhere.
pub(crate) fn conditioned(v: f64, p: f64) -> f64 {
  if (0.0..=1.0).contains(&v) {
    p
  } else {
    f64::NAN
  }
}

/// A closed-form inverse mapped over `(y, v)` under [`confine`].
pub(crate) fn conditional_quantiles(
  y: &Array1<f64>,
  v: &Array1<f64>,
  quantile: impl Fn(f64, f64) -> f64,
) -> Result<Array1<f64>, CopulaError> {
  check_lengths(y, v)?;
  Ok(
    Zip::from(y)
      .and(v)
      .map_collect(|&y, &v| confine(y, v, quantile(y, v))),
  )
}

/// `∂_v C` mapped over the rows `(u, v)` of `x` under [`conditioned`].
pub(crate) fn conditional_cdf(x: &Array2<f64>, h: impl Fn(f64, f64) -> f64) -> Array1<f64> {
  Zip::from(x.column(0))
    .and(x.column(1))
    .map_collect(|&u, &v| conditioned(v, h(u, v)))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn the_budget_is_brents_worst_case_on_the_bracket() {
    let floor = f64::EPSILON;
    let k = ((1.0 - floor) / (floor * floor + ABSOLUTE_TOLERANCE))
      .log2()
      .ceil() as usize;
    assert_eq!(k, 104);
    assert_eq!(MAX_ITER, (k + 1) * (k + 1) - 2);
  }
}
