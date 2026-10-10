//! The rules every family's conditional inverse and h-function share: one length check, NaN outside the unit
//! interval, and the absolute tolerance of the numerical inverse.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::Zip;

use crate::error::CopulaError;

/// Brent's absolute tolerance `t` for the numerical inverse: positive, as his `zero` requires, and below every relative
/// term on the bracket `[f64::EPSILON, 1]`.
pub(crate) const ABSOLUTE_TOLERANCE: f64 = f64::MIN_POSITIVE;

/// The one domain test for a level `y` or a conditioning value `v`.
pub(crate) fn in_unit(x: f64) -> bool {
  (0.0..=1.0).contains(&x)
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
  if in_unit(y) && in_unit(v) {
    u
  } else {
    f64::NAN
  }
}

/// The h-function rule: `p` where the conditioning `v` lies in `[0, 1]`, NaN elsewhere.
pub(crate) fn conditioned(v: f64, p: f64) -> f64 {
  if in_unit(v) { p } else { f64::NAN }
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
