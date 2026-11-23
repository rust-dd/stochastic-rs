//! A series at unit scale and centred, and the refusal of a constant's rounding or an exact fit.

use ndarray::ArrayView1;

use super::ROUNDING_MARGIN;
use crate::StatsError;
use crate::input::as_slice_or_copy;
use crate::input::reject_constant;
use crate::input::validate;
use crate::traits::RealExt;

/// An exact fit leaves a residual mean square near `ε² ≈ 5e-32` of the series' own variance; one
/// below this fraction is refused, as is a series whose variance an explosive tail dominates.
const EXACT_FIT: f64 = 1e-24;

/// `y` less its mean, twice: the second pass takes off the rounding of the first's mean, as large
/// as the variation itself where the level dwarfs it; scale `y` first, the sum overflows.
pub(crate) fn centred(y: &[f64]) -> Vec<f64> {
  let mut deviations = y.to_vec();
  for _ in 0..2 {
    let mean = deviations.iter().sum::<f64>() / deviations.len() as f64;
    deviations.iter_mut().for_each(|v| *v -= mean);
  }
  deviations
}

/// The power of two nearest the inverse of the largest magnitude in `y`.
fn unit_scale(y: &[f64]) -> f64 {
  let spread = f64::max_or_nan_fold(y.iter().map(|v| v.abs()));
  2f64.powi(-(spread.log2().round() as i32).clamp(-1022, 1022))
}

/// `y` times its [`unit_scale`], and that scale: exact, it leaves a scale-free statistic as it is
/// and puts it on the scale of the trend terms beside it.
pub(crate) fn unit_scaled(mut y: Vec<f64>) -> (Vec<f64>, f64) {
  let unit = unit_scale(&y);
  for v in &mut y {
    *v *= unit;
  }
  (y, unit)
}

/// `v`, read off data times `unit`, divided by `unit` `power` times: one factor at a time, so that
/// none overflows where the quotient does not.
pub(crate) fn per_unit(v: f64, unit: f64, power: usize) -> f64 {
  (0..power).fold(v, |v, _| v / unit)
}

/// [`unit_scaled`] `y` with `min_n` finite points or more and more than a constant's rounding; the
/// length comes first, as a series too short to compare is vacuously constant.
pub(crate) fn prepared_with_unit(
  y: ArrayView1<f64>,
  min_n: usize,
) -> Result<(Vec<f64>, f64), StatsError> {
  let y = as_slice_or_copy(y);
  validate(&y, min_n)?;
  reject_constant(&y, "constant series")?;
  Ok(unit_scaled(y.into_owned()))
}

/// [`prepared_with_unit`]'s series.
pub(crate) fn prepared(y: ArrayView1<f64>, min_n: usize) -> Result<Vec<f64>, StatsError> {
  Ok(prepared_with_unit(y, min_n)?.0)
}

/// `DegenerateData("zero residual variance")` when `residuals` keep at most `EXACT_FIT` of `y`'s
/// variance, or `(ROUNDING_MARGIN κ ε)²` at the solve's κ; `Err` as `reject_constant` on `y`.
pub(crate) fn reject_exact_fit(
  y: &[f64],
  residuals: &[f64],
  condition: f64,
) -> Result<(), StatsError> {
  reject_constant(y, "constant series")?;
  let mean = y.iter().sum::<f64>() / y.len() as f64;
  let spread = f64::max_or_nan_fold(y.iter().map(|v| (v - mean).abs()));
  // Over the largest deviation, so that no square overflows.
  let square = |v: f64| (v / spread) * (v / spread);
  // Both sums run in index order, interleaved so that neither waits on the other.
  let (mut own, mut left) = (-0.0, -0.0);
  for (v, u) in y.iter().zip(residuals) {
    own += square(v - mean);
    left += square(*u);
  }
  for v in y.iter().skip(residuals.len()) {
    own += square(v - mean);
  }
  for u in residuals.iter().skip(y.len()) {
    left += square(*u);
  }
  let (own, left) = (own / y.len() as f64, left / residuals.len() as f64);
  let rounding = (ROUNDING_MARGIN * condition * f64::EPSILON).powi(2);
  if left <= EXACT_FIT.max_or_nan(rounding) * own {
    return Err(StatsError::DegenerateData("zero residual variance"));
  }
  Ok(())
}
