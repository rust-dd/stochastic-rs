//! A series at unit scale and centred, and the refusal of a constant's rounding or an exact fit.

use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;

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
pub(crate) fn centred<T: RealExt>(y: &[T]) -> Vec<T> {
  let mut deviations = y.to_vec();
  for _ in 0..2 {
    let mean = deviations.iter().copied().sum::<T>() / T::from_usize_(deviations.len());
    deviations.iter_mut().for_each(|v| *v -= mean);
  }
  deviations
}

/// `2^k` for a `k` whose power is a normal double.
fn two_to(k: i32) -> f64 {
  f64::from_bits(((1023 + k) as u64) << 52)
}

/// The `k` whose `2^k` is nearest `size` in ratio, from its binary exponent and mantissa: a rounded
/// `log2` reads the exponent into the answer beside `√2`. 0 unless `size` is positive and finite.
fn nearest_power_of_two(size: f64) -> i32 {
  if !(size > 0.0 && size.is_finite()) {
    return 0;
  }
  let (normal, shift) = if size < f64::MIN_POSITIVE {
    (size * two_to(54), -54)
  } else {
    (size, 0)
  };
  let bits = normal.to_bits();
  let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023 + shift;
  let mantissa = f64::from_bits((bits & ((1 << 52) - 1)) | (1023 << 52));
  exponent + i32::from(mantissa >= std::f64::consts::SQRT_2)
}

/// The power of two nearest the inverse of the largest magnitude in `y`, held inside `T`'s range.
fn unit_scale<T: RealExt>(y: &[T]) -> T {
  let size = T::max_or_nan_fold(y.iter().map(|v| v.abs()));
  let range = nearest_power_of_two(T::max_value().to_f64().unwrap()) - 2;
  let k = nearest_power_of_two(size.to_f64().unwrap()).clamp(-range, range);
  T::from_f64_fast(two_to(-k))
}

/// `y` times its [`unit_scale`], and that scale: exact, it leaves a scale-free statistic as it is
/// and puts it on the scale of the trend terms beside it.
pub(crate) fn unit_scaled<T: RealExt>(mut y: Vec<T>) -> (Vec<T>, T) {
  let unit = unit_scale(&y);
  for v in &mut y {
    *v = *v * unit;
  }
  (y, unit)
}

/// Each column of `x` times its own [`unit_scale`], and those scales.
pub(crate) fn columns_unit_scaled(x: ArrayView2<f64>) -> (Array2<f64>, Vec<f64>) {
  let mut scaled = x.to_owned();
  let scale = |mut column: ndarray::ArrayViewMut1<f64>| {
    let unit = unit_scale(&column.to_vec());
    column.mapv_inplace(|v| v * unit);
    unit
  };
  let units = scaled.columns_mut().into_iter().map(scale).collect();
  (scaled, units)
}

/// `v` over `unit^power`, a value read off data times `unit` in the data's own units; `Numerical`
/// where it leaves the normal doubles, zero from zero excepted.
pub(crate) fn in_data_units(v: f64, unit: f64, power: f64) -> Result<f64, StatsError> {
  let mapped = v / unit.powf(power);
  if mapped.is_normal() || v == 0.0 {
    Ok(mapped)
  } else {
    Err(StatsError::Numerical {
      stage: "data scale",
    })
  }
}

/// `y` at unit scale, [`centred`], at the unit of what is left, with the two units: what a statistic
/// free of level and unit reads, as a level at unit scale leaves a spread of rounding beside it.
pub(crate) fn about_mean<T: RealExt>(y: Vec<T>) -> (Vec<T>, [T; 2]) {
  let (at_unit, unit) = unit_scaled(y);
  let (deviations, spread_unit) = unit_scaled(centred(&at_unit));
  (deviations, [unit, spread_unit])
}

/// `params` fitted to the data times `unit`, in the data's units (`params[i]` as `unit^powers[i]`);
/// `Numerical` where one is not finite, or leaves the double range or rounds to zero there.
pub(crate) fn params_in_data_units<const K: usize>(
  params: [f64; K],
  unit: f64,
  powers: [usize; K],
) -> Result<[f64; K], StatsError> {
  if !params.iter().all(|p| p.is_finite()) {
    return Err(StatsError::Numerical { stage: "estimate" });
  }
  let mapped: [f64; K] = std::array::from_fn(|i| per_unit(params[i], unit, powers[i]));
  if (0..K).any(|i| !mapped[i].is_finite() || (mapped[i] == 0.0 && params[i] != 0.0)) {
    return Err(StatsError::Numerical {
      stage: "data scale",
    });
  }
  Ok(mapped)
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

#[cfg(test)]
mod tests {
  use super::*;

  /// A power of two moves no bit: the unit of `m 2^e` is the unit of `m` over `2^e`, also for the
  /// mantissas beside `√2`, where a rounded `log2` reads the exponent into its answer.
  #[test]
  fn the_unit_of_a_series_follows_a_power_of_two_exactly() {
    let root = std::f64::consts::SQRT_2.to_bits();
    for offset in (-2000_i64..=2000).step_by(125).chain([-1, 0, 1]) {
      let mantissa = f64::from_bits(root.wrapping_add_signed(offset));
      let unit = unit_scale(&[mantissa]);
      for e in [-1000, -700, -52, -3, -1, 1, 3, 52, 700, 1000] {
        let scaled = unit_scale(&[-mantissa * two_to(e)]);
        assert_eq!(
          scaled * two_to(e),
          unit,
          "mantissa √2 {offset:+} ulps at 2^{e}"
        );
      }
    }
  }

  #[test]
  fn the_unit_is_the_nearest_power_of_two_to_the_inverse_of_the_largest_size() {
    for (size, unit) in [(1.0, 1.0), (1.4, 1.0), (1.5, 0.5), (0.7, 2.0), (0.75, 1.0)] {
      assert_eq!(unit_scale(&[0.1 * size, -size]), unit, "{size}");
    }
    assert_eq!(unit_scale(&[5e-324_f64]), two_to(1022));
    assert_eq!(unit_scale(&[f64::MAX]), two_to(-1022));
    assert_eq!(unit_scale(&[3.0e-45_f32, 0.0]), 2f32.powi(126));
    assert_eq!(unit_scale(&[0.0_f64, -0.0]), 1.0);
  }
}
