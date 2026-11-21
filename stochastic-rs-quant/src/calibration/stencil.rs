use ndarray::Array1;
use stochastic_rs_distributions::RealExt;

/// `1e-3·|x|`, at least `1e-5`, every numeric Jacobian's step: `ε^{1/3}` for the Fourier walks'
/// `ε ≈ 1e-8` (Nocedal & Wright 2006, ch. 8), shared by `HestonPricer` at a `1e-6` truncation.
pub(crate) fn step(x: f64) -> f64 {
  1e-5_f64.max_or_nan(1e-3 * x.abs())
}

/// `∂r/∂x` by differences of step `h` inside `[lo, hi]`: central where the stencil fits, else the
/// second-order one-sided one inside it, first order cut to a box too narrow for either.
pub(crate) fn derivative(
  x: f64,
  h: f64,
  (lo, hi): (f64, f64),
  mut residuals: impl FnMut(f64) -> Array1<f64>,
) -> Array1<f64> {
  if x.is_nan() || (x - h >= lo && x + h <= hi) {
    return (residuals(x + h) - residuals(x - h)) / (2.0 * h);
  }
  if x + 2.0 * h <= hi {
    let (at, near, far) = (residuals(x), residuals(x + h), residuals(x + 2.0 * h));
    return (4.0 * near - 3.0 * at - far) / (2.0 * h);
  }
  if x - 2.0 * h >= lo {
    let (at, near, far) = (residuals(x), residuals(x - h), residuals(x - 2.0 * h));
    return (3.0 * at - 4.0 * near + far) / (2.0 * h);
  }
  let (below, above) = (lo.max_or_nan(x - h), hi.min_or_nan(x + h));
  (residuals(above) - residuals(below)) / (above - below)
}

/// Per coordinate −1 within its step of `lo`, 1 within it of `hi`, 0 where the central stencil
/// fits: the bound a point rests on.
pub(crate) fn sides(point: &Array1<f64>, bounds: &[(f64, f64)]) -> Array1<f64> {
  let places = point.iter().zip(bounds);
  let side = |(&x, &(lo, hi)): (&f64, &(f64, f64))| match step(x) {
    h if x - h < lo => -1.0,
    h if x + h > hi => 1.0,
    _ => 0.0,
  };
  places.map(side).collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  /// `r(x) = (x³, eˣ)` on `[0, 1]`: second order everywhere, the bounds included.
  #[test]
  fn the_stencil_is_second_order_at_and_beside_each_bound() {
    let r = |x: f64| ndarray::array![x.powi(3), x.exp()];
    for x in [0.0_f64, 2e-4, 0.5, 1.0 - 2e-4, 1.0] {
      let want = ndarray::array![3.0 * x * x, x.exp()];
      let got = derivative(x, 1e-3, (0.0, 1.0), r);
      let gap = f64::max_or_nan_fold((&got - &want).iter().map(|d| d.abs()));
      assert!(gap < 4e-6, "{x}: {got} vs {want}");
    }
  }

  /// A NaN step reaches the box-narrow stencil too, whose ends `max`/`min` would clamp to the box.
  #[test]
  fn a_nan_point_or_step_differences_to_nan() {
    let r = |x: f64| ndarray::array![x, 2.0 * x];
    for (x, h) in [(f64::NAN, 1e-3), (0.5, f64::NAN)] {
      let all_nan = derivative(x, h, (0.0, 1.0), r).iter().all(|d| d.is_nan());
      assert!(all_nan, "x = {x}, h = {h}");
    }
    assert!(step(f64::NAN).is_nan());
    assert_eq!((step(0.0), step(-2.0)), (1e-5, 2e-3));
  }
}
