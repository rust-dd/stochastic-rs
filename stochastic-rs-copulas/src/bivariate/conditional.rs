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

/// The quantile rule: `u` of `(y, v)` held in `[0, 1]` against rounding, NaN where `y` or `v` leaves `[0, 1]` or `u` is
/// not finite; applied after the inverse, as an early return measured slower in the sampling bench.
pub(crate) fn confine(y: f64, v: f64, u: f64) -> f64 {
  let clamped = u.clamp(0.0, 1.0);
  if in_unit(y) && in_unit(v) && u.is_finite() {
    clamped
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

#[cfg(test)]
mod tests {
  use crate::bivariate::amh::Amh;
  use crate::bivariate::clayton::Clayton;
  use crate::bivariate::fgm::Fgm;
  use crate::bivariate::frank::Frank;
  use crate::bivariate::gaussian::GaussianCopula;
  use crate::bivariate::marshall_olkin::MarshallOlkin;
  use crate::bivariate::plackett::Plackett;
  use crate::bivariate::t_copula::TCopula;

  /// The brief's grid, its edges, the extreme uniform draws `2⁻⁵³` and `1 − 2⁻⁵³`, and tiny levels of the public API.
  const GRID: [f64; 14] = [
    0.0,
    5e-324,
    1e-300,
    1e-15,
    f64::EPSILON / 2.0,
    1e-6,
    0.01,
    0.3,
    0.5,
    0.7,
    0.99,
    1.0 - 1e-6,
    1.0 - f64::EPSILON / 2.0,
    1.0,
  ];

  fn assert_rounding_only(family: &str, inverse: impl Fn(f64, f64) -> f64) {
    let mut worst = 0.0_f64;
    for y in GRID {
      for v in GRID {
        let u = inverse(y, v);
        assert!(u.is_finite(), "{family}: h⁻¹({y:e} | {v:e}) = {u}");
        worst = worst.max(-u).max(u - 1.0);
      }
    }
    assert!(
      worst <= 4.0 * f64::EPSILON,
      "{family}: {worst:e} outside [0, 1]"
    );
  }

  /// `confine` only absorbs rounding: every closed form's raw inverse is finite and within four rounding units of
  /// `[0, 1]` on the grid, at each family's parameter extremes.
  #[test]
  fn the_raw_closed_forms_leave_the_unit_interval_by_rounding_only() {
    let frank = [
      -1e300, -1000.0, -800.0, -30.0, -1e-10, 0.0, 1e-10, 4.0, 30.0, 709.0, 750.0, 800.0, 1000.0,
      1e300,
    ];
    for theta in frank {
      assert_rounding_only(&format!("frank θ = {theta:e}"), Frank::inverse(theta));
    }
    for theta in [1e-10, 0.5, 2.0, 10.0, 50.0, 100.0, 1000.0, 1e10] {
      let inverse =
        |y, v: f64| Clayton::finish(theta, Clayton::level(theta, y), y, v, v.powf(theta));
      assert_rounding_only(&format!("clayton θ = {theta:e}"), inverse);
    }
    for theta in [-1.0, -0.5, 0.0, 0.5, 0.99, 1.0 - 1e-9, 1.0] {
      assert_rounding_only(&format!("amh θ = {theta:e}"), Amh::inverse(theta));
    }
    for theta in [1e-12, 0.01, 0.2, 1.0, 3.0, 50.0, 1e10, 1e154, 1e200, 1e300] {
      assert_rounding_only(&format!("plackett θ = {theta:e}"), Plackett::inverse(theta));
    }
    for theta in [-1.0, -0.5, 0.0, 0.5, 1.0] {
      assert_rounding_only(&format!("fgm θ = {theta:e}"), Fgm::inverse(theta));
    }
    for rho in [-1.0 + 1e-9, -0.5, 0.0, 0.5, 0.999999, 1.0 - f64::EPSILON] {
      assert_rounding_only(
        &format!("gaussian ρ = {rho:e}"),
        GaussianCopula::inverse(rho),
      );
    }
    for (rho, nu) in [
      (-0.99, 0.5),
      (0.0, 1.0),
      (0.5, 4.0),
      (0.99, 30.0),
      (-0.5, 1e6),
    ] {
      assert_rounding_only(
        &format!("t ρ = {rho}, ν = {nu:e}"),
        TCopula::inverse(rho, nu),
      );
    }
    for (alpha, beta) in [
      (0.3, 0.6),
      (1.0, 1.0),
      (1.0, 0.5),
      (0.5, 1.0),
      (1e-6, 1e-6),
      (0.999, 0.001),
    ] {
      assert_rounding_only(
        &format!("marshall_olkin α = {alpha}, β = {beta}"),
        MarshallOlkin::inverse(alpha, beta),
      );
    }
  }
}
