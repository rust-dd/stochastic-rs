//! What every family's fit, conditional inverse and h-function share: the input checks and the NaN rule, the stable
//! logarithmic helpers the h-functions are built from, Knuth's two-sum and the `ln h` quantile solver.
//! Reference: Shewchuk, J.R. (1997), "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates", *Discrete & Computational Geometry* 18(3), 305-363, Theorem 7 (Knuth's two-sum).

use std::cell::Cell;

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::Zip;
use stochastic_rs_distributions::RealExt;

use crate::bivariate::BivariateFamily;
use crate::error::CopulaError;
use crate::optim::zero;

/// Brent's absolute tolerance `t` for the numerical inverse on `[0, 1]`: the least positive `f64`, as his `zero` needs
/// `t > 0`, so a root keeps its relative precision down to the subnormals.
pub(crate) const ABSOLUTE_TOLERANCE: f64 = f64::from_bits(1);

/// The one domain test for a level `y` or a conditioning value `v`.
pub(crate) fn in_unit(x: f64) -> bool {
  (0.0..=1.0).contains(&x)
}

/// `ln(a/b)` for `a, b > 0` to a relative rounding error: `ln_1p` of the exact difference when the ratio is near 1, the
/// logarithms' difference when the ratio leaves the normal range, where it exceeds 708 in magnitude.
pub(crate) fn ln_ratio(a: f64, b: f64) -> f64 {
  let ratio = a / b;
  if (0.5..=2.0).contains(&ratio) {
    ((a - b) / b).ln_1p()
  } else if ratio >= f64::MIN_POSITIVE && ratio.is_finite() {
    ratio.ln()
  } else {
    a.ln() - b.ln()
  }
}

/// `(m, M, ln(m/M), ℓ)`, `m ≤ M` the two of `x, y = −ln u, −ln v`, the ratio's logarithm by `ln_log_ratio` and
/// `ℓ = ln(1 + (m/M)^θ)`: Gumbel's `−ln C = M e^{ℓ/θ}` and Galambos's `T = m e^{−ℓ/θ}`.
pub(crate) fn log_score_parts(theta: f64, u: f64, v: f64) -> (f64, f64, f64, f64) {
  let (x, y) = (-u.ln(), -v.ln());
  let ln_ratio = -ln_log_ratio(u, v, x, y).abs();
  (
    x.min(y),
    x.max(y),
    ln_ratio,
    (theta * ln_ratio).exp().ln_1p(),
  )
}

/// `ln(x/y)` for `x, y = −ln a, −ln b`: near a ratio of 1, `ln_1p` of `x − y = ln(b/a)`, which `ln_ratio` holds to a
/// rounding error, so `θ ln(x/y)` stays exact where `(x/y)^θ` is not negligible.
pub(crate) fn ln_log_ratio(a: f64, b: f64, x: f64, y: f64) -> f64 {
  let ratio = x / y;
  if (0.5..=2.0).contains(&ratio) {
    (ln_ratio(b, a) / y).ln_1p()
  } else {
    ratio.ln()
  }
}

/// `ln(1 − x^θ)` from `ln x < 0`: `ln(1 − e^t)` at `t = θ ln x`, and `ln θ + ln(−ln x)` where `t` is subnormal, which drops
/// only `t/2`, below every rounding unit, and keeps the digits the product loses.
pub(crate) fn ln_one_minus_pow(theta: f64, ln_x: f64) -> f64 {
  let t = theta * ln_x;
  if -t >= f64::MIN_POSITIVE {
    ln_one_minus_exp(t)
  } else {
    theta.ln() + (-ln_x).ln()
  }
}

/// `ln((1 − a^θ)/(1 − b^θ))` from `ln a, ln b < 0`: where both `θ ln` are above −1, the ratio of the logarithms times
/// `expm1(t)/t` of each, so the two `ln θ` cancel exactly instead of in rounding.
pub(crate) fn ln_one_minus_pow_ratio(theta: f64, ln_a: f64, ln_b: f64) -> f64 {
  let (t_a, t_b) = (theta * ln_a, theta * ln_b);
  if t_a > -1.0 && t_b > -1.0 {
    ln_ratio(-ln_a, -ln_b) + (expm1_over(t_a) / expm1_over(t_b)).ln()
  } else {
    ln_one_minus_pow(theta, ln_a) - ln_one_minus_pow(theta, ln_b)
  }
}

/// `expm1(t)/t`, 1 at `t = 0`.
pub(crate) fn expm1_over(t: f64) -> f64 {
  if t == 0.0 { 1.0 } else { t.exp_m1() / t }
}

/// `ln_1p(t)/t`, 1 at `t = 0`.
pub(crate) fn ln_1p_over(t: f64) -> f64 {
  if t == 0.0 { 1.0 } else { t.ln_1p() / t }
}

/// `ln(expm1(z))` for `z ≥ 0`, from `ln z` where `z` is subnormal.
pub(crate) fn ln_expm1(z: f64, ln_z: f64) -> f64 {
  if z < f64::MIN_POSITIVE {
    ln_z
  } else if z < 36.0 {
    z.exp_m1().ln()
  } else {
    z + (-(-z).exp()).ln_1p()
  }
}

/// `ln(eᵃ + eᵇ)`; an infinite larger term is the answer, where `a − b` would be NaN.
pub(crate) fn ln_add_exp(a: f64, b: f64) -> f64 {
  let top = a.max_or_nan(b);
  if top.is_infinite() {
    top
  } else {
    top + softplus(-(a - b).abs())
  }
}

/// `ln((1 − u)/(1 − v))` to a relative rounding error: the complements are exact from `1/2` up, and near a ratio of 1
/// below it the numerator is `v − u` in one rounding, not the difference of two rounded complements.
pub(crate) fn ln_ratio_complement(u: f64, v: f64) -> f64 {
  if u >= 0.5 && v >= 0.5 {
    return ln_ratio(1.0 - u, 1.0 - v);
  }
  let v_bar = 1.0 - v;
  if (0.5..=2.0).contains(&((1.0 - u) / v_bar)) {
    ((v - u) / v_bar).ln_1p()
  } else {
    (-u).ln_1p() - (-v).ln_1p()
  }
}

/// `u + v − 1`, in one rounding for `s = u + v ∈ [1/2, 2]`: `s − 1` is exact there, `e` is the rounding error of `s`
/// (Knuth's two-sum), so a sum near 1 keeps the digits of both terms.
pub(crate) fn sum_minus_one(u: f64, v: f64) -> f64 {
  let s = u + v;
  let z = s - u;
  let e = (u - (s - z)) + (v - z);
  (s - 1.0) + e
}

/// `x − y` of the quantiles of `u` and `v` by `solve(anchor, level, gap, start)` from the exact level gap, anchored at the
/// level farther from ½.
pub(crate) fn level_difference(
  (u, v, x, y): (f64, f64, f64, f64),
  threshold: impl FnOnce(f64, f64) -> f64,
  solve: impl FnOnce(f64, f64, f64, f64) -> f64,
) -> f64 {
  let plain = x - y;
  let (anchor, level, partner, gap, sign) = if x.abs() >= y.abs() {
    (x, u, y, u - v, 1.0)
  } else {
    (y, v, x, v - u, -1.0)
  };
  if gap.abs().ln() < threshold(anchor, partner) {
    sign * solve(anchor, level, gap, sign * plain)
  } else {
    plain
  }
}

/// `x + y` of the quantiles of `u` and `v` by `solve(score, level, gap, start)` from the exact gap `u + v − 1`, anchored
/// at the lower level.
pub(crate) fn level_sum(
  (u, v, x, y): (f64, f64, f64, f64),
  admit: impl FnOnce(f64, f64, f64, f64) -> bool,
  solve: impl FnOnce(f64, f64, f64, f64) -> f64,
) -> f64 {
  let plain = x + y;
  let (p, score, other) = if u <= v { (u, x, y) } else { (v, y, x) };
  let gap = sum_minus_one(u, v);
  if admit(plain, gap, score, other) {
    solve(score, p, gap, plain)
  } else {
    plain
  }
}

/// `ln(1 + eˣ)`.
pub(crate) fn softplus(x: f64) -> f64 {
  x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// `ln(1 − eᶻ)` for `z ≤ 0` without cancellation: `ln_1p(−eᶻ)` when `eᶻ` is small, `ln(−expm1(z))` near 1.
pub(crate) fn ln_one_minus_exp(z: f64) -> f64 {
  if z < -std::f64::consts::LN_2 {
    (-z.exp()).ln_1p()
  } else {
    (-z.exp_m1()).ln()
  }
}

/// `MarginalOutOfRange` unless every value lies in `[0, 1]`, a NaN failing: the input rule of every batch entry point.
pub(crate) fn check_unit<'a>(values: impl IntoIterator<Item = &'a f64>) -> Result<(), CopulaError> {
  // `&` without short-circuit folds to one vector reduction.
  if values.into_iter().fold(true, |ok, &x| ok & in_unit(x)) {
    Ok(())
  } else {
    Err(CopulaError::MarginalOutOfRange)
  }
}

/// `MarginalOnBoundary` at the first 0 or 1 in `x`: a fit or test runs on pseudo-observations, which never reach either end.
pub(crate) fn check_open(x: ArrayView2<f64>) -> Result<(), CopulaError> {
  // `|` without short-circuit folds to one vector reduction; the cell is looked up only once a 0 or 1 is seen.
  if !x
    .iter()
    .fold(false, |seen, &u| seen | (u == 0.0) | (u == 1.0))
  {
    return Ok(());
  }
  match x.indexed_iter().find(|&(_, &u)| u == 0.0 || u == 1.0) {
    None => Ok(()),
    Some(((row, column), &value)) => Err(CopulaError::MarginalOnBoundary { row, column, value }),
  }
}

/// `DimensionMismatch` unless `x` has the two columns `(u, v)`; every batch method runs it before reading a column.
pub(crate) fn check_columns(x: ArrayView2<f64>) -> Result<(), CopulaError> {
  if x.ncols() == 2 {
    return Ok(());
  }
  Err(CopulaError::DimensionMismatch {
    expected: 2,
    got: x.ncols(),
  })
}

/// `x` is n × 2, `n ≥ 2`, with every entry in `[0, 1]` (NaN fails), checked in that order; no uniformity test, so a valid
/// sample never fails by chance.
pub(crate) fn check_pairs(x: ArrayView2<f64>) -> Result<(), CopulaError> {
  check_columns(x)?;
  if x.nrows() < 2 {
    return Err(CopulaError::InsufficientData {
      needed: 2,
      got: x.nrows(),
    });
  }
  check_unit(x)
}

/// `DimensionMismatch` unless every quantile level in `y` has its conditioning value in `v`.
pub(crate) fn check_lengths(y: ArrayView1<f64>, v: ArrayView1<f64>) -> Result<(), CopulaError> {
  if y.len() == v.len() {
    return Ok(());
  }
  Err(CopulaError::DimensionMismatch {
    expected: y.len(),
    got: v.len(),
  })
}

/// `u` with an exact 0 or 1 moved one ulp inside `(0, 1)`: the one adjustment a sampler or a vine's h makes to a value
/// it computed, so a draw or a conditioning value never lands on an edge by rounding.
pub(crate) fn inside(u: f64) -> f64 {
  if u == 0.0 {
    0.0_f64.next_up()
  } else if u == 1.0 {
    1.0_f64.next_down()
  } else {
    u
  }
}

/// The quantile rule: `u` held in `[0, 1]` against rounding, NaN where it is not finite.
pub(crate) fn confine(u: f64) -> f64 {
  // `u·0` is ±0 for a finite `u` and NaN otherwise, so the rule is one select, no branch per pair.
  u.clamp(0.0, 1.0) + u * 0.0
}

/// A closed-form inverse mapped over checked `(y, v)`, each in `[0, 1]`, under [`confine`].
pub(crate) fn conditional_quantiles(
  y: ArrayView1<f64>,
  v: ArrayView1<f64>,
  quantile: impl Fn(f64, f64) -> f64,
) -> Array1<f64> {
  Zip::from(y)
    .and(v)
    .map_collect(|&y, &v| confine(quantile(y, v)))
}

/// The quantiles of a family given by `ln h(u | v)`: Brent's `zero` in `u ∈ [0, 1]` of `h − y` up to `y = 1/2` and of
/// `(1 − y) − (1 − h)` above it, where `h` rounds away the tail, and in `ln u` of `ln h − ln y` for a subnormal `y`.
pub(crate) fn numerical_quantiles(
  y: ArrayView1<f64>,
  v: ArrayView1<f64>,
  family: BivariateFamily,
  ln_h: impl Fn(f64, f64) -> f64,
) -> Result<Array1<f64>, CopulaError> {
  let mut out = Array1::zeros(y.len());
  for (u, (&y, &v)) in out.iter_mut().zip(y.iter().zip(v)) {
    *u = numerical_quantile(y, v, &ln_h).ok_or_else(|| {
      CopulaError::Numerical(format!(
        "{family:?} h-inverse did not converge at (y={y:?}, v={v:?})"
      ))
    })?;
  }
  Ok(out)
}

/// One quantile of `numerical_quantiles`; `None` for a NaN `ln h` or a solve past Brent's bound.
pub(crate) fn numerical_quantile(y: f64, v: f64, ln_h: &impl Fn(f64, f64) -> f64) -> Option<f64> {
  let nan = Cell::new(false);
  let ln_h = |u: f64| {
    let l = ln_h(u, v);
    nan.set(nan.get() | l.is_nan());
    l
  };
  // The atom at `u = 0`, which a point-mass conditional law puts there, covers every level up to its mass.
  let root = if y == 0.0 || ln_h(0.0) >= y.ln() {
    Some(0.0)
  } else if y > 0.5 {
    zero(0.0, 1.0, ABSOLUTE_TOLERANCE, |u| {
      (1.0 - y) + ln_h(u).exp_m1()
    })
  } else if y >= f64::MIN_POSITIVE {
    zero(0.0, 1.0, ABSOLUTE_TOLERANCE, |u| ln_h(u).exp() - y)
  } else {
    let (ln_y, least) = (y.ln(), f64::from_bits(1));
    if ln_h(least) >= ln_y {
      Some(0.0)
    } else {
      zero(least.ln(), 0.0, ABSOLUTE_TOLERANCE, |t| {
        ln_h(t.exp()) - ln_y
      })
      .map(f64::exp)
    }
  };
  root.filter(|_| !nan.get())
}

/// `∂_v C` mapped over the rows `(u, v)` of `x`, two columns with each coordinate in `[0, 1]`.
pub(crate) fn conditional_cdf(
  x: ArrayView2<f64>,
  h: impl Fn(f64, f64) -> f64,
) -> Result<Array1<f64>, CopulaError> {
  check_columns(x)?;
  check_unit(x)?;
  Ok(
    Zip::from(x.column(0))
      .and(x.column(1))
      .map_collect(|&u, &v| h(u, v)),
  )
}

/// `C(u, v)` over the rows of `x`, each coordinate in `[0, 1]`: `interior` on the open square, 0 on a zero coordinate
/// and the other coordinate on a unit one, exactly.
pub(crate) fn copula_cdf(
  x: ArrayView2<f64>,
  interior: impl Fn(f64, f64) -> f64,
) -> Result<Array1<f64>, CopulaError> {
  check_columns(x)?;
  check_unit(x)?;
  Ok(
    Zip::from(x.column(0))
      .and(x.column(1))
      .map_collect(|&u, &v| {
        if u == 0.0 || v == 0.0 {
          0.0
        } else if u == 1.0 {
          v
        } else if v == 1.0 {
          u
        } else {
          interior(u, v)
        }
      }),
  )
}

/// `density` over the rows of `x`, each coordinate in `[0, 1]`, on the open square, and NaN on its boundary, where the
/// copula has no density.
pub(crate) fn on_open_square(
  x: ArrayView2<f64>,
  density: impl Fn(f64, f64) -> f64,
) -> Result<Array1<f64>, CopulaError> {
  check_columns(x)?;
  check_unit(x)?;
  Ok(
    Zip::from(x.column(0))
      .and(x.column(1))
      .map_collect(|&u, &v| {
        if 0.0 < u && u < 1.0 && 0.0 < v && v < 1.0 {
          density(u, v)
        } else {
          f64::NAN
        }
      }),
  )
}

/// `exp` of `ln_density` under [`on_open_square`].
pub(crate) fn copula_pdf(
  x: ArrayView2<f64>,
  ln_density: impl Fn(f64, f64) -> f64,
) -> Result<Array1<f64>, CopulaError> {
  on_open_square(x, |u, v| ln_density(u, v).exp())
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
    let frank: [f64; 14] = [
      -1e300, -1000.0, -800.0, -30.0, -1e-10, 0.0, 1e-10, 4.0, 30.0, 709.0, 750.0, 800.0, 1000.0,
      1e300,
    ];
    for theta in frank {
      // The reflection and the identity `Frank::percent_point` applies around its `θ > 0` inverse.
      let inverse = Frank::inverse(theta.abs());
      let reflected = |y, v: f64| {
        if theta == 0.0 {
          y
        } else if theta > 0.0 {
          inverse(y, v, 1.0 - v)
        } else {
          inverse(y, 1.0 - v, v)
        }
      };
      assert_rounding_only(&format!("frank θ = {theta:e}"), reflected);
    }
    for theta in [1e-10, 0.5, 2.0, 10.0, 50.0, 100.0, 1000.0, 1e10] {
      let inverse =
        |y, v: f64| Clayton::finish(theta, Clayton::level(theta, y), y, v, v.powf(theta));
      assert_rounding_only(&format!("clayton θ = {theta:e}"), inverse);
    }
    for theta in [-1.0, -0.5, 0.0, 0.5, 0.99, 1.0 - 1e-9] {
      assert_rounding_only(&format!("amh θ = {theta:e}"), Amh::inverse(theta));
    }
    assert_rounding_only("amh θ = 1", Amh::inverse_at_one());
    for theta in [1e-12, 0.01, 0.2, 1.0, 3.0, 50.0, 1e10, 1e154, 1e200, 1e300] {
      assert_rounding_only(&format!("plackett θ = {theta:e}"), Plackett::inverse(theta));
    }
    for theta in [-1.0, -0.5, 0.0, 0.5, 1.0] {
      assert_rounding_only(&format!("fgm θ = {theta:e}"), Fgm::inverse(theta));
    }
    for rho in [-1.0 + 1e-9, -0.5, 0.5, 0.999999, 1.0 - f64::EPSILON] {
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
