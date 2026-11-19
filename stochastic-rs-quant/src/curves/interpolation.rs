//! Interpolation methods for yield curve construction.
//!
//! Reference: Hagan & West (2006), Applied Mathematical Finance 13(2), 89–129
//! Reference: Hagan & West (2006), "Methods for Constructing a Yield Curve", Wilmott Magazine

use ndarray::Array1;

use super::types::CurvePoint;
use crate::traits::RealExt;

/// The pillars after `t = 0`; one at `t = 0` is ignored, since the curve is 1 there.
fn after_origin<T: RealExt>(points: &[CurvePoint<T>]) -> &[CurvePoint<T>] {
  &points[points.partition_point(|p| p.time <= T::zero())..]
}

/// The value where no method interpolates: NaN for `t < 0`, 1 at `t = 0` or with no pillar, the
/// first pillar's own, and the last zero rate held flat from the last pillar on.
fn off_grid<T: RealExt>(points: &[CurvePoint<T>], t: T) -> Option<T> {
  if t.is_nan() || t < T::zero() {
    return Some(T::nan());
  }
  let (Some(first), Some(last)) = (points.first(), points.last()) else {
    return Some(T::one());
  };
  if t == T::zero() {
    Some(T::one())
  } else if t == first.time {
    Some(first.discount_factor)
  } else if t >= last.time {
    let r = -last.discount_factor.ln() / last.time;
    Some((-r * t).exp())
  } else {
    None
  }
}

/// `D₁^(t/t₁)`, the first pillar's zero rate held flat, for `t` before that pillar.
fn flat_short_end<T: RealExt>(points: &[CurvePoint<T>], t: T) -> Option<T> {
  let first = points.first()?;
  (t < first.time).then(|| (first.discount_factor.ln() * (t / first.time)).exp())
}

/// `D(t)` a share `w` of the way between two pillars, from their `ln D` (log-linear) or zero
/// rates; the bootstrap's trial segment reads through it too.
#[inline]
pub(super) fn between_pillars<T: RealExt>(log_linear: bool, left: T, right: T, w: T, t: T) -> T {
  if log_linear {
    (left * (T::one() - w) + right * w).exp()
  } else {
    (-(left + w * (right - left)) * t).exp()
  }
}

/// $D(t) = e^{-r(t)\,t}$ with $r$ linear between the pillars' zero rates $r_i = -\ln D_i / t_i$
/// and flat below the first; NaN for `t < 0`.
pub fn linear_on_zero_rates<T: RealExt>(points: &[CurvePoint<T>], t: T) -> T {
  let points = after_origin(points);
  if let Some(known) = off_grid(points, t).or_else(|| flat_short_end(points, t)) {
    return known;
  }
  let idx = points.partition_point(|p| p.time < t).saturating_sub(1);
  let (p0, p1) = (&points[idx], &points[idx + 1]);

  let r0 = -p0.discount_factor.ln() / p0.time;
  let r1 = -p1.discount_factor.ln() / p1.time;

  let w = (t - p0.time) / (p1.time - p0.time);
  between_pillars(false, r0, r1, w, t)
}

/// $\ln D(t)$ linear between pillars and from $(0, 1)$, so the forward rate is piecewise
/// constant; NaN for `t < 0`.
pub fn log_linear_on_discount_factors<T: RealExt>(points: &[CurvePoint<T>], t: T) -> T {
  let points = after_origin(points);
  if let Some(known) = off_grid(points, t).or_else(|| flat_short_end(points, t)) {
    return known;
  }
  let idx = points.partition_point(|p| p.time < t).saturating_sub(1);
  let (p0, p1) = (&points[idx], &points[idx + 1]);

  let w = (t - p0.time) / (p1.time - p0.time);
  between_pillars(true, p0.discount_factor.ln(), p1.discount_factor.ln(), w, t)
}

/// The natural cubic spline through the pillars' zero rates: a flat zero rate below the first
/// pillar, and linear in the rate when the curve has only two pillars; NaN for `t < 0`.
pub fn cubic_spline_on_zero_rates<T: RealExt>(points: &[CurvePoint<T>], t: T) -> T {
  let points = after_origin(points);
  if let Some(known) = off_grid(points, t).or_else(|| flat_short_end(points, t)) {
    return known;
  }
  if points.len() < 3 {
    return linear_on_zero_rates(points, t);
  }

  let times: Vec<T> = points.iter().map(|p| p.time).collect();
  let rates: Vec<T> = points
    .iter()
    .map(|p| -p.discount_factor.ln() / p.time)
    .collect();

  let m2 = natural_spline_second_derivatives(&times, &rates);

  let idx = points.partition_point(|p| p.time < t).saturating_sub(1);
  let h = times[idx + 1] - times[idx];
  let a = (times[idx + 1] - t) / h;
  let b = (t - times[idx]) / h;
  let six = T::from_f64_fast(6.0);

  let r = a * rates[idx]
    + b * rates[idx + 1]
    + (a * a * a - a) * h * h / six * m2[idx]
    + (b * b * b - b) * h * h / six * m2[idx + 1];

  (-r * t).exp()
}

/// $S''(x_i)$ of the natural cubic spline through $(x_i, y_i)$, zero at both ends: the solution of
///
/// $$
/// h_{i-1} M_{i-1} + 2(h_{i-1} + h_i) M_i + h_i M_{i+1}
///   = 6 \left(\frac{y_{i+1} - y_i}{h_i} - \frac{y_i - y_{i-1}}{h_{i-1}}\right)
/// $$
fn natural_spline_second_derivatives<T: RealExt>(x: &[T], y: &[T]) -> Vec<T> {
  let n = x.len();
  let mut h = vec![T::zero(); n - 1];
  let mut alpha = vec![T::zero(); n];
  let mut l = vec![T::one(); n];
  let mut mu = vec![T::zero(); n];
  let mut z = vec![T::zero(); n];

  for i in 0..n - 1 {
    h[i] = x[i + 1] - x[i];
  }

  let six = T::from_f64_fast(6.0);
  for i in 1..n - 1 {
    alpha[i] = six / h[i] * (y[i + 1] - y[i]) - six / h[i - 1] * (y[i] - y[i - 1]);
  }

  for i in 1..n - 1 {
    let two = T::from_f64_fast(2.0);
    l[i] = two * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
    mu[i] = h[i] / l[i];
    z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
  }

  let mut m = vec![T::zero(); n];
  for i in (0..n - 1).rev() {
    m[i] = z[i] - mu[i] * m[i + 1];
  }
  m
}

/// Hagan & West's monotone convex forward curve (2006, §4 and appendix) from the origin
/// $\tau_0 = 0$, where $-\ln D = 0$; one pillar is a flat zero rate; NaN for `t < 0`.
pub fn monotone_convex<T: RealExt>(points: &[CurvePoint<T>], t: T) -> T {
  let points = after_origin(points);
  if let Some(known) = off_grid(points, t) {
    return known;
  }
  if points.len() < 2 {
    return linear_on_zero_rates(points, t);
  }

  let origin = std::iter::once((T::zero(), T::zero()));
  let (times, rt): (Vec<T>, Vec<T>) = origin
    .chain(points.iter().map(|p| (p.time, -p.discount_factor.ln())))
    .unzip();
  let n = times.len();

  let mut fwd = vec![T::zero(); n];
  for i in 1..n {
    fwd[i] = (rt[i] - rt[i - 1]) / (times[i] - times[i - 1]);
  }
  fwd[0] = fwd[1];

  let mut f_node = vec![T::zero(); n];
  for i in 1..n - 1 {
    let w = (times[i] - times[i - 1]) / (times[i + 1] - times[i - 1]);
    f_node[i] = (T::one() - w) * fwd[i] + w * fwd[i + 1];
  }

  let half = T::from_f64_fast(0.5);
  let first = fwd[1] - half * (f_node[1] - fwd[1]);
  let last = fwd[n - 1] - half * (f_node[n - 2] - fwd[n - 1]);
  f_node[0] = positivity_bound(first, fwd[1]);
  f_node[n - 1] = positivity_bound(last, fwd[n - 1]);
  for i in 1..n - 1 {
    f_node[i] = positivity_bound(f_node[i], fwd[i].min_or_nan(fwd[i + 1]));
  }

  // The first knot at or after `t` is the right end of its interval, so `t` at a knot has x = 1.
  let idx = times.partition_point(|time| *time < t).max(1).min(n - 1);
  let h = times[idx] - times[idx - 1];
  let x = (t - times[idx - 1]) / h;

  let fi = fwd[idx];
  let g0 = f_node[idx - 1] - fi;
  let g1 = f_node[idx] - fi;

  let int_g = integrate_g_hagan_west(g0, g1, x);
  let forward_adj = rt[idx - 1] + h * (fi * x + int_g);

  (-forward_adj).exp()
}

/// Hagan & West's (60)–(62), `bound(0, f, 2·discrete)`, NaN for a NaN node; the cap needs a
/// positive discrete forward, which the paper assumes, and is dropped without one.
fn positivity_bound<T: RealExt>(f: T, discrete: T) -> T {
  let floored = f.max_or_nan(T::zero());
  if discrete > T::zero() {
    floored.min_or_nan(T::from_f64_fast(2.0) * discrete)
  } else {
    floored
  }
}

/// $\int_0^x G(s)\,ds$ of the Hagan–West monotone-convex interval-relative forward $G$, by the
/// region the signs of $g_0, g_1$ select; 0 in the flat case $g_0 = g_1 = 0$.
fn integrate_g_hagan_west<T: RealExt>(g0: T, g1: T, x: T) -> T {
  let zero = T::zero();
  let two = T::from_f64_fast(2.0);
  let three = T::from_f64_fast(3.0);

  if g0 == zero && g1 == zero {
    return zero;
  }

  let g1_plus_2g0 = g1 + two * g0;
  let g0_plus_2g1 = g0 + two * g1;

  // Region 1, g₁+2g₀ and g₀+2g₁ of opposite sign: G(x) = 3(g₀+g₁)x² − 2(g₁+2g₀)x + g₀, so
  // ∫₀ˣ G = (g₀+g₁)x³ − (g₁+2g₀)x² + g₀x.
  let region1 =
    (g1_plus_2g0 < zero && g0_plus_2g1 >= zero) || (g1_plus_2g0 > zero && g0_plus_2g1 <= zero);
  if region1 {
    return (g0 + g1) * x * x * x - g1_plus_2g0 * x * x + g0 * x;
  }

  // Region 2: g₀ and g₁+2g₀ have opposite signs → flat-then-quadratic at η.
  let region2 = (g0 < zero && g1_plus_2g0 >= zero) || (g0 > zero && g1_plus_2g0 <= zero);
  if region2 {
    let denom = g1 - g0;
    if denom == zero {
      return g0 * x;
    }
    let eta = g1_plus_2g0 / denom;
    if x < eta {
      // Flat segment.
      return g0 * x;
    }
    // ∫₀ˣ G = g₀x + (g₁−g₀)(x−η)³ / [3(1−η)²]
    let one_minus_eta = T::one() - eta;
    if one_minus_eta == zero {
      return g0 * x;
    }
    let diff = x - eta;
    return g0 * x + (g1 - g0) * diff * diff * diff / (three * one_minus_eta * one_minus_eta);
  }

  // Region 3: g₁ and g₀+2g₁ have opposite signs → quadratic-then-flat at η.
  let region3 = (g1 <= zero && g0_plus_2g1 > zero) || (g1 >= zero && g0_plus_2g1 < zero);
  if region3 {
    let denom = g1 - g0;
    if denom == zero {
      return g0 * x;
    }
    let eta = three * g1 / denom;
    if x <= eta {
      // Quadratic segment on [0, η]: G(s) = g₁ + (g₀−g₁)((η−s)/η)²
      // ∫₀ˣ G = g₁ x + (g₀−g₁)(η³ − (η−x)³) / (3η²)
      if eta == zero {
        return g1 * x;
      }
      let eta_minus_x = eta - x;
      let cube_diff = eta * eta * eta - eta_minus_x * eta_minus_x * eta_minus_x;
      return g1 * x + (g0 - g1) * cube_diff / (three * eta * eta);
    }
    // ∫₀ˣ G = g₁ x + (g₀−g₁) η / 3
    return g1 * x + (g0 - g1) * eta / three;
  }

  // Region 4: g₀ and g₁ same sign → two-piece quadratic, apex at A.
  let denom = g0 + g1;
  if denom == zero {
    return g0 * x;
  }
  let eta = g1 / denom;
  let a_apex = -(eta * g0 + (T::one() - eta) * g1) / two;

  if x < eta {
    // ∫₀ˣ G = A x + (g₀−A)(η³ − (η−x)³) / (3η²)
    if eta == zero {
      return a_apex * x;
    }
    let eta_minus_x = eta - x;
    let cube_diff = eta * eta * eta - eta_minus_x * eta_minus_x * eta_minus_x;
    return a_apex * x + (g0 - a_apex) * cube_diff / (three * eta * eta);
  }
  // ∫₀^η G + ∫_η^x G
  // ∫₀^η G = A η + (g₀−A) η/3
  let int_to_eta = a_apex * eta + (g0 - a_apex) * eta / three;
  let one_minus_eta = T::one() - eta;
  if one_minus_eta == zero {
    return int_to_eta + a_apex * (x - eta);
  }
  let diff = x - eta;
  int_to_eta
    + a_apex * diff
    + (g1 - a_apex) * diff * diff * diff / (three * one_minus_eta * one_minus_eta)
}

/// The discount factor at `t`, NaN for `t < 0`; a pillar at `t = 0` is ignored. Before the first
/// pillar the zero rate is flat, except under `MonotoneConvex`, which shapes `[0, t₁]` too.
pub fn interpolate_discount_factor<T: RealExt>(
  points: &[CurvePoint<T>],
  t: T,
  method: super::types::InterpolationMethod,
) -> T {
  use super::types::InterpolationMethod;
  match method {
    InterpolationMethod::LinearOnZeroRates => linear_on_zero_rates(points, t),
    InterpolationMethod::LogLinearOnDiscountFactors => log_linear_on_discount_factors(points, t),
    InterpolationMethod::CubicSplineOnZeroRates => cubic_spline_on_zero_rates(points, t),
    InterpolationMethod::MonotoneConvex => monotone_convex(points, t),
  }
}

/// The pillars' zero rates `−ln Dᵢ / tᵢ`; NaN at a pillar with `t ≤ 0`, which has none.
pub fn zero_rates_from_points<T: RealExt>(points: &[CurvePoint<T>]) -> Array1<T> {
  points
    .iter()
    .map(|p| {
      if p.time > T::zero() {
        -p.discount_factor.ln() / p.time
      } else {
        T::nan()
      }
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  fn from_forwards(forwards: &[f64]) -> Vec<CurvePoint<f64>> {
    let mut minus_log = 0.0;
    forwards
      .iter()
      .enumerate()
      .map(|(i, f)| {
        minus_log += f;
        CurvePoint {
          time: (i + 1) as f64,
          discount_factor: (-minus_log).exp(),
        }
      })
      .collect()
  }

  /// Hagan & West's (60): unbounded, `f₀ = f₁ᵈ − ½(f₁ − f₁ᵈ)` is 0.025 on these forwards, above
  /// the cap `2f₁ᵈ = 0.02`; the forward at `0⁺` is `f₀`.
  #[test]
  fn monotone_convex_caps_the_first_node_at_twice_its_discrete_forward() {
    let points = from_forwards(&[0.01, -0.05, 0.01]);
    let eps = 1e-6;
    let forward_at_zero = -monotone_convex(&points, eps).ln() / eps;
    assert!((forward_at_zero - 0.02).abs() < 1e-6, "{forward_at_zero}");
  }

  /// Hagan & West's (62) at the far end: unbounded, `f_n` is 0.025 against the cap of 0.02.
  #[test]
  fn monotone_convex_caps_the_last_node_at_twice_its_discrete_forward() {
    let points = from_forwards(&[0.01, -0.05, 0.01]);
    let eps = 1e-6;
    let ratio = monotone_convex(&points, 3.0 - eps) / points[2].discount_factor;
    assert!(
      (ratio.ln() / eps - 0.02).abs() < 1e-6,
      "{}",
      ratio.ln() / eps
    );
  }

  /// A NaN pillar makes NaN every interval whose end nodes read it, not only its own two.
  #[test]
  fn monotone_convex_propagates_a_nan_pillar_into_its_neighbours() {
    let mut points = from_forwards(&[0.02, 0.02, 0.02, 0.02, 0.02, 0.02]);
    points[2].discount_factor = f64::NAN;
    for t in [1.5, 2.5, 3.5, 4.5] {
      assert!(monotone_convex(&points, t).is_nan(), "t = {t}");
    }
    assert!((monotone_convex(&points, 0.5) - (-0.01_f64).exp()).abs() < 1e-15);
    assert!(monotone_convex(&points, 5.5).is_finite());
  }

  #[test]
  fn zero_rates_from_points_is_nan_at_the_origin() {
    let mut points = from_forwards(&[0.03, 0.03]);
    points.insert(
      0,
      CurvePoint {
        time: 0.0,
        discount_factor: 1.0,
      },
    );
    let rates = zero_rates_from_points(&points);
    assert!(rates[0].is_nan());
    assert!((rates[1] - 0.03).abs() < 1e-15 && (rates[2] - 0.03).abs() < 1e-15);
  }

  fn pts() -> Vec<CurvePoint<f64>> {
    vec![
      CurvePoint {
        time: 0.5,
        discount_factor: (-0.04_f64 * 0.5).exp(),
      },
      CurvePoint {
        time: 1.0,
        discount_factor: (-0.05_f64 * 1.0).exp(),
      },
      CurvePoint {
        time: 2.0,
        discount_factor: (-0.05_f64 * 2.0).exp(),
      },
    ]
  }

  #[test]
  fn linear_at_pillar_recovers_discount_factor() {
    let df = linear_on_zero_rates(&pts(), 1.0);
    assert!((df - (-0.05_f64).exp()).abs() < 1e-12);
  }

  #[test]
  fn log_linear_at_pillar_recovers_discount_factor() {
    let df = log_linear_on_discount_factors(&pts(), 1.0);
    assert!((df - (-0.05_f64).exp()).abs() < 1e-12);
  }

  #[test]
  fn cubic_spline_at_pillar() {
    let df = cubic_spline_on_zero_rates(&pts(), 1.0);
    assert!((df - (-0.05_f64).exp()).abs() < 1e-9);
  }

  #[test]
  fn monotone_convex_returns_finite() {
    let r = monotone_convex(&pts(), 0.75);
    assert!(r.is_finite());
  }

  fn make_curve(times: &[f64], rates: &[f64]) -> Vec<CurvePoint<f64>> {
    times
      .iter()
      .zip(rates.iter())
      .map(|(&t, &r)| CurvePoint {
        time: t,
        discount_factor: (-r * t).exp(),
      })
      .collect()
  }

  #[test]
  fn monotone_convex_recovers_pillars_exactly() {
    let times = [0.5, 1.0, 2.0, 5.0, 10.0];
    let rates = [0.020, 0.030, 0.040, 0.035, 0.030];
    let points = make_curve(&times, &rates);
    for (&t, &r) in times.iter().zip(rates.iter()) {
      let df = monotone_convex(&points, t);
      let expected = (-r * t).exp();
      assert!(
        (df - expected).abs() < 1e-12,
        "pillar t={t}: got DF={df}, expected {expected}"
      );
    }
  }

  #[test]
  fn monotone_convex_preserves_positivity_on_oscillating_input() {
    // Sharply oscillating forwards — the Tier-1 sketch would overshoot, the
    // full HW algorithm flattens via Region 2/3 corner cases.
    let times = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0];
    let rates = [0.01, 0.08, 0.005, 0.07, 0.01, 0.06];
    let points = make_curve(&times, &rates);
    let n = 200;
    for i in 1..=n {
      let t = 0.25 + (2.0 - 0.25) * (i as f64 / n as f64);
      let df = monotone_convex(&points, t);
      assert!(
        df > 0.0 && df.is_finite(),
        "DF at t={t} non-positive or nan: {df}"
      );
      assert!(df <= 1.0 + 1e-10, "DF at t={t} above 1: {df}");
    }
  }

  #[test]
  fn monotone_convex_area_preservation_recovers_discrete_forwards() {
    // The integrated forward between adjacent pillars should equal the
    // discrete forward times the interval length, modulo fp tolerance.
    let times = [0.5, 1.0, 2.0, 3.0];
    let rates = [0.03, 0.04, 0.05, 0.045];
    let points = make_curve(&times, &rates);
    for i in 1..times.len() {
      let t_prev = times[i - 1];
      let t_curr = times[i];
      let r_prev = rates[i - 1];
      let r_curr = rates[i];
      // Expected discrete forward between pillars.
      let expected_fwd = (r_curr * t_curr - r_prev * t_prev) / (t_curr - t_prev);
      // Recover integrated forward from DFs at both pillars.
      let df_prev = monotone_convex(&points, t_prev);
      let df_curr = monotone_convex(&points, t_curr);
      let recovered = (df_prev / df_curr).ln() / (t_curr - t_prev);
      assert!(
        (recovered - expected_fwd).abs() < 1e-10,
        "interval [{t_prev},{t_curr}]: got fwd={recovered}, expected {expected_fwd}"
      );
    }
  }

  #[test]
  fn monotone_convex_flat_curve_stays_flat() {
    let times = [0.5, 1.0, 2.0, 5.0];
    let rates = [0.03, 0.03, 0.03, 0.03];
    let points = make_curve(&times, &rates);
    for t in [0.6, 0.75, 1.5, 3.0, 4.0] {
      let df = monotone_convex(&points, t);
      assert!(
        (df - (-0.03_f64 * t).exp()).abs() < 1e-10,
        "flat curve at t={t}: got {df}, expected {}",
        (-0.03_f64 * t).exp()
      );
    }
  }
}
