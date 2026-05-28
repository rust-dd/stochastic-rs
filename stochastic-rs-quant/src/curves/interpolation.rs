//! Interpolation methods for yield curve construction.
//!
//! Reference: Hagan & West, "Interpolation Methods for Curve Construction",
//! Applied Mathematical Finance, 13(2), 89-129 (2006).
//! Reference: Hagan & West, "Methods for Constructing a Yield Curve",
//! Wilmott Magazine (2006).

use ndarray::Array1;

use super::types::CurvePoint;
use crate::traits::FloatExt;

/// Linearly interpolate on zero rates.
///
/// Zero rates at knots: $r_i = -\ln D_i / t_i$. Between knots, $r(t)$ is linearly interpolated,
/// then $D(t) = e^{-r(t)\,t}$.
pub fn linear_on_zero_rates<T: FloatExt>(points: &[CurvePoint<T>], t: T) -> T {
  if points.is_empty() {
    return T::one();
  }
  if t <= points[0].time {
    return points[0].discount_factor;
  }
  if t >= points[points.len() - 1].time {
    let last = &points[points.len() - 1];
    let r = -last.discount_factor.ln() / last.time;
    return (-r * t).exp();
  }

  let idx = points.partition_point(|p| p.time < t).saturating_sub(1);
  let (p0, p1) = (&points[idx], &points[idx + 1]);

  let r0 = if p0.time > T::zero() {
    -p0.discount_factor.ln() / p0.time
  } else {
    T::zero()
  };
  let r1 = -p1.discount_factor.ln() / p1.time;

  let w = (t - p0.time) / (p1.time - p0.time);
  let r = r0 + w * (r1 - r0);
  (-r * t).exp()
}

/// Log-linear interpolation on discount factors (piecewise constant forward rates).
///
/// $\ln D(t)$ is linearly interpolated between knots, which implies piecewise constant
/// instantaneous forward rates.
pub fn log_linear_on_discount_factors<T: FloatExt>(points: &[CurvePoint<T>], t: T) -> T {
  if points.is_empty() {
    return T::one();
  }
  if t <= points[0].time {
    return points[0].discount_factor;
  }
  if t >= points[points.len() - 1].time {
    let last = &points[points.len() - 1];
    let r = -last.discount_factor.ln() / last.time;
    return (-r * t).exp();
  }

  let idx = points.partition_point(|p| p.time < t).saturating_sub(1);
  let (p0, p1) = (&points[idx], &points[idx + 1]);

  let w = (t - p0.time) / (p1.time - p0.time);
  let ln_d = p0.discount_factor.ln() * (T::one() - w) + p1.discount_factor.ln() * w;
  ln_d.exp()
}

/// Natural cubic spline on zero rates.
///
/// Fits a natural cubic spline (second derivative = 0 at boundaries) through the zero rate
/// knots, then converts back to discount factors.
pub fn cubic_spline_on_zero_rates<T: FloatExt>(points: &[CurvePoint<T>], t: T) -> T {
  let n = points.len();
  if n == 0 {
    return T::one();
  }
  if t <= points[0].time {
    return points[0].discount_factor;
  }
  if t >= points[n - 1].time {
    let last = &points[n - 1];
    let r = -last.discount_factor.ln() / last.time;
    return (-r * t).exp();
  }
  if n < 3 {
    return linear_on_zero_rates(points, t);
  }

  let times: Vec<T> = points.iter().map(|p| p.time).collect();
  let rates: Vec<T> = points
    .iter()
    .map(|p| {
      if p.time > T::zero() {
        -p.discount_factor.ln() / p.time
      } else {
        T::zero()
      }
    })
    .collect();

  let m2 = natural_cubic_spline_coefficients(&times, &rates);

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

/// Compute second derivatives for natural cubic spline (M = 0 at boundaries).
fn natural_cubic_spline_coefficients<T: FloatExt>(x: &[T], y: &[T]) -> Vec<T> {
  let n = x.len();
  let mut h = vec![T::zero(); n - 1];
  let mut alpha = vec![T::zero(); n];
  let mut l = vec![T::one(); n];
  let mut mu = vec![T::zero(); n];
  let mut z = vec![T::zero(); n];

  for i in 0..n - 1 {
    h[i] = x[i + 1] - x[i];
  }

  let three = T::from_f64_fast(3.0);
  for i in 1..n - 1 {
    alpha[i] = three / h[i] * (y[i + 1] - y[i]) - three / h[i - 1] * (y[i] - y[i - 1]);
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

/// Monotone convex interpolation on forward rates (Hagan & West, 2006,
/// "Interpolation Methods for Curve Construction", Applied Mathematical
/// Finance 13(2), 89-129).
///
/// Constructs a piecewise instantaneous forward rate curve $f(t)$ that:
/// - Is positive whenever all discrete forwards are positive
/// - Preserves monotonicity / convexity of the discrete forwards
/// - Exactly reproduces the discrete forward rates (area preservation)
///
/// The discrete forward rate for interval $[t_{i-1}, t_i]$ is
/// $f_i = (r(t_i) t_i - r(t_{i-1}) t_{i-1}) / (t_i - t_{i-1})$. Node
/// forwards $f_i^-$ are computed as length-weighted averages of adjacent
/// discrete forwards (HW eq.10), then clamped to $[0, 2 \min(f_i, f_{i+1})]$
/// (HW eq.16) to guarantee a positivity-preserving interior. The
/// interval-relative forward $G(x) = f(x) - f_i$ on $[0, 1]$ uses one of
/// **four region forms** (HW Appendix, eqs 25-27 plus the no-correction
/// cubic) chosen by the signs of $g_0 = f_{i-1}^- - f_i$ and
/// $g_1 = f_i^- - f_i$:
///
/// - **Region 1 (no correction):** sign(g₁+2g₀) and sign(g₀+2g₁) differ →
///   $G(x) = 3(g_0+g_1)x^2 - 2(g_1+2g_0)x + g_0$.
/// - **Region 2 (flat-then-quadratic):** $g_0$ and $g_1+2g_0$ have opposite
///   signs → flat $g_0$ on $[0, \eta]$, quadratic transition to $g_1$ on
///   $[\eta, 1]$ with $\eta = (g_1+2g_0)/(g_1-g_0)$.
/// - **Region 3 (quadratic-then-flat):** $g_1$ and $g_0+2g_1$ have opposite
///   signs → quadratic transition from $g_0$ to $g_1$ on $[0, \eta]$, flat
///   $g_1$ on $[\eta, 1]$ with $\eta = 3 g_1/(g_1-g_0)$.
/// - **Region 4 (two-piece quadratic):** $g_0, g_1$ same sign → both pieces
///   are quadratic, meeting at $\eta = g_1/(g_0+g_1)$ with apex
///   $A = -(\eta g_0 + (1-\eta) g_1)/2$.
///
/// Each region preserves $\int_0^1 G(x) dx = 0$ (= area preservation
/// $\int_{t_{i-1}}^{t_i} f(s) ds = h_i f_i$), so the resulting zero-rate
/// curve exactly reproduces the input discount factors at the pillars.
///
/// Reference: Hagan & West (2006), §4 and Appendix. Cross-validated against
/// the Google `tf-quant-finance` implementation
/// (`rates/hagan_west/monotone_convex.py`).
pub fn monotone_convex<T: FloatExt>(points: &[CurvePoint<T>], t: T) -> T {
  let n = points.len();
  if n == 0 {
    return T::one();
  }
  if t <= points[0].time {
    return points[0].discount_factor;
  }
  if t >= points[n - 1].time {
    let last = &points[n - 1];
    let r = -last.discount_factor.ln() / last.time;
    return (-r * t).exp();
  }
  if n < 3 {
    return linear_on_zero_rates(points, t);
  }

  let times: Vec<T> = points.iter().map(|p| p.time).collect();
  let rt: Vec<T> = points
    .iter()
    .map(|p| {
      if p.time > T::zero() {
        -p.discount_factor.ln()
      } else {
        T::zero()
      }
    })
    .collect();

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
  f_node[0] = fwd[1] - half * (f_node[1] - fwd[1]);
  f_node[0] = f_node[0].max(T::zero());
  f_node[n - 1] = fwd[n - 1] - half * (f_node[n - 2] - fwd[n - 1]);
  f_node[n - 1] = f_node[n - 1].max(T::zero());

  let two = T::from_f64_fast(2.0);
  for i in 1..n - 1 {
    let fi = fwd[i];
    let fi1 = fwd[i + 1];
    let min_f = fi.min(fi1);
    if min_f > T::zero() {
      f_node[i] = f_node[i].max(T::zero()).min(two * min_f);
    } else {
      f_node[i] = f_node[i].max(T::zero());
    }
  }

  // `partition_point(|p| p.time < t)` returns the index of the first pillar
  // with `time >= t`, i.e. the right endpoint of the interval containing t.
  // Clamp to `1..n` so callers in the open (t_0, t_n) range always hit a
  // valid bracketing pair `[idx-1, idx]`.
  let idx = points.partition_point(|p| p.time < t).max(1).min(n - 1);
  let h = times[idx] - times[idx - 1];
  let x = (t - times[idx - 1]) / h;

  let fi = fwd[idx];
  let g0 = f_node[idx - 1] - fi;
  let g1 = f_node[idx] - fi;

  let int_g = integrate_g_hagan_west(g0, g1, x);
  let forward_adj = rt[idx - 1] + h * (fi * x + int_g);

  (-forward_adj).exp()
}

/// Compute $\int_0^x G(s) ds$ for the Hagan-West (2006) monotone-convex
/// interval-relative forward $G$, picking among the four region forms by
/// the signs of $g_0, g_1$. Returns 0 on the degenerate flat case ($g_0 = g_1 = 0$).
fn integrate_g_hagan_west<T: FloatExt>(g0: T, g1: T, x: T) -> T {
  let zero = T::zero();
  let two = T::from_f64_fast(2.0);
  let three = T::from_f64_fast(3.0);

  if g0 == zero && g1 == zero {
    return zero;
  }

  let g1_plus_2g0 = g1 + two * g0;
  let g0_plus_2g1 = g0 + two * g1;

  // Region 1: g₁+2g₀ and g₀+2g₁ have opposite signs (the no-correction case).
  // G(x) = 3(g₀+g₁)x² − 2(g₁+2g₀)x + g₀
  // ∫₀ˣ G = (g₀+g₁)x³ − (g₁+2g₀)x² + g₀ x
  let region1 = (g1_plus_2g0 < zero && g0_plus_2g1 >= zero)
    || (g1_plus_2g0 > zero && g0_plus_2g1 <= zero);
  if region1 {
    return (g0 + g1) * x * x * x - g1_plus_2g0 * x * x + g0 * x;
  }

  // Region 2: g₀ and g₁+2g₀ have opposite signs → flat-then-quadratic at η.
  let region2 =
    (g0 < zero && g1_plus_2g0 >= zero) || (g0 > zero && g1_plus_2g0 <= zero);
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
  let region3 =
    (g1 <= zero && g0_plus_2g1 > zero) || (g1 >= zero && g0_plus_2g1 < zero);
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

/// Interpolate a discount factor at time `t` using the given method.
pub fn interpolate_discount_factor<T: FloatExt>(
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

/// Compute zero rates from curve points as ndarray.
pub fn zero_rates_from_points<T: FloatExt>(points: &[CurvePoint<T>]) -> Array1<T> {
  Array1::from_vec(
    points
      .iter()
      .map(|p| {
        if p.time > T::zero() {
          -p.discount_factor.ln() / p.time
        } else {
          T::zero()
        }
      })
      .collect(),
  )
}

#[cfg(test)]
mod tests {
  use super::*;

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
      assert!(df > 0.0 && df.is_finite(), "DF at t={t} non-positive or nan: {df}");
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
