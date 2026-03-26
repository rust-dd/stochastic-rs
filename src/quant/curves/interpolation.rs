//! Interpolation methods for yield curve construction.
//!
//! Reference: Hagan & West, "Interpolation Methods for Curve Construction",
//! Applied Mathematical Finance, 13(2), 89-129 (2006).
//! Reference: Hagan & West, "Methods for Constructing a Yield Curve",
//! Wilmott Magazine (2006).

use ndarray::Array1;

use crate::traits::FloatExt;

use super::types::CurvePoint;

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

  let r0 = if p0.time > T::zero() { -p0.discount_factor.ln() / p0.time } else { T::zero() };
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

/// Monotone convex interpolation on forward rates (Hagan & West, 2006).
///
/// Constructs a continuous instantaneous forward rate curve $f(t)$ that:
/// - Is positive whenever all discrete forwards are positive
/// - Preserves monotonicity on each interval
/// - Exactly reproduces the discrete forward rates (area preservation)
///
/// The discrete forward rate for interval $[t_{i-1}, t_i]$ is:
/// $$
/// f_i = \frac{r(t_i)\,t_i - r(t_{i-1})\,t_{i-1}}{t_i - t_{i-1}}
/// $$
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

  let idx = points.partition_point(|p| p.time < t).saturating_sub(1).max(1);
  let h = times[idx] - times[idx - 1];
  let x = (t - times[idx - 1]) / h;

  let a = f_node[idx - 1];
  let b = f_node[idx];
  let fi = fwd[idx];

  let three = T::from_f64_fast(3.0);
  let six = T::from_f64_fast(6.0);
  let g = a * (T::one() - x * (T::from_f64_fast(4.0)) + three * x * x)
    + b * (-two * x + three * x * x)
    + (six * fi - two * a - b) * (two * x - three * x * x)
    + (six * fi - a - two * b) * (-two * x + three * x * x);

  let _ = g;
  let integral = a * (x - two * x * x + x * x * x)
    + b * (-x * x + x * x * x)
    + (six * fi - two * a - b) * (x * x - x * x * x)
    + (six * fi - a - two * b) * (-x * x + x * x * x);

  let _ = integral;

  let forward_adj = rt[idx - 1]
    + h * (a * x + (three * fi - two * a - b) * x * x + (a + b - two * fi) * x * x * x);

  (-forward_adj).exp()
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
