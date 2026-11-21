//! Reference: Abramowitz, M., Stegun, I.A. (1964), NBS AMS 55, 6.4.6, 6.4.7, 6.4.12.

use std::f64::consts::PI;

use super::is_pole;
use super::sinpi;

/// Trigamma `ψ′(x)`, NaN at the poles: the recurrence lifts `x` to at least 12, where the series
/// runs to `1/x¹⁵`, and below ½ the reflection `ψ′(x) = π²/sin²(πx) − ψ′(1 − x)` holds.
pub fn trigamma(x: f64) -> f64 {
  if is_pole(x) {
    return f64::NAN;
  }
  if x < 0.5 {
    let s = sinpi(x);
    return PI * PI / (s * s) - trigamma(1.0 - x);
  }
  let mut y = x;
  let mut value = 0.0;
  while y < 12.0 {
    value += 1.0 / (y * y);
    y += 1.0;
  }
  let inverse = 1.0 / y;
  let inverse_squared = inverse * inverse;
  value + inverse + inverse_squared / 2.0 + inverse_squared * inverse / 6.0
    - inverse_squared.powi(2) * inverse / 30.0
    + inverse_squared.powi(3) * inverse / 42.0
    - inverse_squared.powi(4) * inverse / 30.0
    + 5.0 * inverse_squared.powi(5) * inverse / 66.0
    - 691.0 * inverse_squared.powi(6) * inverse / 2_730.0
    + 7.0 * inverse_squared.powi(7) * inverse / 6.0
}
