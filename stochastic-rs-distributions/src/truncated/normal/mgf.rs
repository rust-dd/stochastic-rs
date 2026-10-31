//! The truncated normal's mgf: `e^{t·x_c − ½(c_s − c)(c_s + c)}` times the tilted and plain masses' ratio, `x_c` the
//! tilted reference point in the law's own coordinate and `c_s`, `c` the two reference points in standard units.

use super::SimdTruncatedNormal;
use super::kernels::TiltedKernel;
use crate::traits::SimdFloatExt;
use crate::truncated::local::Model;
use crate::truncated::local::limit_or;
use crate::truncated::quadrature::Truncation;

/// A tilt clamped at an end with a rate past this is `e^{−r·y − y²/2}` there, its curvature's share `1/r²` below 2⁻⁵⁴:
/// the exponential's mass, where the quadrature would meet a slope past the doubles.
const STEEP: f64 = 134_217_728.0;

impl<T: SimdFloatExt> SimdTruncatedNormal<T> {
  /// A tilt that clamps at a bound takes that bound as `x_c`, so no `μt` cancels against `σt·c_s`; at an infinite
  /// `t` the limit of `e^{tx}`.
  pub(super) fn mgf(&self, t: f64) -> Option<f64> {
    if t.is_nan() {
      return Some(f64::NAN);
    }
    let (lo, up) = (self.lower.to_f64()?, self.upper.to_f64()?);
    if t.is_infinite() {
      let end = if t > 0.0 { up } else { lo };
      return Some(if end * t > 0.0 { f64::INFINITY } else { 0.0 });
    }
    let plain = match self.analytics()? {
      Model::Full(plain) => plain,
      Model::Local(l) => return Some(l.mgf(t, (lo, up))),
    };
    let (mu, sigma) = (self.base.mean().to_f64()?, self.base.std_dev().to_f64()?);
    let (alpha, beta) = self.standardised_bounds()?;
    let (c, mass) = (plain.centre(), plain.mass());
    let s = sigma * t;
    let end = if s > beta {
      Some((up, beta))
    } else if s < alpha {
      Some((lo, alpha))
    } else {
      None
    };
    if s.is_infinite() && end.is_none() {
      // Toward an open end the exponent is `σt(μ/σ + σt/2)`, past any double unless the bracket vanishes.
      let lead = mu / sigma + 0.5 * sigma * t;
      return Some(if lead * t > 0.0 { f64::INFINITY } else { 0.0 });
    }
    let rate = end.map(|(_, z)| (s - z).abs());
    let (cs, ln_ratio) = match (end, rate) {
      (Some((_, z)), Some(rate)) if rate >= STEEP => {
        let reach = -(-rate * (beta - alpha)).exp_m1();
        let tilted = reach / rate;
        let ln_ratio = if tilted >= f64::MIN_POSITIVE {
          (tilted / mass).ln()
        } else if rate.is_finite() {
          reach.ln() - rate.ln() - mass.ln()
        } else {
          reach.ln() - ln_rate(s, z, sigma, t) - mass.ln()
        };
        (z, ln_ratio)
      }
      _ => {
        let tilted = Truncation::mass_only(TiltedKernel { s, c: 0.0 }, alpha, beta);
        (tilted.centre(), (tilted.mass() / mass).ln())
      }
    };
    let term = |k: f64, f: f64, g: f64| {
      (
        k * f.signum() * g.signum(),
        k.abs().ln() + f.abs().ln() + g.abs().ln(),
      )
    };
    let (h1, h2) = (0.5 * cs - 0.5 * c, 0.5 * cs + 0.5 * c);
    let exponent = if cs == alpha || cs == beta {
      let x_c = if cs == alpha { lo } else { up };
      limit_or(
        t * x_c - (h1 * h2) * 2.0,
        &[term(1.0, t, x_c), term(-2.0, h1, h2)],
      )
    } else if cs == s {
      limit_or(
        mu * t + (0.5 * (s * s) + 0.5 * (c * c)),
        &[term(1.0, t, mu), term(0.5, s, s), term(0.5, c, c)],
      )
    } else {
      limit_or(
        mu * t + (s * cs - 0.5 * (cs - c) * (cs + c)),
        &[term(1.0, t, mu), term(1.0, s, cs), term(-2.0, h1, h2)],
      )
    };
    Some((exponent + ln_ratio).exp())
  }
}

/// `ln|s − z|` for a tilt `s = σt` clamped at `z` where the difference passes the doubles: `s` and `z` of opposite
/// signs, or `s` itself past them, which takes `σ > 1`, so `ln σ + ln|t − z/σ|`.
fn ln_rate(s: f64, z: f64, sigma: f64, t: f64) -> f64 {
  if s.is_finite() {
    s.abs().ln() + (z.abs() / s.abs()).ln_1p()
  } else {
    sigma.ln() + (t - z / sigma).abs().ln()
  }
}
