//! The truncated normal between its own coordinate and the engine's: bounds held at their own doubles and their width,
//! points beside a bound, and answers mapped back from the exact point nearest them.

use super::NormalKernel;
use super::SimdTruncatedNormal;
use crate::traits::SimdFloatExt;
use crate::truncated::quadrature::Cut;
use crate::truncated::quadrature::Offset;
use crate::truncated::quadrature::Point;
use crate::truncated::quadrature::Truncation;
use crate::truncated::quadrature::accurate_sum;
use crate::truncated::quadrature::two_sum;

/// `2⁻¹⁰³²`: a subnormal below it is `2⁻¹⁰⁷⁴` coarse past 2e-13 of itself.
pub(super) const SUBNORMAL_DIGITS: f64 = f64::MIN_POSITIVE / 1024.0;

impl<T: SimdFloatExt> SimdTruncatedNormal<T> {
  /// The mean from the point its sums were taken about, exact in the law's coordinate: a bound, `μ`, or the bounds'
  /// midpoint for a flat kernel, whose standardised image no rounded width maps back.
  pub(super) fn mean_from(
    &self,
    t: &Truncation<NormalKernel>,
    frame: Cut,
    offset: [f64; 2],
  ) -> Option<f64> {
    if t.about_midpoint() {
      return self.anchored(self.midpoint()?, offset);
    }
    if self.tail.is_none() {
      let mu = self.base.mean().to_f64()?;
      return self.anchored((mu, 0.0), [1.0, self.mode_offset(t)?]);
    }
    self.anchored((self.located(Point::From(frame))?, 0.0), offset)
  }

  /// `E[Z] = (k(α) − k(β))/∫k` for `k = e^{−z²/2}`, an interval about `μ`: summed odd powers cancel to ε of the width
  /// where the mean lies near `μ`, this keeps its digits; `α ± β` from the bounds themselves.
  fn mode_offset(&self, t: &Truncation<NormalKernel>) -> Option<f64> {
    let (mu, sigma) = (self.base.mean().to_f64()?, self.base.std_dev().to_f64()?);
    let (lo, up) = (self.lower.to_f64()?, self.upper.to_f64()?);
    let [alpha, beta, _] = self.bounds;
    let k = |z: f64| (-0.5 * z * z).exp();
    // `(β² − α²)/2 = (β − α)(β + α)/2`, each factor from the bounds' own doubles.
    let half_gap = 0.5 * ((up - lo) / sigma) * (accurate_sum([lo, up, -mu, -mu]).0 / sigma);
    let diff = if half_gap.is_nan() || !alpha.is_finite() || !beta.is_finite() {
      k(alpha) - k(beta)
    } else if half_gap >= 0.0 {
      -k(alpha) * (-half_gap).exp_m1()
    } else {
      k(beta) * half_gap.exp_m1()
    };
    Some(diff / t.mass())
  }

  /// `hi + lo + σ·unit·t`, from halves where it overflows short of its value.
  pub(super) fn anchored(&self, (hi, lo): (f64, f64), [unit, t]: [f64; 2]) -> Option<f64> {
    let sigma = self.base.std_dev().to_f64()?;
    let x = hi + (lo + (sigma * unit) * t);
    Some(if x.is_finite() {
      x
    } else {
      2.0 * (0.5 * hi + (0.5 * lo + (0.5 * sigma * unit) * t))
    })
  }

  /// The bounds' midpoint and its rest, the law's flat reference point.
  pub(super) fn midpoint(&self) -> Option<(f64, f64)> {
    Some(midpoint(self.lower.to_f64()?, self.upper.to_f64()?))
  }

  /// From halves where `μ + σz` overflows short of its value.
  #[inline]
  pub(super) fn unstandardise(&self, z: f64) -> f64 {
    let (mu, sigma) = (
      self.base.mean().to_f64().unwrap(),
      self.base.std_dev().to_f64().unwrap(),
    );
    let x = mu + sigma * z;
    if x.is_finite() || !z.is_finite() {
      x
    } else {
      2.0 * (0.5 * mu + (0.5 * sigma) * z)
    }
  }

  pub(super) fn standardise(&self, x: f64) -> Option<f64> {
    Some(standardised(
      x,
      self.base.mean().to_f64()?,
      self.base.std_dev().to_f64()?,
    ))
  }

  /// `x` beside a bound nearer it than `μ` by its own offset `(x − bound)/σ`, so a point next to an end keeps the
  /// digits its distance from it has, which `(x − μ)/σ` rounds away.
  pub(super) fn cut(&self, x: f64) -> Option<Cut> {
    let (lo, up) = (self.lower.to_f64()?, self.upper.to_f64()?);
    let (bound, held) = if x - lo <= up - x {
      (lo, self.held().0)
    } else {
      (up, self.held().1)
    };
    let d = x - bound;
    if d.abs() < (x - self.base.mean().to_f64()?).abs() {
      let t = d / self.base.std_dev().to_f64()?;
      if t.is_finite() {
        return Some(held.shifted(t));
      }
    }
    Some(Cut::at(self.standardise(x)?))
  }

  /// `(x − lower)/σ` as `m·2^q` where the subnormal it rounds to holds fewer digits than 2e-13, which `cut` would
  /// take.
  pub(super) fn zone_offset(&self, x: f64) -> Option<Offset> {
    let (lo, sigma) = (self.lower.to_f64()?, self.base.std_dev().to_f64()?);
    let d = x - lo;
    if !(lo.is_finite() && d > 0.0 && d / sigma < SUBNORMAL_DIGITS) {
      return None;
    }
    let (Offset { m: md, q: qd }, Offset { m: ms, q: qs }) = (Offset::of(d), Offset::of(sigma));
    let m = md / ms;
    Some(if m < 1.0 {
      Offset {
        m: 2.0 * m,
        q: qd - qs - 1,
      }
    } else {
      Offset { m, q: qd - qs }
    })
  }

  /// A point the engine reports: one held beside a bound maps back from that bound itself, one beside a flat kernel's
  /// midpoint from the bounds' midpoint.
  pub(super) fn located(&self, point: Point) -> Option<f64> {
    let (lo, up) = (self.lower.to_f64()?, self.upper.to_f64()?);
    let sigma = self.base.std_dev().to_f64()?;
    if let Point::Mid { width, t, .. } = point {
      return self.anchored(self.midpoint()?, [width, t]);
    }
    if let Point::From(cut) = point {
      let (s, rest, t) = cut.parts();
      let (at_lo, at_up) = self.held();
      for (bound, held) in [(lo, at_lo), (up, at_up)] {
        let (hs, hrest, _) = held.parts();
        let x = bound + sigma * t;
        if bound.is_finite() && (s, rest) == (hs, hrest) && x.is_finite() {
          return Some(x);
        }
      }
    }
    if let Point::Beside { s, sign, ln_t } = point
      && s == self.held().0.x
      && lo.is_finite()
    {
      return Some(lo + sign * (ln_t + self.ln_sigma).exp());
    }
    Some(self.unstandardise(point.value()))
  }

  /// The bounds' standardised doubles.
  pub(super) fn standardised_bounds(&self) -> (f64, f64) {
    (self.bounds[0], self.bounds[1])
  }

  /// The engine's bounds at their own doubles; off `μ` the far one is also held beside the near one by the width, which
  /// the two bounds' roundings would change by an ulp of their place.
  pub(super) fn held(&self) -> (Cut, Cut) {
    let [alpha, beta, width] = self.bounds;
    let (lo, up) = (Cut::at(alpha), Cut::at(beta));
    if width.is_nan() {
      (lo, up)
    } else if alpha > 0.0 {
      (lo, Cut::held(beta, alpha, width))
    } else {
      (Cut::held(alpha, beta, -width), up)
    }
  }
}

/// `(x − μ)/σ`, from halves where `x − μ` overflows; an infinite `x` stays infinite.
#[inline]
pub(super) fn standardised(x: f64, mu: f64, sigma: f64) -> f64 {
  let d = x - mu;
  if d.is_infinite() && x.is_finite() {
    halved(x, mu, sigma)
  } else {
    d / sigma
  }
}

/// Out of line, so the common path computes one quotient, not both.
#[cold]
#[inline(never)]
fn halved(x: f64, mu: f64, sigma: f64) -> f64 {
  (0.5 * x - 0.5 * mu) / (0.5 * sigma)
}

/// `(up − lo)/σ` for an interval off `μ`, from halves where `up − lo` overflows; NaN about `μ` or past the doubles.
pub(super) fn width(alpha: f64, beta: f64, lo: f64, up: f64, sigma: f64) -> f64 {
  let w = if (up - lo).is_finite() {
    (up - lo) / sigma
  } else {
    (0.5 * up - 0.5 * lo) / (0.5 * sigma)
  };
  let off = alpha > 0.0 || beta < 0.0;
  if off && alpha.is_finite() && beta.is_finite() && w.is_finite() {
    w
  } else {
    f64::NAN
  }
}

/// `(lo + up)/2` as a double and its rest, from halves where the sum overflows.
fn midpoint(lo: f64, up: f64) -> (f64, f64) {
  if (lo + up).is_finite() {
    let (s, e) = two_sum(lo, up);
    (0.5 * s, 0.5 * e)
  } else {
    two_sum(0.5 * lo, 0.5 * up)
  }
}
