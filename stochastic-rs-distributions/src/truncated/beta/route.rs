//! Beside an end, where `(1 − x)^{β−1}` is `e^{−(β−1)x}` to within 2⁻⁵³ out to the clipped end of the mass, the beta is
//! the truncated gamma of rate `β − 1`, which runs at a power-of-two unit; beside 1 its mirror, of rate `α − 1` in `1 − x`.

use std::f64::consts::LN_2;

use super::super::SimdTruncatedGamma;
use super::super::gamma::GammaKernel;
use super::super::gamma::unit_exponent;
use crate::special::ldexp;
use crate::truncated::quadrature::Cut;
use crate::truncated::quadrature::Filled;
use crate::truncated::quadrature::Point;
use crate::truncated::quadrature::Reflected;

/// `(e − 1)(ln(1 − x) + x)` within 2⁻⁵³: it is at most `(e − 1)x²/(2(1 − x))` above a unit exponent, below one the
/// whole `(1 − e)·x/(1 − x)` against a rate of 0.
fn holds(e: f64, x: f64) -> bool {
  let error = if e > 1.0 {
    (e - 1.0) * x * (x / (2.0 * (1.0 - x)))
  } else {
    (1.0 - e) * x / (1.0 - x)
  };
  error <= f64::EPSILON / 2.0
}

/// The scale `1/(e − 1)` of the rate; `MAX` below a unit exponent, whose factor is 1 to within `x/MAX`.
fn scale(e: f64) -> f64 {
  if e > 1.0 {
    (1.0 / (e - 1.0)).min(f64::MAX)
  } else {
    f64::MAX
  }
}

/// A point the gamma's clipped upper end lies past: half its reference point, the clamped mode or a step off a
/// singular 0.
fn reference(k: f64, theta: f64, lo: f64, up: f64) -> f64 {
  let centre = if k > 1.0 {
    (k - 1.0) * theta
  } else {
    (up - lo).min(theta) / 10.0
  };
  lo.max(0.5 * centre.min(up))
}

/// The gamma of rate `β − 1` on `[lo, up]` where it holds the kernel out to its clipped end; below a unit `β` nothing
/// falls, so that end is `up`.
pub(super) fn gamma_route(a: f64, b: f64, lo: f64, up: f64) -> Option<SimdTruncatedGamma<f64>> {
  let theta = scale(b);
  if holds(b, up) {
    return Some(SimdTruncatedGamma::new(a, theta, lo, up));
  }
  if b <= 1.0 || !holds(b, reference(a, theta, lo, up)) {
    return None;
  }
  let g = SimdTruncatedGamma::new(a, theta, lo, up);
  holds(b, g.clipped_end()?).then_some(g)
}

/// The gamma of rate `α − 1` in `1 − x`, read in `z = (x − 1)/2^p`, where its branch point is the upper end and the
/// cdf keeps the relative digits of the lower tail in `x`.
#[derive(Clone, Debug)]
pub(super) struct Mirror {
  t: Filled<Reflected<GammaKernel>>,
  p: i32,
  ln_unit: f64,
}

impl Mirror {
  /// Where it holds the kernel out to the clipped end of the mass in `1 − x`, and `1 − x` is exact on the support or
  /// the law below ½ underflows, which at or below a unit `α`, with no rate to fall by, it never does.
  pub(super) fn route(a: f64, b: f64, lo: f64, up: f64) -> Option<Self> {
    let exact = lo >= 0.5;
    let (theta, near, far) = (scale(a), 1.0 - up, 1.0 - lo);
    if (!exact && a <= 1.0)
      || (!holds(a, far) && (a <= 1.0 || !holds(a, reference(b, theta, near, far))))
    {
      return None;
    }
    let p = unit_exponent(theta.min(far));
    let s = if a > 1.0 {
      ldexp(1.0, -p) / (a - 1.0)
    } else {
      ldexp(f64::MAX, -p)
    };
    let kernel = Reflected(GammaKernel::new(b, s));
    let (z_lo, z_up) = (ldexp(lo - 1.0, -p), ldexp(up - 1.0, -p));
    let mut m = Self {
      p,
      ln_unit: f64::from(p) * LN_2,
      t: Filled::placed(kernel, Cut::at(z_lo), Cut::at(z_up)),
    };
    if m.t.narrow().is_some() {
      return None;
    }
    m.t.fill();
    let end = -ldexp(m.t.clipped().0, p);
    (holds(a, end) && (exact || (m.cdf(0.5) == 0.0 && m.pdf(0.5) == 0.0))).then_some(m)
  }

  /// `x − 1` in units, exact where `x ≥ ½`.
  fn units(&self, x: f64) -> f64 {
    ldexp(x - 1.0, -self.p)
  }

  fn at(&self, z: f64) -> f64 {
    1.0 + ldexp(z, self.p)
  }

  pub(super) fn pdf(&self, x: f64) -> f64 {
    self.t.density(self.units(x), self.ln_unit)
  }

  pub(super) fn cdf(&self, x: f64) -> f64 {
    self.t.cdf(Cut::at(self.units(x)))
  }

  pub(super) fn quantile(&self, q: f64) -> f64 {
    match self.t.quantile(q) {
      Point::Beside { s, sign, ln_t } => self.at(s) + sign * (ln_t + self.ln_unit).exp(),
      point => self.at(point.value()),
    }
  }

  /// `(mean, variance, skewness, excess kurtosis)`.
  pub(super) fn moments(&self) -> [f64; 4] {
    let [mean, variance, skewness, kurtosis] = self.t.moments(1.0, self.p);
    [self.at(mean), variance, skewness, kurtosis]
  }

  pub(super) fn entropy(&self) -> f64 {
    self.t.entropy() + self.ln_unit
  }
}
