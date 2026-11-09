//! A law below its kernel's coordinate resolution, as its local model `e^{ψy + κy²/2}`,
//! `y = x − r`: a truncated exponential, normal or uniform in its scale `2^p`, as any law.

use std::f64::consts::LN_2;

use super::quadrature::Cut;
use super::quadrature::Filled;
use super::quadrature::Kernel;
use super::quadrature::Point;
use super::quadrature::Terms;
use super::quadrature::Truncation;
use super::quadrature::accurate_sum;
use crate::special::ldexp;

/// `sign·e^{ln}`: a derivative the doubles may not hold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Signed {
  sign: f64,
  ln: f64,
}

impl Signed {
  pub(super) const ZERO: Self = Self {
    sign: 0.0,
    ln: f64::NEG_INFINITY,
  };

  pub(super) fn new(sign: f64, ln: f64) -> Self {
    if sign == 0.0 || ln == f64::NEG_INFINITY {
      Self::ZERO
    } else {
      Self { sign, ln }
    }
  }

  /// `sign(x)·|x|/e^{ln_scale}`, where `x` itself is a double.
  pub(super) fn over(x: f64, ln_scale: f64) -> Self {
    Self::new(
      x.signum() * f64::from(u8::from(x != 0.0)),
      x.abs().ln() - ln_scale,
    )
  }

  /// The sum, in logs: a difference cancels only to the digits its larger term carries.
  pub(super) fn plus(self, other: Self) -> Self {
    let (big, small) = if self.ln >= other.ln {
      (self, other)
    } else {
      (other, self)
    };
    if small.sign == 0.0 {
      return big;
    }
    let d = (small.ln - big.ln).exp();
    if big.sign == small.sign {
      Self::new(big.sign, big.ln + d.ln_1p())
    } else {
      Self::new(big.sign, big.ln + (-d).ln_1p())
    }
  }

  fn at(self, ln_scale: f64) -> f64 {
    self.sign * (self.ln + ln_scale).exp()
  }
}

/// `ln` of a bound on `|e1|·|ln(1 + z) − z + z²/2|`, the term `e1·ln(1 + z)` past its quadratic:
/// the derivative `z²/(1 + z)` bounds it by `|z|³/(3 min(1, 1 + z))`.
pub(super) fn ln_cubic(e1: f64, z: f64) -> f64 {
  if e1 == 0.0 || z == 0.0 {
    return f64::NEG_INFINITY;
  }
  if z <= -1.0 || z.is_nan() {
    return f64::INFINITY;
  }
  e1.abs().ln() + 3.0 * z.abs().ln() - 3f64.ln() - z.min(0.0).ln_1p()
}

/// `e^{a u + b u²/2}` relative to `u = 0`, about the reference point `c`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct QuadKernel {
  a: f64,
  b: f64,
  c: f64,
}

impl Kernel for QuadKernel {
  const QUADRATIC: bool = true;

  fn ln_k(&self, u: f64) -> f64 {
    u * (self.a + 0.5 * self.b * u)
  }

  fn anchor(&mut self, c: Cut, _at_mode: bool) {
    self.c = c.x;
  }

  /// An infinite offset lies where a proper law's kernel has vanished, past `b = −0`'s `0·∞`.
  fn ln_rel(&self, _u: f64, y: f64, _branch: Option<(f64, f64)>) -> f64 {
    if y.is_infinite() {
      return f64::NEG_INFINITY;
    }
    y * (self.b.mul_add(self.c, self.a) + 0.5 * self.b * y)
  }

  fn d1(&self, u: f64, _y: f64) -> f64 {
    self.b.mul_add(u, self.a)
  }

  fn d2(&self, _u: f64, _y: f64) -> f64 {
    self.b
  }

  fn mode(&self) -> Cut {
    Cut::at(if self.b < 0.0 {
      -self.a / self.b
    } else if self.a == 0.0 {
      0.0
    } else {
      self.a.signum() * f64::INFINITY
    })
  }

  fn branch_points(&self) -> [Option<(f64, f64)>; 2] {
    [None, None]
  }

  fn length_scale(&self) -> f64 {
    1.0
  }

  fn support(&self) -> (f64, f64) {
    (f64::NEG_INFINITY, f64::INFINITY)
  }

  fn base_moments(&self) -> [f64; 4] {
    [-self.a / self.b, (-1.0 / self.b).sqrt(), 0.0, 0.0]
  }
}

/// A law's integral: the engine in the kernel's own coordinate, or the local model of a law narrower than it resolves.
#[derive(Clone, Debug)]
pub(super) enum Model<K> {
  Full(Filled<K>),
  Local(Local),
}

impl<K: Kernel> Model<K> {
  /// The engine's integral of `kernel` on `[lo, up]`, or the local model where the engine has no
  /// normal width and the model exists; the engine is built and summed where the model holds it.
  pub(super) fn new(
    kernel: K,
    lo: impl Into<Cut>,
    up: impl Into<Cut>,
    local: impl FnOnce(f64) -> Option<Local>,
  ) -> Self {
    let mut model = Self::Full(Filled::placed(kernel, lo.into(), up.into()));
    if let Self::Full(t) = &mut model {
      match t.narrow().and_then(local) {
        Some(l) => model = Self::Local(l),
        None => t.fill(),
      }
    }
    model
  }
}

/// The model of a narrow law about its reference point `r`, in units of `2^p`.
#[derive(Clone, Debug)]
pub(super) struct Local {
  r: f64,
  p: i32,
  t: Filled<QuadKernel>,
  /// The interval as offsets from `r`, and the log-kernel's slope and curvature there: a tilt's own model reads them.
  offsets: (f64, f64),
  slope: Signed,
  curvature: Signed,
}

impl Local {
  /// `e^{ψy + κy²/2}` on `[r + lo, r + up]` in units of its shortest e-fold, width or interval;
  /// `None` without one, or where `ln_cubic(y)`, the remainder's log bound, passes ε at a clip.
  pub(super) fn new(
    r: f64,
    (lo, up): (f64, f64),
    slope: Signed,
    curvature: Signed,
    ln_cubic: impl Fn(f64) -> f64,
  ) -> Option<Self> {
    // From halves only where the width overflows: half a subnormal width rounds to 0.
    let ln_width = if (up - lo).is_finite() {
      (up - lo).ln()
    } else {
      (0.5 * up - 0.5 * lo).ln() + LN_2
    };
    let ln_len = (-slope.ln).min(-0.5 * curvature.ln).min(ln_width);
    if !ln_len.is_finite() {
      return None;
    }
    let p = (ln_len / LN_2).floor() as i32;
    let ln_unit = f64::from(p) * LN_2;
    let (a, b) = (slope.at(ln_unit), curvature.at(2.0 * ln_unit));
    // A kernel that rises toward an open end with no curvature the unit resolves has no finite mass.
    if b == 0.0 && ((a >= 0.0 && up == f64::INFINITY) || (a <= 0.0 && lo == f64::NEG_INFINITY)) {
      return None;
    }
    let t = Filled::new(QuadKernel { a, b, c: 0.0 }, ldexp(lo, -p), ldexp(up, -p));
    let (ya, yb) = t.clipped();
    if ln_cubic(ldexp(ya, p)).max(ln_cubic(ldexp(yb, p))) > f64::EPSILON.ln() {
      return None;
    }
    Some(Self {
      r,
      p,
      t,
      offsets: (lo, up),
      slope,
      curvature,
    })
  }

  fn units(&self, x: f64) -> f64 {
    ldexp(x - self.r, -self.p)
  }

  fn ln_unit(&self) -> f64 {
    f64::from(self.p) * LN_2
  }

  /// `(mean, variance, skewness, excess kurtosis)`.
  pub(super) fn moments(&self) -> [f64; 4] {
    let (frame, [unit, offset], [variance, skewness, kurtosis]) =
      self.t.located_moments(1.0, self.p);
    [self.at(frame, unit * offset), variance, skewness, kurtosis]
  }

  /// `r + 2^p(frame + d)`, summed in units of the frame's exact parts, scaled once, so a point far
  /// nearer 0 than both keeps its digits; where `r` in units passes the doubles, it is `r`.
  fn at(&self, frame: Cut, d: f64) -> f64 {
    let (s, lo, t) = frame.parts();
    let r = ldexp(self.r, -self.p);
    if r.is_finite() {
      ldexp(accurate_sum([r, s, lo, t, d]).0, self.p)
    } else {
      accurate_sum([self.r, ldexp(s, self.p), ldexp(lo + (t + d), self.p)]).0
    }
  }

  pub(super) fn entropy(&self) -> f64 {
    self.t.entropy() + self.ln_unit()
  }

  /// The density over a further `e^{ln_scale}`.
  pub(super) fn pdf(&self, x: f64, ln_scale: f64) -> f64 {
    self.t.density(self.units(x), self.ln_unit() + ln_scale)
  }

  pub(super) fn cdf(&self, x: f64) -> f64 {
    self.t.cdf(Cut::at(self.units(x)))
  }

  pub(super) fn quantile(&self, q: f64) -> f64 {
    match self.t.quantile(q) {
      Point::Mid { centre, width, t } => self.at(centre, width * t),
      point => self.r + ldexp(point.value(), self.p),
    }
  }

  /// `E e^{tX}`: the tilted law is the model at `x_c`, its log-kernel's peak on the interval (a
  /// bound where it clamps), in its unit; the exponent `t·x_c` plus the kernel's rise from `r`.
  pub(super) fn mgf(&self, t: f64, (lo, up): (f64, f64)) -> f64 {
    let tilt = self.slope.plus(Signed::over(t, 0.0));
    // `(ψ + t)/|κ|`; a tilt that cancels the slope peaks at `r`.
    let peak = if tilt.sign == 0.0 {
      0.0
    } else {
      tilt.sign * (tilt.ln - self.curvature.ln).exp()
    };
    let (x_c, y_c) = if peak <= self.offsets.0 {
      (lo, self.offsets.0)
    } else if peak >= self.offsets.1 {
      (up, self.offsets.1)
    } else {
      (self.r + peak, peak)
    };
    let ln_y = y_c.abs().ln();
    let along = Signed::new(self.curvature.sign * y_c.signum(), self.curvature.ln + ln_y);
    let Some(tilted) = Local::new(
      x_c,
      (lo - x_c, up - x_c),
      tilt.plus(along),
      self.curvature,
      |_| f64::NEG_INFINITY,
    ) else {
      // A tilted kernel with no finite mass: it rises toward an open end, so `e^{tx}` does.
      return f64::INFINITY;
    };
    let ln_k = |model: &Truncation<QuadKernel>| {
      let (QuadKernel { a, b, .. }, c) = (*model.kernel(), model.centre());
      c * (a + 0.5 * b * c)
    };
    let ratio = (tilted.t.mass() / self.t.mass()).ln() + f64::from(tilted.p - self.p) * LN_2;
    let mut terms = Terms::new();
    terms.product(t, x_c, 0);
    for (sign, ln) in [
      (self.slope.sign * y_c.signum(), self.slope.ln + ln_y),
      (self.curvature.sign, self.curvature.ln + 2.0 * ln_y - LN_2),
    ] {
      // `e^{ln}` as `m·2^p`, so a term past the doubles enters the sum at its own scale.
      let p = (ln / LN_2).floor();
      if sign != 0.0 && p.is_finite() {
        terms.scaled(sign * (ln - p * LN_2).exp(), p as i32);
      }
    }
    for v in [ln_k(&tilted.t), -ln_k(&self.t), ratio] {
      terms.scaled(v, 0);
    }
    terms.total().exp()
  }
}
