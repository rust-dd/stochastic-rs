//! A law narrower than its kernel's coordinate resolves, as its local model `e^{ψy + κy²/2}`, `y = x − r`: a truncated
//! exponential, normal or uniform in its own scale `2^p`, which the engine integrates as an ordinary law.

use std::f64::consts::LN_2;

use super::quadrature::Cut;
use super::quadrature::Kernel;
use super::quadrature::Truncation;
use super::quadrature::ldexp;

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

/// `ln` of a bound on `|e1|·|ln(1 + z) − z + z²/2|`, the term `e1·ln(1 + z)` past its quadratic: the derivative
/// `z²/(1 + z)` bounds it by `|z|³/(3 min(1, 1 + z))`.
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
  Full(Truncation<K>),
  Local(Local),
}

impl<K: Kernel> Model<K> {
  /// The engine's integral, or the local model where it reports no normal width and the model exists.
  pub(super) fn of(t: Truncation<K>, local: impl FnOnce(f64) -> Option<Local>) -> Self {
    match t.narrow().and_then(local) {
      Some(l) => Self::Local(l),
      None => Self::Full(t),
    }
  }
}

/// The model of a narrow law about its reference point `r`, in units of `2^p`.
#[derive(Clone, Debug)]
pub(super) struct Local {
  r: f64,
  p: i32,
  t: Truncation<QuadKernel>,
  /// The interval as offsets from `r`, and the log-kernel's slope and curvature there: a tilt's own model reads them.
  offsets: (f64, f64),
  slope: Signed,
  curvature: Signed,
}

impl Local {
  /// `e^{ψy + κy²/2}` on `[r + lo, r + up]` in units of the shortest of its e-fold length, normal width and interval;
  /// `None` where no such scale exists, or where `ln_cubic(y)`, its remainder's log bound, passes ε at a clipped end.
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
    let t = Truncation::new(QuadKernel { a, b, c: 0.0 }, ldexp(lo, -p), ldexp(up, -p));
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
    let [mean, variance, skewness, kurtosis] = self.t.moments(1.0);
    [
      self.r + ldexp(mean, self.p),
      ldexp(variance, 2 * self.p),
      skewness,
      kurtosis,
    ]
  }

  pub(super) fn entropy(&self) -> f64 {
    self.t.entropy() + self.ln_unit()
  }

  /// The density over a further `e^{ln_scale}`.
  pub(super) fn pdf(&self, x: f64, ln_scale: f64) -> f64 {
    self.t.density(self.units(x), self.ln_unit() + ln_scale)
  }

  pub(super) fn cdf(&self, x: f64) -> f64 {
    self.t.cdf(self.units(x))
  }

  pub(super) fn quantile(&self, q: f64) -> f64 {
    self.r + ldexp(self.t.quantile(q).value(), self.p)
  }

  /// `E e^{tX}`: the tilted law is the model at `x_c`, where its log-kernel peaks on the interval (the law's own bound
  /// where it clamps), in its own unit; the exponent is `t·x_c` plus the plain kernel's rise from `r` to `x_c`.
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
    let (l1, l2) = (self.slope.ln + ln_y, self.curvature.ln + 2.0 * ln_y - LN_2);
    let (q1, q2) = (
      self.slope.sign * y_c.signum() * l1.exp(),
      self.curvature.sign * l2.exp(),
    );
    let head = limit_or(
      t * x_c + (q1 + q2),
      &[
        (t.signum() * x_c.signum(), t.abs().ln() + x_c.abs().ln()),
        (q1.signum(), l1),
        (q2.signum(), l2),
      ],
    );
    let ln_k = |model: &Truncation<QuadKernel>| {
      let (QuadKernel { a, b, .. }, c) = (*model.kernel(), model.centre());
      c * (a + 0.5 * b * c)
    };
    let ratio = (tilted.t.mass() / self.t.mass()).ln() + f64::from(tilted.p - self.p) * LN_2;
    (head + (ln_k(&tilted.t) - ln_k(&self.t)) + ratio).exp()
  }
}

/// `plain`, or where its terms met `∞ − ∞` the limit the term of largest log-magnitude gives; a term is `(sign, ln|·|)`.
pub(super) fn limit_or(plain: f64, terms: &[(f64, f64)]) -> f64 {
  if !plain.is_nan() {
    return plain;
  }
  let top = terms.iter().fold((0.0, f64::NEG_INFINITY), |best, &t| {
    if t.1 > best.1 { t } else { best }
  });
  top.0 * f64::INFINITY
}
