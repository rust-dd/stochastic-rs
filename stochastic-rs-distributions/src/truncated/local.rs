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
}

impl Local {
  /// `e^{ψy + κy²/2}` on `[r + lo, r + up]` in units of the shortest of its e-fold length, normal width and interval,
  /// so its coefficients are at most 1 and its interval at least 1 long; `None` where no such scale exists.
  pub(super) fn new(
    r: f64,
    (lo, up): (f64, f64),
    slope: Signed,
    curvature: Signed,
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
    let kernel = QuadKernel {
      a: slope.at(ln_unit),
      b: curvature.at(2.0 * ln_unit),
      c: 0.0,
    };
    let t = Truncation::new(kernel, ldexp(lo, -p), ldexp(up, -p));
    Some(Self { r, p, t })
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

  /// `E e^{tX}` as `e^{t·x_c}` times the tilted model's mass at its centre `x_c` over the model's: `t·x_c` is formed
  /// once, so no `t·r` cancels against the tilt's own offset.
  pub(super) fn mgf(&self, t: f64) -> f64 {
    let QuadKernel { a, b, .. } = *self.t.kernel();
    let tau = ldexp(t, self.p);
    let (lo, up) = self.t.bounds();
    // The tilted exponential's mass diverges toward an open end it rises to.
    if b == 0.0
      && ((a + tau >= 0.0 && up == f64::INFINITY) || (a + tau <= 0.0 && lo == f64::NEG_INFINITY))
    {
      return f64::INFINITY;
    }
    let tilted = Truncation::mass_only(
      QuadKernel {
        a: a + tau,
        b,
        c: 0.0,
      },
      lo,
      up,
    );
    let (c, ct) = (self.t.centre(), tilted.centre());
    let ln_k = |u: f64| u * (a + 0.5 * b * u);
    let exponent = t * (self.r + ldexp(ct, self.p)) + ln_k(ct) - ln_k(c)
      + (tilted.mass().ln() - self.t.mass().ln());
    exponent.exp()
  }
}
