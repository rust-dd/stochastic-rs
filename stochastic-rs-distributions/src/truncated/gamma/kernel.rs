//! The truncated gamma's kernel in units of `2^p`, the least power of two at or above θ.

use crate::truncated::quadrature::Cut;
use crate::truncated::quadrature::Kernel;
use crate::truncated::quadrature::ln_1p_mx;
use crate::truncated::quadrature::ln_ratio;
use crate::truncated::quadrature::two_prod;

/// `u^{k−1} e^{−u/s}` on `[0, ∞)` about the reference point `c`; a unit shape has no log term, whose `0·ln 0` would
/// read NaN at 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct GammaKernel {
  k: f64,
  /// θ in the kernel's unit: at most 1 where θ is the shorter length, else above 1, `+∞` where `upper` lies so far
  /// below θ that `e^{−u/s}` is 1 on the interval.
  s: f64,
  c: f64,
  /// The rest of the exact reference point past `c`, which the offsets `y` are taken from.
  c_lo: f64,
  ln_c: f64,
  /// `(k − 1)/c − 1/s`, the slope at `c`, `None` where it overflows.
  slope: Option<f64>,
  /// `k − 1 ≤ 3e3`: the plain form's measured error, 1.6e-14·√(k − 1) out to the farthest normal pdf, stays in 1e-12.
  plain: bool,
}

impl GammaKernel {
  #[inline]
  pub(super) fn new(k: f64, s: f64) -> Self {
    Self {
      k,
      s,
      c: 0.0,
      c_lo: 0.0,
      ln_c: f64::NEG_INFINITY,
      slope: None,
      plain: k - 1.0 <= 3e3,
    }
  }
}

impl GammaKernel {
  /// `(k − 1)(ln(x/c) − y/c) + slope·y`, the first part `ln_1p_mx` down to `x = c/4`, the quotient's log below.
  #[inline(never)]
  fn ln_rel_exact(&self, x: f64, y: f64, slope: f64) -> f64 {
    let z = y / self.c;
    if z < -0.75 {
      (self.k - 1.0) * (ln_ratio(x, self.c, self.ln_c) - z) + slope * y
    } else {
      // `y/(c + c_lo)` past its rounding, which the left tail's `1/(1 + z)` amplifies; far out it would only overflow.
      let z_lo = if z.abs() <= 1.0 {
        ((-z).mul_add(self.c, y) - z * self.c_lo) / self.c
      } else {
        0.0
      };
      (self.k - 1.0) * (ln_1p_mx(z) - z_lo * z / (1.0 + z)) + slope * y
    }
  }
}

impl Kernel for GammaKernel {
  #[inline]
  fn ln_k(&self, x: f64) -> f64 {
    if x < 0.0 {
      return f64::NEG_INFINITY;
    }
    let lead = if self.k == 1.0 {
      0.0
    } else {
      (self.k - 1.0) * x.ln()
    };
    lead - x / self.s
  }

  /// Above a unit shape the slope from `(k − 1)s − c` at the exact centre `c.x + c_lo`, `k − 1` exact and the product
  /// by FMA, so it cancels to its true value next to the mode, 0 at it; below one the two terms share a sign.
  #[inline]
  fn anchor(&mut self, c: Cut, at_mode: bool) {
    let (hi, lo) = c.near();
    self.c = c.x;
    self.c_lo = (hi - c.x) + lo;
    self.ln_c = c.x.ln();
    let slope = if self.k == 1.0 {
      -1.0 / self.s
    } else if self.k < 1.0 {
      (self.k - 1.0) / c.x - 1.0 / self.s
    } else if at_mode {
      0.0
    } else if self.s.is_infinite() {
      (self.k - 1.0) / c.x
    } else {
      let k1 = self.k - 1.0;
      let k1_lo = (self.k - k1) - 1.0;
      let (p, p_lo) = two_prod(k1, self.s);
      (((p - c.x) + p_lo) + (k1_lo * self.s - self.c_lo)) / (c.x * self.s)
    };
    self.slope = slope.is_finite().then_some(slope);
  }

  /// The plain `(k − 1) ln(x/c) − y/s`, `ln_1p` within half of `c` and else the quotient's log, which a `y/c` past the
  /// doubles never reaches; past the shape bound, where the slope is finite, the cancellation-free form out of line.
  #[inline]
  fn ln_rel(&self, x: f64, y: f64, _branch: Option<(f64, f64)>) -> f64 {
    if self.k == 1.0 {
      return -y / self.s;
    }
    match self.slope {
      Some(slope) if !self.plain => self.ln_rel_exact(x, y, slope),
      _ => {
        let z = y / self.c;
        let ln_ratio = if z.abs() > 0.5 {
          ln_ratio(x, self.c, self.ln_c)
        } else {
          z.ln_1p()
        };
        (self.k - 1.0) * ln_ratio - y / self.s
      }
    }
  }

  #[inline]
  fn ln_rel_regular(&self, _x: f64, y: f64, _s: f64) -> f64 {
    -(self.k - 1.0) * self.ln_c - y / self.s
  }

  #[inline]
  fn d1(&self, x: f64, y: f64) -> f64 {
    match self.slope {
      _ if self.k == 1.0 => -1.0 / self.s,
      Some(slope) if self.k > 1.0 => slope - (self.k - 1.0) * (y / self.c) / x,
      _ => (self.k - 1.0) / x - 1.0 / self.s,
    }
  }

  #[inline]
  fn d2(&self, x: f64, _y: f64) -> f64 {
    if self.k == 1.0 {
      0.0
    } else {
      -(self.k - 1.0) / (x * x)
    }
  }

  /// `(k − 1)s` to a double-double: `k − 1` is exact below 2⁵³ and past it carries its own rounding.
  #[inline]
  fn mode(&self) -> Cut {
    if self.k <= 1.0 {
      return Cut::at(0.0);
    }
    let k1 = self.k - 1.0;
    let k1_lo = (self.k - k1) - 1.0;
    let hi = k1 * self.s;
    if hi.is_infinite() {
      return Cut::at(hi);
    }
    Cut::exact(hi, k1.mul_add(self.s, -hi) + k1_lo * self.s)
  }

  #[inline]
  fn branch_points(&self) -> [Option<(f64, f64)>; 2] {
    [(self.k != 1.0).then_some((0.0, self.k)), None]
  }

  #[inline]
  fn length_scale(&self) -> f64 {
    self.s
  }

  #[inline]
  fn support(&self) -> (f64, f64) {
    (0.0, f64::INFINITY)
  }

  #[inline]
  fn base_moments(&self) -> [f64; 4] {
    let root = self.k.sqrt();
    [self.k * self.s, root * self.s, 2.0 / root, 6.0 / self.k]
  }
}
