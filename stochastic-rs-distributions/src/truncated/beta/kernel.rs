//! The truncated beta's kernel on `[0, 1]`.

use crate::truncated::quadrature::Cut;
use crate::truncated::quadrature::Kernel;
use crate::truncated::quadrature::ln_1p_mx;

/// `x^{a−1} (1 − x)^{b−1}` on `[0, 1]` about the reference point `c`; a unit shape has no log term, whose `0·ln 0`
/// would read NaN at its end.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct BetaKernel {
  a: f64,
  b: f64,
  c: f64,
  /// `1 − c`, exact where `c` is held beside 1.
  cc: f64,
  ln_c: f64,
  ln_cc: f64,
  /// `(a − 1)/c − (b − 1)/(1 − c)`, the slope at `c`, `None` where it overflows.
  slope: Option<f64>,
  mode: Cut,
  /// The mode is stationary and held exactly, so its slope is 0 by construction.
  exact: bool,
}

impl BetaKernel {
  pub(super) fn new(a: f64, b: f64) -> Self {
    let (mode, exact) = held_mode(a, b);
    Self {
      a,
      b,
      c: 0.5,
      cc: 0.5,
      ln_c: -std::f64::consts::LN_2,
      ln_cc: -std::f64::consts::LN_2,
      slope: None,
      mode,
      exact,
    }
  }

  /// `(a − 1)(ln(x/c) − y/c)`: `ln_1p_mx` beside `c`, read from `x` toward 0, and `(a − 1)y/c` one quotient where
  /// `y/c` overflows past a subnormal `c`.
  fn left(&self, x: f64, y: f64) -> f64 {
    let z = y / self.c;
    if self.a == 1.0 {
      0.0
    } else if z < -0.5 {
      (self.a - 1.0) * ((x.ln() - self.ln_c) - z)
    } else if z.is_infinite() {
      (self.a - 1.0) * (x.ln() - self.ln_c) - (self.a - 1.0) * y / self.c
    } else {
      (self.a - 1.0) * ln_1p_mx(z)
    }
  }

  /// `(b − 1)(ln((1 − x)/(1 − c)) + y/(1 − c))`, `1 − x` toward 1 as the offset from 1 where `branch` holds it, else as
  /// `(1 − c) − y`: never from a rounded `x`.
  fn right(&self, y: f64, branch: Option<(f64, f64)>) -> f64 {
    let z = -y / self.cc;
    if self.b == 1.0 {
      0.0
    } else if z < -0.5 {
      let gap = match branch {
        Some((1.0, t)) => -t,
        _ => self.cc - y,
      };
      (self.b - 1.0) * ((gap.ln() - self.ln_cc) - z)
    } else if z.is_infinite() {
      (self.b - 1.0) * ((self.cc - y).ln() - self.ln_cc) + (self.b - 1.0) * y / self.cc
    } else {
      (self.b - 1.0) * ln_1p_mx(z)
    }
  }

  /// `(a − 1)y/c − (b − 1)y/(1 − c)`, from the slope where it is finite; where it overflows the mode is far and the two
  /// terms do not cancel.
  fn linear(&self, y: f64) -> f64 {
    match self.slope {
      Some(slope) => slope * y,
      None => self.per(self.a, y, self.c) - self.per(self.b, y, self.cc),
    }
  }

  /// `(e − 1)·y/d`, 0 for a unit exponent, whose end `d` may be 0, one quotient where `y/d` overflows.
  fn per(&self, e: f64, y: f64, d: f64) -> f64 {
    let q = y / d;
    if e == 1.0 {
      0.0
    } else if q.is_finite() {
      (e - 1.0) * q
    } else {
      (e - 1.0) * y / d
    }
  }
}

impl Kernel for BetaKernel {
  fn ln_k(&self, x: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) {
      return f64::NEG_INFINITY;
    }
    let left = if self.a == 1.0 {
      0.0
    } else {
      (self.a - 1.0) * x.ln()
    };
    let right = if self.b == 1.0 {
      0.0
    } else {
      (self.b - 1.0) * (-x).ln_1p()
    };
    left + right
  }

  /// The slope from `(a − 1)(1 − c) − (b − 1)c` by an exact product, so a clamped end next to the mode keeps it; 0 at
  /// a stationary mode held exactly, never at an end the density merely peaks at.
  fn anchor(&mut self, c: Cut, at_mode: bool) {
    self.c = c.x;
    self.cc = -c.offset(1.0);
    self.ln_c = self.c.ln();
    self.ln_cc = self.cc.ln();
    let (a1, b1) = (self.a - 1.0, self.b - 1.0);
    let slope = if a1 == 0.0 || b1 == 0.0 {
      self.per(self.a, 1.0, self.c) - self.per(self.b, 1.0, self.cc)
    } else {
      let p = b1 * self.c;
      let e = b1.mul_add(self.c, -p);
      (a1.mul_add(self.cc, -p) - e) / self.c / self.cc
    };
    let slope = if at_mode && self.exact { 0.0 } else { slope };
    self.slope = slope.is_finite().then_some(slope);
  }

  fn ln_rel(&self, x: f64, y: f64, branch: Option<(f64, f64)>) -> f64 {
    if y == 0.0 {
      return 0.0;
    }
    self.left(x, y) + self.right(y, branch) + self.linear(y)
  }

  fn ln_rel_regular(&self, x: f64, y: f64, s: f64) -> f64 {
    let (a1, b1) = (self.a - 1.0, self.b - 1.0);
    if s == 0.0 {
      -a1 * self.ln_c + self.right(y, None) - b1 * (y / self.cc)
    } else {
      self.left(x, y) + a1 * (y / self.c) - b1 * self.ln_cc
    }
  }

  fn d1(&self, x: f64, y: f64) -> f64 {
    match self.slope {
      Some(slope) => {
        slope - self.per(self.a, y, self.c) / x - self.per(self.b, y, self.cc) / (self.cc - y)
      }
      None => self.per(self.a, 1.0, x) - self.per(self.b, 1.0, self.cc - y),
    }
  }

  fn d2(&self, x: f64, y: f64) -> f64 {
    let gap = self.cc - y;
    -self.per(self.a, 1.0, x * x) - self.per(self.b, 1.0, gap * gap)
  }

  fn mode(&self) -> Cut {
    self.mode
  }

  fn branch_points(&self) -> [Option<(f64, f64)>; 2] {
    [
      (self.a != 1.0).then_some((0.0, self.a)),
      (self.b != 1.0).then_some((1.0, self.b)),
    ]
  }

  /// `1/(α + β)` from halves, so it never overflows to 0: beside 0 the factor `(1 − x)^{β−1}` falls an e-fold in
  /// about `1/β`, beside 1 `x^{α−1}` in `1/α`.
  fn length_scale(&self) -> f64 {
    0.5 / (0.5 * self.a + 0.5 * self.b)
  }

  fn support(&self) -> (f64, f64) {
    (0.0, 1.0)
  }

  /// The closed forms in `r = (a − b)/√(ab)` and `1/(a + b)`, so no product of the shapes over- or underflows.
  fn base_moments(&self) -> [f64; 4] {
    let (a, b) = (self.a, self.b);
    let root = a.sqrt() * b.sqrt();
    let s = a + b;
    if s < 2.0 {
      let r = (a - b) / root;
      return [
        a / s,
        root / s / (s + 1.0).sqrt(),
        -2.0 * r * (s + 1.0).sqrt() / (s + 2.0),
        6.0 * (r * r * (s + 1.0) / ((s + 2.0) * (s + 3.0)) - 1.0 / (s + 3.0)),
      ];
    }
    let inv = self.length_scale();
    let f = inv * (1.0 + inv) / ((1.0 + 2.0 * inv) * (1.0 + 3.0 * inv));
    let rf = (a - b) * f.sqrt() / root;
    // From the nearer end: a subnormal `1/(a + b)` carries few bits, which `a·inv` scales up to `1`'s ulp.
    let mean = if a > b { 1.0 - b * inv } else { a * inv };
    [
      mean,
      root * inv * (inv / (1.0 + inv)).sqrt(),
      -2.0 * ((a - b) * (inv * (1.0 + inv)).sqrt() / root) / (1.0 + 2.0 * inv),
      6.0 * (rf * rf - inv / (1.0 + 3.0 * inv)),
    ]
  }
}

/// The interior mode when both shapes exceed 1, a double-double beside the end it lies nearer and exact while that
/// offset is normal; one past the least double is held at it, off the kernel's zero. Else the end the density rises
/// toward, the heavier of both (0 on a tie), and ½ for the uniform.
fn held_mode(a: f64, b: f64) -> (Cut, bool) {
  if a > 1.0 && b > 1.0 {
    // Halves, so `a + b − 2` cannot overflow, summed and divided exactly to a double-double.
    let (ha, hb) = (0.5 * (a - 1.0), 0.5 * (b - 1.0));
    let d = ha + hb;
    let d_lo = (ha - (d - (d - ha))) + (hb - (d - ha));
    let quotient = |n: f64| {
      let q = n / d;
      (q, ((-q).mul_add(d, n) - q * d_lo) / d)
    };
    let ((m, m_lo), (gap, gap_lo)) = (quotient(ha), quotient(hb));
    let least = f64::from_bits(1);
    if m <= 0.5 {
      (Cut::exact(m.max(least), m_lo), m >= f64::MIN_POSITIVE)
    } else if 1.0 - gap < 1.0 {
      let hi = 1.0 - gap;
      (Cut::exact(hi, ((1.0 - hi) - gap) - gap_lo), true)
    } else {
      (Cut::exact(1.0, -gap.max(least)), gap >= f64::MIN_POSITIVE)
    }
  } else if a == 1.0 && b == 1.0 {
    (Cut::at(0.5), false)
  } else if a < 1.0 && (1.0 <= b || a <= b) || (a == 1.0 && b > 1.0) {
    (Cut::at(0.0), false)
  } else {
    (Cut::at(1.0), false)
  }
}
