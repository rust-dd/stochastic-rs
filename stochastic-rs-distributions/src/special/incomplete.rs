//! Regularised incomplete gamma and beta functions (Numerical Recipes 3e §6.2, §6.4: series below `a + 1`, Lentz's
//! continued fraction above); from shape 100 on, Temme's uniform expansion near `x = a` and Stirling's scaling elsewhere.

mod large_shape;

use super::gamma;
use super::ln_gamma;

/// From this shape on the series and the fraction need `O(√a)` terms near `x = a`, the switch Numerical Recipes takes.
const LARGE_SHAPE: f64 = 100.0;

/// One tail of the gamma law as `factor · e^{ln_scale}`, the upper `Q` if `upper`, else `P`; the other is one minus it.
#[derive(Clone, Copy)]
struct Tail {
  upper: bool,
  factor: f64,
  ln_scale: f64,
}

impl Tail {
  fn value(self) -> f64 {
    self.factor * self.ln_scale.exp()
  }

  fn ln(self) -> f64 {
    self.factor.ln() + self.ln_scale
  }
}

fn domain(a: f64, x: f64) -> bool {
  !(x < 0.0 || a.is_nan() || a <= 0.0)
}

/// The tail the series (below `a + 1`) or the fraction (above) evaluates, scaled by `x^a e^{−x}/Γ(a)`; the large-shape
/// path stays out of line so this one inlines into each public function.
#[inline]
fn tail(a: f64, x: f64) -> Tail {
  if a >= LARGE_SHAPE {
    return large_shape::tail(a, x);
  }
  let ln_scale = -x + a * x.ln() - ln_gamma(a);
  if x < a + 1.0 {
    Tail {
      upper: false,
      factor: series(a, x),
      ln_scale,
    }
  } else {
    Tail {
      upper: true,
      factor: continued_fraction(a, x),
      ln_scale,
    }
  }
}

/// `Σ_{n≥0} xⁿ/(a(a + 1)⋯(a + n))`, so that `P = sum · x^a e^{−x}/Γ(a)`.
fn series(a: f64, x: f64) -> f64 {
  let mut ap = a;
  let mut sum = 1.0 / a;
  let mut del = sum;
  for _ in 0..200 {
    ap += 1.0;
    del *= x / ap;
    sum += del;
    if del.abs() < sum.abs() * 1e-15 {
      break;
    }
  }
  sum
}

/// Lentz's continued fraction `h`, so that `Q = h · x^a e^{−x}/Γ(a)`.
fn continued_fraction(a: f64, x: f64) -> f64 {
  let fpmin = 1e-300_f64;
  let mut b = x + 1.0 - a;
  let mut c = 1.0 / fpmin;
  let mut d = 1.0 / b;
  let mut h = d;
  for i in 1..=200 {
    let an = -(i as f64) * (i as f64 - a);
    b += 2.0;
    d = an * d + b;
    if d.abs() < fpmin {
      d = fpmin;
    }
    c = b + an / c;
    if c.abs() < fpmin {
      c = fpmin;
    }
    d = 1.0 / d;
    let del = d * c;
    h *= del;
    if (del - 1.0).abs() < 1e-15 {
      break;
    }
  }
  h
}

/// Regularised lower incomplete gamma `P(a, x) = γ(a, x)/Γ(a)`; NaN for a negative `x` or a non-positive `a`.
pub fn gamma_p(a: f64, x: f64) -> f64 {
  if !domain(a, x) {
    return f64::NAN;
  }
  if x == 0.0 {
    return 0.0;
  }
  if x == f64::INFINITY {
    return 1.0;
  }
  let t = tail(a, x);
  if t.upper { 1.0 - t.value() } else { t.value() }
}

/// Regularised upper incomplete gamma `Q(a, x) = Γ(a, x)/Γ(a)`, computed directly so the upper tail keeps its digits.
pub fn gamma_q(a: f64, x: f64) -> f64 {
  if !domain(a, x) {
    return f64::NAN;
  }
  if x == 0.0 {
    return 1.0;
  }
  if x == f64::INFINITY {
    return 0.0;
  }
  let t = tail(a, x);
  if t.upper { t.value() } else { 1.0 - t.value() }
}

/// `ln P(a, x)`, finite where `P` underflows.
pub fn ln_gamma_p(a: f64, x: f64) -> f64 {
  if !domain(a, x) {
    return f64::NAN;
  }
  if x == 0.0 {
    return f64::NEG_INFINITY;
  }
  if x == f64::INFINITY {
    return 0.0;
  }
  let t = tail(a, x);
  if t.upper {
    (-t.ln().exp()).ln_1p()
  } else {
    t.ln()
  }
}

/// `ln Q(a, x)`, finite where `Q` underflows.
pub fn ln_gamma_q(a: f64, x: f64) -> f64 {
  if !domain(a, x) {
    return f64::NAN;
  }
  if x == 0.0 {
    return 0.0;
  }
  if x == f64::INFINITY {
    return f64::NEG_INFINITY;
  }
  let t = tail(a, x);
  if t.upper {
    t.ln()
  } else {
    (-t.ln().exp()).ln_1p()
  }
}

/// Unregularised lower incomplete gamma `γ(a, x) = P(a, x)·Γ(a)`; in log space once `Γ(a)` overflows.
pub fn gamma_li(a: f64, x: f64) -> f64 {
  let g = gamma(a);
  if g.is_finite() {
    gamma_p(a, x) * g
  } else {
    (ln_gamma(a) + ln_gamma_p(a, x)).exp()
  }
}

/// Unregularised upper incomplete gamma `Γ(a, x) = Q(a, x)·Γ(a)`; in log space once `Γ(a)` overflows.
pub fn gamma_ui(a: f64, x: f64) -> f64 {
  let g = gamma(a);
  if g.is_finite() {
    gamma_q(a, x) * g
  } else {
    (ln_gamma(a) + ln_gamma_q(a, x)).exp()
  }
}

/// Regularised incomplete beta `I_x(a, b) = B(x; a, b)/B(a, b)` by Lentz's continued fraction, through the symmetry
/// `I_x(a, b) = 1 − I_{1−x}(b, a)` on the far side of the mean; NaN for an `x` outside `[0, 1]`.
pub fn beta_i(a: f64, b: f64, x: f64) -> f64 {
  if !(0.0..=1.0).contains(&x) {
    return f64::NAN;
  }
  if x == 0.0 || x == 1.0 {
    return x;
  }
  let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
  if x < (a + 1.0) / (a + b + 2.0) {
    bt * betacf(a, b, x) / a
  } else {
    1.0 - bt * betacf(b, a, 1.0 - x) / b
  }
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
  let fpmin = 1e-300_f64;
  let qab = a + b;
  let qap = a + 1.0;
  let qam = a - 1.0;
  let mut c = 1.0;
  let mut d = 1.0 - qab * x / qap;
  if d.abs() < fpmin {
    d = fpmin;
  }
  d = 1.0 / d;
  let mut h = d;
  for m in 1..=200 {
    let m_f = m as f64;
    let m2 = 2.0 * m_f;
    let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if d.abs() < fpmin {
      d = fpmin;
    }
    c = 1.0 + aa / c;
    if c.abs() < fpmin {
      c = fpmin;
    }
    d = 1.0 / d;
    h *= d * c;
    let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if d.abs() < fpmin {
      d = fpmin;
    }
    c = 1.0 + aa / c;
    if c.abs() < fpmin {
      c = fpmin;
    }
    d = 1.0 / d;
    let del = d * c;
    h *= del;
    if (del - 1.0).abs() < 1e-15 {
      break;
    }
  }
  h
}

#[cfg(test)]
mod tests;
