//! Regularised incomplete gamma and beta functions (Numerical Recipes 3e §6.2, §6.4: series below `a + 1`, Lentz's
//! continued fraction above); from shape 100 on, Temme's uniform expansion near `x = a` and Stirling's scaling elsewhere.
//! Reference: Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P. (2007), *Numerical Recipes* 3rd ed., Cambridge UP, §6.2.1 (`invgammp`'s starts: `(p/t)^{1/a}`, `t = 1 − a(0.253 + 0.12a)`, at `a ≤ 1`, Wilson–Hilferty above).
//! Reference: DiDonato, A.R., Morris, A.H. (1986), "Computation of the incomplete gamma function ratios and their inverse", ACM TOMS 12(4), 377-393, DOI 10.1145/22721.23109 (the small-shape form `Q = (x^a J − (x^a − 1))g − (g − 1)`).
//! Reference: Abramowitz, M., Stegun, I.A. (1964), "Handbook of Mathematical Functions", NBS AMS 55, 6.1.34 (the power series of `1/Γ`).

mod large_shape;
mod student;

pub(crate) use student::student_tail;
pub(crate) use student::student_xi;

use super::gamma;
use super::ln_gamma;
use super::ndtri;

/// From this shape on the series and the fraction need `O(√a)` terms near `x = a`, the switch Numerical Recipes takes.
const LARGE_SHAPE: f64 = 100.0;

/// From this shape on `ln Γ(a)`'s last place and `a ln x − x`'s rounding outgrow `ε` in the prefactor: it is Stirling-scaled.
const STIRLING_PREFACTOR: f64 = 10.0;

/// Below this shape `(pΓ(a + 1))^{1/a}` can fall under `e^{−700}`, which needs `700a < −ln p − ln Γ(a + 1) ≤ 745`.
const FLOOR_BINDS: f64 = 1.07;

/// One tail of the gamma law as `factor · e^{ln_scale}`, the upper `Q` if `upper`, else `P`; the other is one minus it.
/// `kernel` is the series' `S` or the fraction's `h` that `factor` scales, NaN in Temme's band.
#[derive(Clone, Copy)]
struct Tail {
  upper: bool,
  factor: f64,
  ln_scale: f64,
  kernel: f64,
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

/// `x^a e^{−x}/Γ(a)` as `(factor, ln_scale)`: from `ln Γ(a)` below shape 10, `large_shape::scaled_prefactor` from there.
#[inline(always)]
fn prefactor(a: f64, x: f64) -> (f64, f64) {
  if a < STIRLING_PREFACTOR {
    (1.0, -x + a * x.ln() - ln_gamma(a))
  } else {
    large_shape::scaled_prefactor(a, x)
  }
}

/// The tail the series (below `a + 1`) or the fraction (above) evaluates, scaled by `prefactor`; the large-shape path
/// stays out of line so this one inlines into each public function.
#[inline(always)]
fn tail(a: f64, x: f64) -> Tail {
  if a >= LARGE_SHAPE {
    return large_shape::tail(a, x);
  }
  let (scale, ln_scale) = prefactor(a, x);
  if x < a + 1.0 {
    let sum = series(a, x);
    Tail {
      upper: false,
      factor: scale * sum,
      ln_scale,
      kernel: sum,
    }
  } else {
    let fraction = continued_fraction(a, x);
    Tail {
      upper: true,
      factor: scale * fraction,
      ln_scale,
      kernel: fraction,
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
  if a < 1.0 && x < a + 1.0 {
    return small_shape_q(a, x);
  }
  let t = tail(a, x);
  if t.upper { t.value() } else { 1.0 - t.value() }
}

/// `Q(a, x)` at `a < 1` below `x = a + 1`, where `1 − P` cancels: `Q = (x^a J − (x^a − 1))g − (g − 1)`, `P = x^a g(1 − J)`,
/// `g = 1/Γ(1 + a)`, from quantities that vanish with `a`: within `6e-15` on 22 000 cells there, where `1 − P` reaches `7e-12`.
fn small_shape_q(a: f64, x: f64) -> f64 {
  // `J = −Σ_{n≥1} a(−x)^n/(n!(a + n))`.
  let (mut power, mut j, mut n) = (1.0, 0.0, 1.0);
  while n < 64.0 {
    power *= -x / n;
    let term = -a * power / (a + n);
    j += term;
    if term.abs() <= 1e-17 * j.abs() {
      break;
    }
    n += 1.0;
  }
  let l = (a * x.ln()).exp_m1();
  let h = recip_gamma_1p_m1(a);
  ((l + 1.0) * j - l) * (1.0 + h) - h
}

/// `1/Γ(1 + a) − 1` for `0 < a < 1` to its relative digits, by the power series of `1/Γ` (A&S 6.1.34).
fn recip_gamma_1p_m1(a: f64) -> f64 {
  const C: [f64; 28] = [
    0.5772156649015329,
    -0.6558780715202539,
    -0.04200263503409524,
    0.16653861138229148,
    -0.04219773455554433,
    -0.009621971527876973,
    0.0072189432466631,
    -0.0011651675918590652,
    -0.00021524167411495098,
    0.0001280502823881162,
    -2.013485478078824e-5,
    -1.2504934821426706e-6,
    1.133027231981696e-6,
    -2.056338416977607e-7,
    6.116095104481416e-9,
    5.002007644469223e-9,
    -1.18127457048702e-9,
    1.0434267116911005e-10,
    7.782263439905071e-12,
    -3.696805618642206e-12,
    5.100370287454476e-13,
    -2.0583260535665066e-14,
    -5.348122539423018e-15,
    1.2267786282382608e-15,
    -1.1812593016974588e-16,
    1.1866922547516004e-18,
    1.4123806553180319e-18,
    -2.29874568443537e-19,
  ];
  C.iter().rev().fold(0.0, |acc, &c| acc * a + c) * a
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
  if a < 1.0 && x < a + 1.0 {
    return small_shape_q(a, x).ln();
  }
  let t = tail(a, x);
  if t.upper {
    t.ln()
  } else {
    (-t.ln().exp()).ln_1p()
  }
}

/// `scale` (a power of two) times the `x` with `P(a, x) = p`, `0 < p < 1`, rounded once where subnormal: Newton in `ln x` on
/// `ln P` below ½ and on `ln Q = ln(1 − p)` above, both concave, from NR3's start, Wilson–Hilferty's in `x` so a huge `a` is itself.
pub(crate) fn gamma_p_inv(a: f64, p: f64, scale: f64) -> f64 {
  let upper = p > 0.5;
  let ln_target = if upper { (1.0 - p).ln() } else { p.ln() };
  // Below ½, `P ≤ x^a/Γ(a + 1)` makes `(pΓ(a + 1))^{1/a}` a lower bound of the root, and the root to rounding below `e^{−700}`.
  let floor = (!upper && a < FLOOR_BINDS).then(|| (p.ln() + ln_gamma(a + 1.0)) / a);
  if let Some(floor) = floor.filter(|&floor| floor < -700.0) {
    return (floor + scale.ln()).exp();
  }
  let floor = floor.unwrap_or(f64::NEG_INFINITY);
  let mut x = if a <= 1.0 {
    let t = 1.0 - a * (0.253 + 0.12 * a);
    let start = if p < t {
      (p.ln() - t.ln()) / a
    } else {
      (1.0 - ((1.0 - p) / (1.0 - t)).ln()).ln()
    };
    start.max(floor).exp()
  } else {
    let base = 1.0 - 1.0 / (9.0 * a) + ndtri(p) / (3.0 * a.sqrt());
    if base > 0.0 {
      a * base * base * base
    } else {
      ((p.ln() + ln_gamma(a + 1.0)) / a).exp()
    }
  };
  for _ in 0..64 {
    let (value, slope) = solved_tail(a, x, upper);
    let step = ((ln_target - value) / slope).clamp(-2.0, 2.0);
    let mut next = x + x * step.exp_m1();
    if next.ln() < floor {
      next = floor.exp();
    }
    if step.abs() <= 1e-9 {
      return scale * next;
    }
    x = next;
  }
  scale * x
}

/// `ln P(a, x)`, or `ln Q` if `upper`, and its derivative in `ln x`: `1/S` or `−1/h` where the tail solved is the series'
/// or the fraction's own, else `prefactor` over the tail.
fn solved_tail(a: f64, x: f64, upper: bool) -> (f64, f64) {
  let (value, kernel) = if upper && a < 1.0 && x < a + 1.0 {
    (small_shape_q(a, x).ln(), f64::NAN)
  } else {
    let t = tail(a, x);
    if t.upper == upper {
      (t.ln(), t.kernel)
    } else {
      ((-t.ln().exp()).ln_1p(), f64::NAN)
    }
  };
  let slope = if kernel.is_nan() {
    let (scale, ln_scale) = prefactor(a, x);
    (scale.ln() + ln_scale - value).exp()
  } else {
    kernel.recip()
  };
  (value, if upper { -slope } else { slope })
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
