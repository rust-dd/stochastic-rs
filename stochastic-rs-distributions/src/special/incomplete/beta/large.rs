//! `I_x(a, b)` from a shape of 100: the prefactor `x^a y^b/B(a, b)` with a two-part exponent, TOMS
//! 708's fraction on the switch's side, and `asym` beside the mean past both shapes 100.
//! Reference: DiDonato, A.R., Morris, A.H. (1992), "Algorithm 708: Significant digit computation of the incomplete beta function ratios", ACM TOMS 18(3), 360-373, DOI 10.1145/131766.131776 (`brcomp`: `√(ab/(2π(a + b))) e^{−a·g(−λ/a) − b·g(λ/b) − Δ}`, `g(t) = t − ln(1 + t)`, `λ = a − (a + b)x`; `bfrac`'s recurrence; `bratio`'s band `λ ≤ 0.03 min(a, b)` for `basym` past 100).
//! Reference: NIST Digital Library of Mathematical Functions, eq. 8.17.22 (the continued fraction), <https://dlmf.nist.gov/8.17.22>.

use std::f64::consts::TAU;

use super::asym;
use super::complement;
use crate::special::KEPT;
use crate::special::STIRLING_SHAPES;
use crate::special::add_parts;
use crate::special::exp_parts;
use crate::special::gamma_q;
use crate::special::kl_divergence;
use crate::special::ln_beta_parts;
use crate::special::ln_gamma_diff;
use crate::special::ln_gamma_q_scaled;
use crate::special::ln_gamma_star;
use crate::special::ln_parts;
use crate::special::ln_shape_beta_parts;
use crate::special::mul_parts;
use crate::special::settle;
use crate::special::sub_parts;
use crate::special::two_sum;

/// Both shapes past it, `asym` serves the band `|a − (a + b)x| ≤ ASYM_BAND·min(a, b)`, where the
/// fraction's step count grows with the cube root of the shapes.
const ASYM_SHAPE: f64 = 100.0;

const ASYM_BAND: f64 = 0.03;

/// The fraction's step cap; outside `asym`'s band it converges in under 100 steps at every shape
/// measured.
const STEPS: f64 = 1000.0;

/// `x^a y^b/(s·B(a, b))`, `s` the fraction's first shape, as `factor·e^{ln_scale}`: the exponent's
/// high part, with its low part and the scale in `factor`; `times()` rounds once below the normals.
#[derive(Clone, Copy)]
struct Prefactor {
  factor: f64,
  ln_scale: f64,
}

impl Prefactor {
  fn ln(self) -> f64 {
    self.factor.ln() + self.ln_scale
  }

  fn times(self, k: f64) -> f64 {
    exp_parts((self.ln_scale, 0.0), self.factor * k, 0)
  }
}

/// The Stirling-scaled parts at both shapes from 10: the exponent `a·g(D/a) + b·g(−D/b)`, `D = (a +
/// b)x − a`, in two parts, `√(ab/(2π(a + b)))` and `Δ = ln Γ*(a) + ln Γ*(b) − ln Γ*(a + b)`.
struct Stirling {
  exponent: (f64, f64),
  scale: f64,
  star: f64,
}

impl Stirling {
  fn new(a: f64, b: f64, x: f64, y: f64) -> Self {
    let half = 0.5 * a + 0.5 * b;
    Self {
      exponent: kl_exponent(a, b, x, y),
      scale: ((0.5 * a / half) * b / TAU).sqrt(),
      star: ln_gamma_star(a) + ln_gamma_star(b) - ln_gamma_star(a + b),
    }
  }

  fn prefactor(&self, shape: f64) -> Prefactor {
    Prefactor {
      factor: self.scale / shape * (-self.exponent.1 - self.star).exp(),
      ln_scale: -self.exponent.0,
    }
  }
}

/// `D = (a + b)x − a = b − (a + b)y` in two parts from the exact side, with `(a + b)x` and `(a +
/// b)y` beside it, each in two parts.
fn offset(a: f64, b: f64, x: f64, y: f64) -> ((f64, f64), (f64, f64), (f64, f64)) {
  let (s, s_lo) = two_sum(a, b);
  let (xx, yy) = if x <= 0.5 {
    ((x, 0.0), (y, (1.0 - y) - x))
  } else {
    ((x, (1.0 - x) - y), (y, 0.0))
  };
  let sx = mul_parts((s, s_lo), xx);
  let sy = mul_parts((s, s_lo), yy);
  let d = if x <= 0.5 {
    sub_parts(sx, (a, 0.0))
  } else {
    sub_parts((b, 0.0), sy)
  };
  (d, sx, sy)
}

/// `ln x` and `ln y`, each in two parts.
pub(super) type Logs = ((f64, f64), (f64, f64));

/// The inverse's per-law cache of the two-part `ln(a·B)` and `ln(b·B)`, the switch's side's, which
/// every Newton step would otherwise form again (three two-part `ln Γ`).
#[derive(Default)]
pub(super) struct Norms {
  lower: Option<(f64, f64)>,
  upper: Option<(f64, f64)>,
}

/// From this shape (`2¹⁰²²`) the two-sums of `offset` would overflow: `λ` and the exponent,
/// homogeneous in the shapes, are read from the halves and doubled.
const HALVE_FROM: f64 = f64::MAX / 4.0;

/// `λ = a − (a + b)x`, the distance below the mean in the shapes' scale, from the exact side.
fn lambda(a: f64, b: f64, x: f64, y: f64) -> f64 {
  if a.max(b) > HALVE_FROM {
    return 2.0 * lambda(0.5 * a, 0.5 * b, x, y);
  }
  let (d, _, _) = offset(a, b, x, y);
  -(d.0 + d.1)
}

/// `a·g(D/a) + b·g(−D/b)`, `g(u) = u − ln(1 + u)`, in two parts; from the halves, doubled, where a
/// shape passes `HALVE_FROM`, the exponent being homogeneous in the shapes; `+∞` at an end.
fn kl_exponent(a: f64, b: f64, x: f64, y: f64) -> (f64, f64) {
  if x == 0.0 || y == 0.0 {
    return (f64::INFINITY, 0.0);
  }
  if a.max(b) > HALVE_FROM {
    let (e, e_lo) = kl_exponent(0.5 * a, 0.5 * b, x, y);
    return (2.0 * e, 2.0 * e_lo);
  }
  let (d, sx, sy) = offset(a, b, x, y);
  let first = kl_divergence(a, d, sx);
  let second = kl_divergence(b, (-d.0, -d.1), sy);
  if !first.0.is_finite() || !second.0.is_finite() || first.0 + second.0 == f64::INFINITY {
    return (f64::INFINITY, 0.0);
  }
  settle(add_parts(first, second))
}

/// The prefactor below both shapes 10 from `a ln x + b ln y − ln_norm`, `ln_norm` the two-part
/// `ln(s·B(a, b))` or `ln B(a, b)`, every term in two parts.
fn ln_beta_prefactor(
  a: f64,
  b: f64,
  ln_norm: (f64, f64),
  ln_x: (f64, f64),
  ln_y: (f64, f64),
) -> Prefactor {
  if (a * ln_x.0).is_infinite() || (b * ln_y.0).is_infinite() {
    return Prefactor {
      factor: 1.0,
      ln_scale: f64::NEG_INFINITY,
    };
  }
  let sum = add_parts(mul_parts((a, 0.0), ln_x), mul_parts((b, 0.0), ln_y));
  let (h, h_lo) = settle(sub_parts(sum, ln_norm));
  Prefactor {
    factor: h_lo.exp(),
    ln_scale: h,
  }
}

/// `ln x` and `ln y` in two parts from `(x, y)`, the smaller exact and the other carrying its
/// rounding.
pub(super) fn logs(x: f64, y: f64) -> Logs {
  if x <= 0.5 {
    (ln_parts(x, 0.0, 0), ln_parts(y, (1.0 - y) - x, 0))
  } else {
    (ln_parts(x, (1.0 - x) - y, 0), ln_parts(y, 0.0, 0))
  }
}

/// `I_x(a, b)/(x^a y^b/(aB(a, b)))` below the switch (`λ > −1`): TOMS 708's form, reading `λ + 1`
/// whole where `1 − (a + b)x/(a + 1)` cancels, by modified Lentz on `K/a`, free of `1/a`.
pub(super) fn fraction(a: f64, b: f64, x: f64, y: f64, lambda: f64) -> f64 {
  const FPMIN: f64 = 1e-300;
  let sum_finite = (a + b).is_finite();
  let (c, yp1) = (lambda + 1.0, y + 1.0);
  let mut f = c / (a + 1.0);
  if f == 0.0 {
    f = FPMIN;
  }
  let (mut cc, mut d) = (f, 0.0);
  let (mut n, mut scale_prev) = (0.0, 1.0);
  while n < STEPS {
    n += 1.0;
    let s = a + 2.0 * n - 1.0;
    let u = a + n - 1.0;
    let mut w = n * x * (b - n);
    // Where `w` overflows, `b − n` is scaled first by `k`: `β_n·k` and `α_n·k·k_{n−1}`, an
    // equivalence transformation.
    let scale = if w.is_finite() { 1.0 } else { 2f64.powi(-20) };
    if scale < 1.0 {
      w = n * x * ((b - n) * scale);
    }
    let ratio = if sum_finite {
      (u + b) / s
    } else {
      u / s + b / s
    };
    let first = if n == 1.0 { 1.0 / s } else { u / s };
    let alpha = first * ratio * (w * x) * scale_prev;
    let e = (a + n) / (a + 1.0 + 2.0 * n);
    let inner = n + e * (c + n * yp1);
    let beta = w / s + inner * scale;
    scale_prev = scale;
    d = beta + alpha * d;
    if d == 0.0 {
      d = FPMIN;
    }
    cc = beta + alpha / cc;
    if cc == 0.0 {
      cc = FPMIN;
    }
    d = 1.0 / d;
    let delta = cc * d;
    f *= delta;
    if (delta - 1.0).abs() < 1e-15 {
      break;
    }
  }
  1.0 / f
}

/// The side formed directly, `I` (`lower`) or `1 − I`, with its logarithm, finite where it
/// underflows, and the prefactor's logarithm.
pub(super) struct Side {
  pub(super) lower: bool,
  pub(super) value: f64,
  pub(super) ln: f64,
  /// `ln(value/bt)` from the route itself, where the two logarithms of the exponent's size would
  /// cancel.
  pub(super) ln_ratio: f64,
  pub(super) ln_prefactor: f64,
}

/// `I_x(a, b)`'s side: `asym` in its band, else the fraction on the switch's side, the gamma limit
/// below `KEPT`, the series past ½ at a shape ≤ 1; a copy a caller spares `beta_i` logs and cache.
#[inline(always)]
pub(super) fn side(
  a: f64,
  b: f64,
  x: f64,
  y: f64,
  given: Option<Logs>,
  norms: Option<&mut Norms>,
) -> Side {
  let small = a.min(b);
  // The two-part logs are read below both shapes 10 only, so a caller without them pays for none.
  let mut given = given;
  let lambda = lambda(a, b, x, y);
  // Numerical Recipes' switch `x < (a + 1)/(a + b + 2)` as `λ > 2x − 1` from the exact side: at a
  // law narrower than the doubles about its mean the rounded `x` cannot tell the sides apart.
  let lower = lambda
    > if x <= 0.5 {
      2.0 * x - 1.0
    } else {
      1.0 - 2.0 * y
    };
  let shape = if lower { a } else { b };
  // An underflowed side keeps its place only in its log, where the Stirling exponent reads `+∞`;
  // the two-part `ln(s·B)` serves to a shape of 32.
  let from_logs = small < STIRLING_SHAPES || ((x == 0.0 || y == 0.0) && small <= 32.0);
  let pf = if !from_logs {
    let st = Stirling::new(a, b, x, y);
    if small > ASYM_SHAPE && lambda.abs() <= ASYM_BAND * small {
      let below = lambda >= 0.0;
      let (value, ln, ln_ratio) = if below {
        asym::expansion(a, b, st.exponent, st.star)
      } else {
        asym::expansion(b, a, st.exponent, st.star)
      };
      return Side {
        lower: below,
        value,
        ln,
        ln_ratio,
        ln_prefactor: st.prefactor(1.0).ln(),
      };
    }
    st.prefactor(shape)
  } else {
    let other = if lower { b } else { a };
    let (ln_x, ln_y) = *given.get_or_insert_with(|| logs(x, y));
    let norm = match norms {
      Some(n) => *(if lower { &mut n.lower } else { &mut n.upper })
        .get_or_insert_with(|| ln_shape_beta_parts(shape, other)),
      None => ln_shape_beta_parts(shape, other),
    };
    ln_beta_prefactor(a, b, norm, ln_x, ln_y)
  };
  let ln_shape = shape.ln();
  let ln_prefactor = pf.ln() + ln_shape;
  // Past the mean below `KEPT` the fraction's `1 − x` keeps under half of `x`'s digits: the gamma
  // limit from `x` itself, its logarithm finite where `Q` underflows.
  let (g_shape, g_other, held) = if lower { (b, a, y) } else { (a, b, x) };
  if held < KEPT
    && let Some((n, s)) = complement::gamma_limit_arg(g_shape, g_other, held)
  {
    let z = n * s;
    let (ln, ln_kernel) = ln_gamma_q_scaled(g_shape, z);
    // `Q/bt` from the fraction's own logarithm, where `ln Q` and `ln bt`, both about `−z`, would
    // cancel.
    let ln_ratio = match ln_kernel {
      Some(k) => {
        k + g_shape * (n.ln() + ((s - held) / held).ln_1p())
          - 0.5 * (g_shape - 1.0) * s
          - ln_gamma_diff(g_other, g_shape).0
      }
      None => ln - ln_prefactor,
    };
    return Side {
      lower,
      value: gamma_q(g_shape, z),
      ln,
      ln_ratio,
      ln_prefactor,
    };
  }
  if lower {
    let h = fraction(a, b, x, y, lambda);
    let v = pf.times(h);
    if v > 0.5 && a <= 1.0 && x <= 0.5 {
      let (ln_x, _) = *given.get_or_insert_with(|| logs(x, y));
      let (value, ln) = complement::upper_series_ln(a, b, x, ln_x.0);
      return Side {
        lower: false,
        value,
        ln,
        ln_ratio: ln - ln_prefactor,
        ln_prefactor,
      };
    }
    let ln_h = h.ln();
    return Side {
      lower: true,
      value: v,
      ln: pf.ln() + ln_h,
      ln_ratio: ln_h - ln_shape,
      ln_prefactor,
    };
  }
  let h = fraction(b, a, y, x, -lambda);
  let rest = pf.times(h);
  if rest > 0.5 && b <= 1.0 && y <= 0.5 {
    let (_, ln_y) = *given.get_or_insert_with(|| logs(x, y));
    let (value, ln) = complement::upper_series_ln(b, a, y, ln_y.0);
    return Side {
      lower: true,
      value,
      ln,
      ln_ratio: ln - ln_prefactor,
      ln_prefactor,
    };
  }
  let ln_h = h.ln();
  Side {
    lower: false,
    value: rest,
    ln: pf.ln() + ln_h,
    ln_ratio: ln_h - ln_shape,
    ln_prefactor,
  }
}

/// `I_x(a, b)` with `y = 1 − x`, the smaller exact, at a shape from 100.
pub(super) fn beta_i(a: f64, b: f64, x: f64, y: f64) -> f64 {
  let s = side(a, b, x, y, None, None);
  if s.lower { s.value } else { 1.0 - s.value }
}

/// The density `x^{a−1} y^{b−1}/B(a, b)` at a shape from 100: the prefactor over `xy`.
pub(super) fn density(a: f64, b: f64, x: f64, y: f64) -> f64 {
  let pf = if a.min(b) >= STIRLING_SHAPES {
    Stirling::new(a, b, x, y).prefactor(1.0)
  } else {
    let (ln_x, ln_y) = logs(x, y);
    ln_beta_prefactor(a, b, ln_beta_parts(a, b), ln_x, ln_y)
  };
  pf.times(1.0 / y) / x
}

/// A point of `(0, 1)` as the smaller of `x` and `y = 1 − x` with its logarithm `w`, which keeps
/// the place of a point below the subnormals; `x` and `y` are the doubles the law is read at.
pub(super) struct Point {
  pub(super) x: f64,
  pub(super) y: f64,
  pub(super) x_small: bool,
  pub(super) w: f64,
  ln_small: (f64, f64),
  ln_other: (f64, f64),
}

impl Point {
  /// The point whose `x` (if `x_small`) or `y` is `e^w`; past ½ the roles swap, so the smaller side
  /// is held.
  pub(super) fn from_ln(w: f64, x_small: bool) -> Self {
    let s = w.exp();
    if s > 0.5 {
      let o = 1.0 - s;
      let ln_small = ln_parts(o, 0.0, 0);
      return Self::arrange(o, s, !x_small, ln_small.0, ln_small, ln_parts(s, 0.0, 0));
    }
    let o = 1.0 - s;
    // A subnormal double keeps few bits: its own log, not `w`, is where the tails are read, so the
    // finish lands on the double the Newton saw; at 0 only `w` places the point.
    let (ln_small, ln_other) = if s == 0.0 {
      ((w, 0.0), (0.0, 0.0))
    } else if s < f64::MIN_POSITIVE {
      (ln_parts(s * 2f64.powi(60), 0.0, -60), ((-s).ln_1p(), 0.0))
    } else {
      (ln_parts(s, 0.0, 0), ln_parts(o, (1.0 - o) - s, 0))
    };
    Self::arrange(s, o, x_small, w, ln_small, ln_other)
  }

  fn arrange(
    small: f64,
    other: f64,
    x_small: bool,
    w: f64,
    ln_small: (f64, f64),
    ln_other: (f64, f64),
  ) -> Self {
    let (x, y) = if x_small {
      (small, other)
    } else {
      (other, small)
    };
    Self {
      x,
      y,
      x_small,
      w,
      ln_small,
      ln_other,
    }
  }

  /// The logarithm of the side the point does not hold, `ln y` if `x_small`.
  pub(super) fn ln_other(&self) -> f64 {
    self.ln_other.0
  }

  /// The held side's logarithm in two parts, exact for the double the tails were read at.
  pub(super) fn ln_held(&self) -> (f64, f64) {
    self.ln_small
  }
}

/// `ln I_x(a, b)`, `ln(1 − I_x(a, b))`, the smaller exact and the other through `ln_1p`, and the
/// prefactor's.
pub(super) struct Tails {
  pub(super) ln_lower: f64,
  pub(super) ln_upper: f64,
  pub(super) ln_prefactor: f64,
  /// Which tail the route computed itself, and its `ln(value/bt)`.
  pub(super) lower: bool,
  pub(super) ln_ratio: f64,
}

/// Both tails' logarithms at `point`, at every shape, finite where the tails underflow; `norms`
/// caches the law's `ln(s·B)` across a Newton's steps.
pub(super) fn ln_tails(a: f64, b: f64, point: &Point, norms: Option<&mut Norms>) -> Tails {
  let (ln_x, ln_y) = if point.x_small {
    (point.ln_small, point.ln_other)
  } else {
    (point.ln_other, point.ln_small)
  };
  let s = side(a, b, point.x, point.y, Some((ln_x, ln_y)), norms);
  let other = (-s.value).ln_1p();
  let (ln_lower, ln_upper) = if s.lower {
    (s.ln, other)
  } else {
    (other, s.ln)
  };
  Tails {
    ln_lower,
    ln_upper,
    ln_prefactor: s.ln_prefactor,
    lower: s.lower,
    ln_ratio: s.ln_ratio,
  }
}
