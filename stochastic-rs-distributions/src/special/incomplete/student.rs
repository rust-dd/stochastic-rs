//! The Student-t law's incomplete beta, `I_w(ν/2, ½)` at `w = ν/(ν + x²)` and its reflection `I_z(½, ν/2)`, `z = 1 − w`:
//! the series in `z` near zero, in `w` in the far tail, between them Lentz's fraction or the expansion in incomplete gammas.
//! Reference: Abramowitz, M., Stegun, I.A. (1964), "Handbook of Mathematical Functions", NBS AMS 55, 26.7.1 (`A(t|ν) = 1 − I_w(ν/2, ½)`).
//! Reference: NIST Digital Library of Mathematical Functions, eqs. 8.17.4 (the reflection), 8.17.8 (the series of positive terms) and 8.17.20 (the step `a → a + 1`), <https://dlmf.nist.gov/8.17>.
//! Reference: DiDonato, A.R., Morris, A.H. (1992), "Algorithm 708: Significant digit computation of the incomplete beta function ratios", ACM TOMS 18(3), 360-373, DOI 10.1145/131766.131776, eqs. (9.1)–(9.2) (the expansion in `Γ(b + 2n, Tξ)`, `T = a + (b − 1)/2`).
//! Reference: NIST Digital Library of Mathematical Functions, eq. 4.6.4 (`ln z = 2 Σ ((z − 1)/(z + 1))^{2k+1}/(2k + 1)`), <https://dlmf.nist.gov/4.6.4>.
//! Reference: Dekker, T.J. (1971), "A floating-point technique for extending the available precision", *Numerische Mathematik* 18(3), 224-242, DOI 10.1007/BF01397083 (exact products and quotient remainders).

use std::f64::consts::SQRT_2;

use super::betacf;
use super::large_shape::LN2_HI;
use super::large_shape::LN2_LO;
use super::large_shape::ln_ratio;
use super::large_shape::two_sum;
use crate::special::mills_ratio;

/// Up to this `|ln_scale|` a tail branch's plain `c·ξ` rounds by a few `ε` at most; past it `refined` re-forms it.
const REFINE: f64 = 1.0;

const TWO_THIRDS: (f64, f64) = (0.6666666666666666, 3.700743415417188e-17);

/// `e^{64·LN2_LO}` and `2⁻⁶⁴`, the frame `e^{ln_scale + 64 ln 2}` a value below `e^{−700}` is formed in.
const FRAME: (f64, f64) = (1.0000000122125756, 1.0 / 18_446_744_073_709_551_616.0);

/// Below this `ν/2` the expansion is taken at `ν/2 + n ≥ LIFT`, the `n` steps between by the upward recurrence: at `T = 7.75`
/// its terms fall by `(2n)!/(2πT)^{2n}`, under `1e-16` by the eleventh.
const LIFT: f64 = 8.0;

/// From this `ξ = ln(1 + x²/ν)` on the series in `w = e^{−ξ}` converges geometrically by `0.22`, and the expansion's
/// `ξ^{2n}` would need its fourteenth term.
const FAR_TAIL: f64 = 1.5;

/// `d_n`, the coefficients of `(sinh(s/2)/(s/2))^{−½} = Σ d_n s^{2n}`, exact rationals rounded; past `d_14` the expansion's
/// terms are below `1e-19` of its sum wherever it is used.
const D: [f64; 15] = [
  1.0,
  -1.0 / 48.0,
  1.0 / 2560.0,
  -61.0 / 7741440.0,
  1261.0 / 7431782400.0,
  -79.0 / 20761804800.0,
  66643.0 / 761775532277760.0,
  -2.044523359411974e-12,
  4.833351797967704e-14,
  -1.152434101767386e-15,
  2.76605204359937e-17,
  -6.67428195089166e-19,
  1.61745507718158e-20,
  -3.93397792009138e-22,
  9.597634062586047e-24,
];

/// `P(T ≤ −|x|)`: `½ − ½·factor·e^{ln_scale}` near zero (`center`), else `½·factor·e^{ln_scale}`; `xi = ln(1 + x²/ν)`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StudentTail {
  center: bool,
  factor: f64,
  ln_scale: f64,
  /// `c` of a tail branch's `ln_scale = −c·ξ`, as an exact sum of two doubles; `(0, 0)` where `ξ` stays bounded.
  rate: (f64, f64),
  pub(crate) xi: f64,
}

impl StudentTail {
  /// `P(T ≤ −|x|)`, at most ½ to the bit with a NaN kept; below `e^{−700}` by `scaled_exp`, so a subnormal one rounds
  /// at its last product.
  pub(crate) fn lower(self) -> f64 {
    if self.center {
      return 0.5 - 0.5 * self.factor * self.ln_scale.exp();
    }
    let half = 0.5 * self.factor;
    if self.ln_scale < -700.0 {
      return scaled_exp(half, self.ln_scale, 0.0);
    }
    let lower = half * self.ln_scale.exp();
    if lower > 0.5 { 0.5 } else { lower }
  }

  /// `P(T ≤ |x|)`, near zero as `½ + ½I_z` rather than one minus the lower half.
  pub(crate) fn upper(self) -> f64 {
    if self.center {
      0.5 + 0.5 * self.factor * self.ln_scale.exp()
    } else {
      1.0 - self.lower()
    }
  }

  /// `ln_scale` re-formed in two parts where a tail branch's passes `−REFINE`, its low part moved into `factor`, so `lower`
  /// keeps its factor's digits where `|ln F|` nears 745; the quantile's `ln_lower` needs no such care.
  pub(crate) fn refined(self, nu: f64, x: f64) -> Self {
    if self.rate.0 == 0.0 || !(self.ln_scale < -REFINE && self.ln_scale > -800.0) {
      return self;
    }
    let (ln_scale, lo) = tail_exponent(nu, x, self.rate);
    Self {
      factor: self.factor.mul_add(lo, self.factor),
      ln_scale,
      ..self
    }
  }

  /// `ln P(T ≤ −|x|)`, finite where the probability underflows, at most `−ln 2` with a NaN kept.
  pub(crate) fn ln_lower(self) -> f64 {
    if self.center {
      return self.lower().ln();
    }
    let ln_lower = (0.5 * self.factor).ln() + self.ln_scale;
    if ln_lower > -std::f64::consts::LN_2 {
      -std::f64::consts::LN_2
    } else {
      ln_lower
    }
  }
}

/// `(z, w, ξ)` at `|x|`: `x²/(ν + x²)`, `ν/(ν + x²)` and `ln(1 + x²/ν)`, from `x²/ν` up to `x² = ν` and from `ν/x²` past
/// it, its logarithm from `ln ν − 2 ln|x|` once `ν/x²` is subnormal, so none overflows or cancels.
#[inline]
fn geometry(nu: f64, x: f64) -> (f64, f64, f64) {
  if x * x <= nu {
    let s = x * x / nu;
    return (s / (1.0 + s), 1.0 / (1.0 + s), s.ln_1p());
  }
  let r = nu / x / x;
  let ln_r = if r >= f64::MIN_POSITIVE {
    r.ln()
  } else {
    nu.ln() - 2.0 * x.ln()
  };
  (1.0 / (1.0 + r), r / (1.0 + r), r.ln_1p() - ln_r)
}

/// Up to this exponent the density's plain form rounds by about three ulps a unit of it; past it `student_density` forms the
/// exponent in two parts, where it would otherwise reach a thousand ulps.
const DENSITY_REFINE: f64 = 8.0;

/// The t density `R(ν/2)/√(2π) e^{−((ν + 1)/2)ξ}`, its exponent in two parts past `DENSITY_REFINE`; inline, so a loop at
/// one `ν` keeps `ln(R/√(2π))` out of it as the generic law's did.
#[inline]
pub(crate) fn student_density(nu: f64, ratio: f64, x: f64) -> f64 {
  density(nu, ratio, x, DENSITY_REFINE)
}

/// `student_density` with the exponent in two parts past `REFINE`, a few ulps at every `x`.
pub(crate) fn student_density_precise(nu: f64, ratio: f64, x: f64) -> f64 {
  density(nu, ratio, x, REFINE)
}

#[inline]
fn density(nu: f64, ratio: f64, x: f64, refine: f64) -> f64 {
  let s = x * x / nu;
  let xi = if s.is_finite() {
    s.ln_1p()
  } else {
    student_xi(nu, x)
  };
  let exponent = 0.5 * (nu + 1.0) * xi;
  if exponent > refine && exponent < 800.0 {
    return refined_density(nu, ratio, x);
  }
  ((ratio / std::f64::consts::TAU.sqrt()).ln() - exponent).exp()
}

#[inline(never)]
fn refined_density(nu: f64, ratio: f64, x: f64) -> f64 {
  let (hi, lo) = tail_exponent(nu, x, (0.5 * nu, 0.5));
  scaled_exp(ratio / std::f64::consts::TAU.sqrt(), hi, lo)
}

/// `−c·ln(1 + x²/ν)` in two parts, `c = rate.0 + rate.1` exactly.
fn tail_exponent(nu: f64, x: f64, rate: (f64, f64)) -> (f64, f64) {
  let (c, c_lo) = two_sum(rate.0, rate.1);
  let (xi, xi_lo) = xi_parts(nu, x.abs());
  let p = c * xi;
  (-p, -(c.mul_add(xi, -p) + c * xi_lo + c_lo * xi))
}

/// `factor·e^{hi + lo}` for a small `lo`; below `e^{−700}` `hi + 64·LN2_HI` is exact and its exponential normal, so only
/// the last product rounds into the subnormals.
fn scaled_exp(factor: f64, hi: f64, lo: f64) -> f64 {
  let factor = factor.mul_add(lo, factor);
  if hi < -700.0 {
    factor * FRAME.0 * (hi + 64.0 * LN2_HI).exp() * FRAME.1
  } else {
    factor * hi.exp()
  }
}

/// `ξ = ln(1 + x²/ν)` in two parts at `|x|`, `x²/ν` from the exact square and the quotient's remainder; where that
/// overflows `2 ln|x| − ln ν`, which `ν/x²` no longer moves.
fn xi_parts(nu: f64, x: f64) -> (f64, f64) {
  let x2 = x * x;
  let s = x2 / nu;
  if s.is_finite() {
    let s_lo = ((-s).mul_add(nu, x2) + x.mul_add(x, -x2)) / nu;
    return ln_1p_parts(s, s_lo);
  }
  let ((lx, lx_lo), (ln, ln_lo)) = (ln_ratio(x, 1.0), ln_ratio(nu, 1.0));
  let (h, e) = two_sum(2.0 * lx, -ln);
  (h, e + 2.0 * lx_lo - ln_lo)
}

/// `ln(1 + s)` in two parts for `s ≥ 0` given in two: `2 atanh(s/(2 + s))` below `√2 − 1`, else `k ln 2 + 2 atanh(y)`,
/// `y = (m − 1)/(m + 1)`, `1 + s = 2ᵏm`, `m ∈ [1/√2, √2)`; `2y` and `2y³/3` compensated, the rest below `2y⁵/5`.
fn ln_1p_parts(s: f64, s_lo: f64) -> (f64, f64) {
  let (k, num, num_lo, den, den_lo) = if s < SQRT_2 - 1.0 {
    let (d, d_err) = two_sum(2.0, s);
    (0.0, s, s_lo, d, d_err + s_lo)
  } else {
    let (q, q_err) = two_sum(1.0, s);
    let bits = q.to_bits();
    let mut k = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m > SQRT_2 {
      m *= 0.5;
      k += 1;
    }
    let m_lo = (q_err + s_lo) * (m / q);
    let (n, n_err) = two_sum(m - 1.0, m_lo);
    let (d, d_err) = two_sum(m, 1.0);
    (f64::from(k), n, n_err, d, d_err + m_lo)
  };
  let y = num / den;
  let y_lo = ((-y).mul_add(den, num) + num_lo - y * den_lo) / den;
  let y2 = y * y;
  let y2_lo = y.mul_add(y, -y2) + 2.0 * y * y_lo;
  let y3 = y2 * y;
  let y3_lo = y2.mul_add(y, -y3) + y2 * y_lo + y2_lo * y;
  let t = y3 * TWO_THIRDS.0;
  let t_lo = y3.mul_add(TWO_THIRDS.0, -t) + y3 * TWO_THIRDS.1 + y3_lo * TWO_THIRDS.0;
  // `Σ_j y^{2j}/(2j + 5)` to `j = 10`: at `y² ≤ 0.0295` the next term is under `1e-17` of the sum.
  let rest = [
    1.0 / 25.0,
    1.0 / 23.0,
    1.0 / 21.0,
    1.0 / 19.0,
    1.0 / 17.0,
    1.0 / 15.0,
    1.0 / 13.0,
    1.0 / 11.0,
    1.0 / 9.0,
    1.0 / 7.0,
    0.2,
  ]
  .iter()
  .fold(0.0_f64, |acc, &c| acc.mul_add(y2, c));
  let (h, h_err) = two_sum(k * LN2_HI, 2.0 * y);
  let (g, g_err) = two_sum(h, t);
  two_sum(
    g,
    h_err + g_err + k * LN2_LO + 2.0 * y_lo + t_lo + 2.0 * y3 * y2 * rest,
  )
}

/// `ln(1 + x²/ν)` at any `x`, without overflow or cancellation.
#[inline]
fn student_xi(nu: f64, x: f64) -> f64 {
  geometry(nu, x.abs()).2
}

/// The t law's lower tail at `|x|`, `ratio = R(ν/2)`: `I_z(½, ν/2)`'s series near zero, `I_w(ν/2, ½)`'s in the far tail,
/// between them Lentz's fraction below `ν/2 = 8` and `w*`, else the expansion, lifted past `ν/2 = 8` by 8.17.20.
pub(crate) fn student_tail(nu: f64, ratio: f64, x: f64) -> StudentTail {
  let x = x.abs();
  let a = 0.5 * nu;
  let (z, w, xi) = geometry(nu, x);
  // `Γ(a + ½)/Γ(a + 1) = R(a)/√a`, which stays normal as `a → 0` where `√(πa)` would not.
  let scaled = ratio / a.sqrt();
  if z < 0.29 && z * a <= 0.23 {
    // `I_z(½, a) = |x| √(2/π) R(a) w^{a + ½} Σ (a + ½)_n z^n/(3/2)_n`, `R(a) = Γ(a + ½)/(Γ(a)√a)`.
    let sum = series(|n| z * (a + 0.5 + n) / (1.5 + n));
    return StudentTail {
      center: true,
      factor: x * (2.0 / std::f64::consts::PI).sqrt() * ratio * sum,
      ln_scale: -(a + 0.5) * xi,
      rate: (0.0, 0.0),
      xi,
    };
  }
  if xi > FAR_TAIL {
    // `I_w(a, ½) = √z R(a)/√(πa) w^a Σ (a + ½)_n w^n/(a + 1)_n`.
    let sum = series(|n| w * (a + 0.5 + n) / (a + 1.0 + n));
    return StudentTail {
      center: false,
      factor: z.sqrt() * scaled / std::f64::consts::PI.sqrt() * sum,
      ln_scale: -a * xi,
      rate: (a, 0.0),
      xi,
    };
  }
  if a >= LIFT {
    return StudentTail {
      center: false,
      factor: expansion(a, xi, ratio),
      ln_scale: -(a - 0.25) * xi,
      rate: (a, -0.25),
      xi,
    };
  }
  if w <= (a + 1.0) / (a + 2.5) {
    // `I_w(a, ½) = √z R(a)/√(πa) w^a h`, `h` Lentz's fraction, which converges in under 20 steps below NR's switch `w*`.
    return StudentTail {
      center: false,
      factor: z.sqrt() * scaled / std::f64::consts::PI.sqrt() * betacf(a, 0.5, w),
      ln_scale: -a * xi,
      rate: (0.0, 0.0),
      xi,
    };
  }
  // `I_w(a, ½) − I_w(b, ½) = Σ_{i<n} Γ(a + i + ½)/(Γ(a + i + 1)√π) w^{a+i} √z`, `b = a + n`, every term positive; the same
  // steps carry `R(b) = Γ(a + ½)/Γ(a + 1) · (a + ½)/√b · Π_{0<j<n} (a + j + ½)/(a + j)`.
  let n = (LIFT - a).ceil();
  let b = a + n;
  let mut term = scaled / std::f64::consts::PI.sqrt() * z.sqrt();
  let (mut steps, mut lift, mut i) = (0.0, scaled * (a + 0.5) / b.sqrt(), 0.0);
  while i < n {
    let inv = (a + i + 1.0).recip();
    steps += term;
    term *= w * (a + i + 0.5) * inv;
    lift *= (a + i + 1.5) * inv;
    i += 1.0;
  }
  // The last step carried one factor `(b + ½)/b` past `R(b)`.
  let ratio_b = lift * b / (b + 0.5);
  StudentTail {
    center: false,
    factor: steps + expansion(b, xi, ratio_b) * (-(n - 0.25) * xi).exp(),
    ln_scale: -a * xi,
    rate: (0.0, 0.0),
    xi,
  }
}

/// `Σ t_n`, `t_0 = 1`, `t_{n+1} = t_n·ratio(n)`, positive terms until one is below `1e-17` of the sum.
fn series(ratio: impl Fn(f64) -> f64) -> f64 {
  let (mut term, mut sum, mut n) = (1.0, 1.0, 0.0);
  while n < 200.0 {
    term *= ratio(n);
    sum += term;
    if term <= 1e-17 * sum {
      break;
    }
    n += 1.0;
  }
  sum
}

/// `I_w(a, ½) e^{Tξ}` by D&M's (9.1): `R(a)√(a/T)/√π Σ d_n g_n`, `g_n = Γ(2n + ½, u)e^u/T^{2n}`, `u = Tξ`, `T = a − ¼`,
/// from `g_0 = √2 R_M(√(2u))` (Mills' ratio) by `T²g_{n+1} = (p + 1)p g_n + √u ξ^{2n}(p + 1 + u)`, `p = 2n + ½`.
fn expansion(a: f64, xi: f64, ratio: f64) -> f64 {
  let t = a - 0.25;
  let u = t * xi;
  let (root_u, xi2, t2) = (u.sqrt(), xi * xi, t * t);
  let mut g = std::f64::consts::SQRT_2 * mills_ratio((2.0 * u).sqrt());
  let (mut sum, mut power) = (g, 1.0);
  for (n, &d) in D.iter().enumerate().skip(1) {
    let p = 2.0 * (n - 1) as f64 + 0.5;
    g = ((p + 1.0) * p * g + root_u * power * (p + 1.0 + u)) / t2;
    power *= xi2;
    let term = d * g;
    sum += term;
    if term.abs() <= 1e-17 * sum {
      break;
    }
  }
  ratio * (a / t).sqrt() / std::f64::consts::PI.sqrt() * sum
}

#[cfg(test)]
mod tests;
