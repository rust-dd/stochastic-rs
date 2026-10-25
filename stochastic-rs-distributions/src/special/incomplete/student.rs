//! The Student-t law's incomplete beta, `I_w(ν/2, ½)` at `w = ν/(ν + x²)` and its reflection `I_z(½, ν/2)`, `z = 1 − w`:
//! the series in `z` near zero, in `w` in the far tail, between them Lentz's fraction or the expansion in incomplete gammas.
//! Reference: Abramowitz, M., Stegun, I.A. (1964), "Handbook of Mathematical Functions", NBS AMS 55, 26.7.1 (`A(t|ν) = 1 − I_w(ν/2, ½)`).
//! Reference: NIST Digital Library of Mathematical Functions, eqs. 8.17.4 (the reflection), 8.17.8 (the series of positive terms) and 8.17.20 (the step `a → a + 1`), <https://dlmf.nist.gov/8.17>.
//! Reference: DiDonato, A.R., Morris, A.H. (1992), "Algorithm 708: Significant digit computation of the incomplete beta function ratios", ACM TOMS 18(3), 360-373, DOI 10.1145/131766.131776, eqs. (9.1)–(9.2) (the expansion in `Γ(b + 2n, Tξ)`, `T = a + (b − 1)/2`).

use super::betacf;
use crate::special::mills_ratio;

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
  pub(crate) xi: f64,
}

impl StudentTail {
  /// `P(T ≤ −|x|)`, at most ½ to the bit; a value below `e^{−700}` is formed in one exponential, so a subnormal one is
  /// rounded once.
  pub(crate) fn lower(self) -> f64 {
    if self.center {
      return 0.5 - 0.5 * self.factor * self.ln_scale.exp();
    }
    let half = 0.5 * self.factor;
    if self.ln_scale < -700.0 {
      (self.ln_scale + half.ln()).exp()
    } else {
      (half * self.ln_scale.exp()).min(0.5)
    }
  }

  /// `P(T ≤ |x|)`, near zero as `½ + ½I_z` rather than one minus the lower half.
  pub(crate) fn upper(self) -> f64 {
    if self.center {
      0.5 + 0.5 * self.factor * self.ln_scale.exp()
    } else {
      1.0 - self.lower()
    }
  }

  /// `ln P(T ≤ −|x|)`, finite where the probability underflows.
  pub(crate) fn ln_lower(self) -> f64 {
    if self.center {
      self.lower().ln()
    } else {
      ((0.5 * self.factor).ln() + self.ln_scale).min(-std::f64::consts::LN_2)
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

/// `ln(1 + x²/ν)` at any `x`, without overflow or cancellation.
#[inline]
pub(crate) fn student_xi(nu: f64, x: f64) -> f64 {
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
      xi,
    };
  }
  if a >= LIFT {
    return StudentTail {
      center: false,
      factor: expansion(a, xi, ratio),
      ln_scale: -(a - 0.25) * xi,
      xi,
    };
  }
  if w <= (a + 1.0) / (a + 2.5) {
    // `I_w(a, ½) = √z R(a)/√(πa) w^a h`, `h` Lentz's fraction, which converges in under 20 steps below NR's switch `w*`.
    return StudentTail {
      center: false,
      factor: z.sqrt() * scaled / std::f64::consts::PI.sqrt() * betacf(a, 0.5, w),
      ln_scale: -a * xi,
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
