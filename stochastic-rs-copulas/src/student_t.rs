//! The Student-t law every t copula and the t pair copula evaluate through: `SimdStudentT`'s cdf and quantile, the
//! log-density its `pdf` exponentiates, the t copula's log-density at t scores, and the one rule for `nu`.
//! Reference: Abramowitz, M., Stegun, I.A. (1964), "Handbook of Mathematical Functions", NBS AMS 55, 26.7.5 (the Cornish–Fisher expansion of the t quantile).

use std::f64::consts::LN_2;
use std::f64::consts::PI;

use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray::Axis;
use stochastic_rs_distributions::DistributionExt;
use stochastic_rs_distributions::special::ln_gamma;
use stochastic_rs_distributions::special::ln_norm_cdf;
use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_distributions::studentt::SimdStudentT;

use crate::bivariate::conditional::ln_add_exp;
use crate::bivariate::conditional::ln_ratio;
use crate::bivariate::integral::rule;
use crate::error::CopulaError;

/// Above this `ν` the law is normal to rounding: the expansion's first term `(x³ + x)/(4ν)` moves the quantile, and `Φ`
/// by under `2⁻⁵⁴` of itself wherever `Φ(x) ≥ 2⁻¹⁰⁷⁴`, `|x| ≤ 38.5`, as `(38.5⁴ + 38.5²)/4 · 2⁵⁴ < 10²²`.
pub(crate) const NORMAL_ABOVE: f64 = 1e22;

/// The law of the t copulas' h-function and its inverse: `SimdStudentT` in the body, the leading tail term
/// `P(T ≤ −t) = (K/ν)t^{−ν}` where its first correction `ν²(ν + 1)/(2(ν + 2)t²)` is below `2⁻⁵⁴`, the normal law past `NORMAL_ABOVE`.
#[derive(Debug, Clone)]
pub(crate) struct Law {
  nu: f64,
  inner: SimdStudentT<f64>,
  ln_scale: f64,
  tail_level: f64,
  tail_start: f64,
  ln_tail_start: f64,
}

/// A quantile `t`, `+∞` where it overflows, and in the tail `ln|t|` and `ln P(T ≤ −|t|)`, which stay finite there.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Score {
  pub(crate) value: f64,
  tail_ln_abs: f64,
  ln_tail: f64,
  /// `P(T ≤ −|t|)`, exact, where the tail holds.
  tail: f64,
}

impl Score {
  fn of(value: f64) -> Self {
    Self {
      value,
      tail_ln_abs: f64::NAN,
      ln_tail: f64::NAN,
      tail: f64::NAN,
    }
  }

  /// `ln|t|`: the tail's own, else formed from the value on demand, as only the overflow paths read it.
  pub(crate) fn ln_abs(&self) -> f64 {
    if self.ln_tail.is_nan() {
      self.value.abs().ln()
    } else {
      self.tail_ln_abs
    }
  }

  /// The sign of `t`, 0 at the median.
  pub(crate) fn sign(&self) -> f64 {
    if self.value == 0.0 {
      0.0
    } else {
      self.value.signum()
    }
  }
}

impl Law {
  /// `ln(K/ν) = ln Γ((ν + 1)/2) − ln Γ(1 + ν/2) + (ν/2) ln ν − ln(2√π)`, `K = Γ((ν + 1)/2)ν^{ν/2}/(√π Γ(ν/2))`, free of the
  /// cancellation of `ln Γ(ν/2)` against `ln ν` at small `ν`; the tail holds from `t² = 2⁵³ ν(ν + 1)/(ν + 2)` for the quantile.
  pub(crate) fn new(nu: f64) -> Self {
    let ln_scale = ln_gamma(0.5 * (nu + 1.0)) - ln_gamma(1.0 + 0.5 * nu) + 0.5 * nu * nu.ln()
      - (2.0 * PI.sqrt()).ln();
    let ln_ratio = ((nu + 1.0) / (nu + 2.0)).ln();
    let ln_quantile_start = 0.5 * (53.0 * LN_2 + nu.ln() + ln_ratio);
    let ln_tail_start = nu.ln() + 26.5 * LN_2 + 0.5 * ln_ratio;
    Self {
      nu,
      inner: SimdStudentT::new(nu),
      ln_scale,
      tail_level: (ln_scale - nu * ln_quantile_start).exp(),
      tail_start: ln_tail_start.exp(),
      ln_tail_start,
    }
  }

  /// The quantile at `p`; `±∞` at `p = 0, 1`, and a normal quantile at a subnormal tail refined by a Newton step on `ln Φ`.
  pub(crate) fn quantile(&self, p: f64) -> Score {
    if p <= 0.0 || p >= 1.0 {
      return Score::of(if p <= 0.0 {
        f64::NEG_INFINITY
      } else {
        f64::INFINITY
      });
    }
    if p == 0.5 {
      return Score::of(0.0);
    }
    let (tail, sign) = if p < 0.5 { (p, -1.0) } else { (1.0 - p, 1.0) };
    if self.nu >= NORMAL_ABOVE {
      return Score::of(ndtri(p));
    }
    if tail <= self.tail_level {
      let ln_tail = tail.ln();
      let ln_abs = (self.ln_scale - ln_tail) / self.nu;
      return Score {
        value: sign * ln_abs.exp(),
        tail_ln_abs: ln_abs,
        ln_tail,
        tail,
      };
    }
    Score::of(
      self
        .inner
        .quantile(p)
        .expect("the Student-t law has a quantile"),
    )
  }

  /// Whether `p` is in the body, where the quantile is `SimdStudentT`'s.
  fn in_body(&self, p: f64) -> bool {
    p.min(1.0 - p) > self.tail_level && p != 0.5 && self.nu < NORMAL_ABOVE
  }

  /// `SimdStudentT`'s quantile wherever `u` is in the body, NaN elsewhere, in a loop of its own: one the quantile
  /// inlines into whole, so its work in `ν` alone leaves the loop.
  pub(crate) fn body_quantiles(&self, u: ArrayView2<f64>) -> Array2<f64> {
    let nu = self.nu;
    let mut z = u.to_owned();
    for mut row in z.axis_iter_mut(Axis(0)) {
      for p in row.iter_mut() {
        *p = if self.in_body(*p) {
          quantile(*p, nu)
        } else {
          f64::NAN
        };
      }
    }
    z
  }

  /// The quantile at `p`, from `t`, its `body_quantiles` entry, where `p` is in the body.
  pub(crate) fn score(&self, p: f64, t: f64) -> Score {
    if self.in_body(p) {
      Score::of(t)
    } else {
      self.quantile(p)
    }
  }

  /// `ln(|a|/|b|)` of two of this law's quantiles: `ln(P_b/P_a)/ν` of their tail probabilities when both have one, so
  /// the ratio of two quantiles past `f64::MAX` stays exact, and so does that of two near ones.
  pub(crate) fn ln_ratio(&self, a: Score, b: Score) -> f64 {
    if a.ln_tail.is_nan() || b.ln_tail.is_nan() {
      a.ln_abs() - b.ln_abs()
    } else {
      ln_ratio(b.tail, a.tail) / self.nu
    }
  }

  /// The cdf at a finite or infinite `t`.
  pub(crate) fn cdf_at(&self, t: f64) -> f64 {
    if self.nu >= NORMAL_ABOVE {
      return norm_cdf(t);
    }
    if t.abs() >= self.tail_start {
      return self.tail(t, self.ln_scale - self.nu * t.abs().ln());
    }
    self.inner.cdf(t).expect("the Student-t law has a cdf")
  }

  /// The cdf at `sign · e^{ln_abs}`, a magnitude given by its logarithm, which may pass `f64::MAX`.
  pub(crate) fn cdf(&self, sign: f64, ln_abs: f64) -> f64 {
    if self.nu < NORMAL_ABOVE && ln_abs >= self.ln_tail_start {
      return self.tail(sign, self.ln_scale - self.nu * ln_abs);
    }
    self.cdf_at(sign * ln_abs.exp())
  }

  /// The cdf at `sign · |t| e^{ln_factor}` for a quantile `t`, through its tail probability `P(T ≤ −|t|)e^{−ν ln_factor}`
  /// where both lie in the tail, which needs no `ln|t|`.
  pub(crate) fn cdf_scaled(&self, t: Score, sign: f64, ln_factor: f64) -> f64 {
    if ln_factor == f64::NEG_INFINITY {
      return self.cdf_at(0.0);
    }
    if t.ln_tail.is_nan() || self.nu >= NORMAL_ABOVE {
      return self.cdf(sign, t.ln_abs() + ln_factor);
    }
    if t.ln_abs() + ln_factor >= self.ln_tail_start {
      return self.tail(sign, t.ln_tail - self.nu * ln_factor);
    }
    self.cdf_at(sign * (t.ln_abs() + ln_factor).exp())
  }

  /// `ln P(T ≤ sign · e^{ln_abs})` to a relative rounding in both tails, the upper one as `ln(1 − P(T ≤ −|t|))`.
  pub(crate) fn ln_cdf(&self, sign: f64, ln_abs: f64) -> f64 {
    let ln_lower = if self.nu >= NORMAL_ABOVE {
      ln_norm_cdf(-ln_abs.exp())
    } else if ln_abs >= self.ln_tail_start {
      self.ln_scale - self.nu * ln_abs
    } else {
      self
        .inner
        .cdf(-ln_abs.exp())
        .expect("the Student-t law has a cdf")
        .ln()
    };
    if sign < 0.0 {
      ln_lower
    } else {
      (-ln_lower.exp()).ln_1p()
    }
  }

  /// `ν ln|t|` of a quantile, from its tail probability where it has one, so it stays exact past `f64::MAX`.
  pub(crate) fn nu_ln_abs(&self, t: Score) -> f64 {
    if t.ln_tail.is_nan() {
      self.nu * t.ln_abs()
    } else {
      self.ln_scale - t.ln_tail
    }
  }

  fn tail(&self, sign: f64, ln_tail: f64) -> f64 {
    let tail = ln_tail.exp();
    if sign < 0.0 { tail } else { 1.0 - tail }
  }
}

/// `0 < nu < ∞` and `nu / 2 > 0`, which only `5e-324` fails: what `SimdStudentT::new` and `SimdChiSquared::new` assert, so
/// the t copulas and the vines' t pairs refuse up front a `nu` those laws would panic on.
pub(crate) fn check_nu(nu: f64) -> Result<(), CopulaError> {
  let constraint = if !nu.is_finite() || nu <= 0.0 {
    "0.0 < nu < inf"
  } else if nu * 0.5 == 0.0 {
    "nu / 2 > 0"
  } else {
    return Ok(());
  };
  Err(CopulaError::InvalidParameter {
    name: "nu",
    value: nu,
    constraint: constraint.into(),
  })
}

pub(crate) fn cdf(x: f64, nu: f64) -> f64 {
  SimdStudentT::new(nu)
    .cdf(x)
    .expect("the Student-t law has a cdf")
}

/// `∓∞` at `p = 0` and `p = 1`; inlined, so a loop over `p` at one `nu` hoists the work that depends on `nu` alone.
#[inline(always)]
pub(crate) fn quantile(p: f64, nu: f64) -> f64 {
  SimdStudentT::new(nu)
    .quantile(p)
    .expect("the Student-t law has a quantile")
}

/// The t copula's log-density at one `(ρ, ν)`, with what no point changes formed once.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TDensity {
  rho: f64,
  nu: f64,
  /// `G − ½ ln(1 − ρ²)`, `G = ln(Γ(ν/2 + 1)Γ(ν/2)/Γ(ν/2 + ½)²)`.
  constant: f64,
  inv_one_minus_rho2: f64,
  inv_nu: f64,
}

impl TDensity {
  pub(crate) fn new(rho: f64, nu: f64) -> Self {
    let one_minus_rho2 = (1.0 - rho) * (1.0 + rho);
    Self {
      rho,
      nu,
      constant: ln_gamma_constant(nu) - 0.5 * one_minus_rho2.ln(),
      inv_one_minus_rho2: one_minus_rho2.recip(),
      inv_nu: nu.recip(),
    }
  }

  /// `ln c` at finite scores, `moderate` where it holds, else `ln_density` at their features.
  pub(crate) fn ln_c(&self, x: f64, y: f64) -> f64 {
    self.moderate(x, y).unwrap_or_else(|| {
      let (fx, fy) = (Feature::of_value(x, self.nu), Feature::of_value(y, self.nu));
      ln_density(&fx, &fy, fy.abs_t - fx.abs_t, self.rho, self.nu)
    })
  }

  /// `G − ½ ln(1 − ρ²) − ((ν + 2)/2) ln(1 + Q/ν) + ((ν + 1)/2) ln((1 + x²/ν)(1 + y²/ν))`, `Q = (x − ρy)²/(1 − ρ²) + y²`,
  /// `x − ρy` by one fma, each term within `15u`: `None` unless the terms' sizes sum to `S` with `9εS ≤ 1e-12 max(1, |ln c|)`.
  pub(crate) fn moderate(&self, x: f64, y: f64) -> Option<f64> {
    let d = (-self.rho).mul_add(y, x);
    let q = d * d * self.inv_one_minus_rho2 + y * y;
    let joint = -0.5 * (self.nu + 2.0) * (q * self.inv_nu).ln_1p();
    let margins =
      0.5 * (self.nu + 1.0) * ((x * x * self.inv_nu).ln_1p() + (y * y * self.inv_nu).ln_1p());
    let sum = self.constant + joint + margins;
    let size = self.constant.abs() + joint.abs() + margins;
    (size.is_finite() && 9.0 * f64::EPSILON * size <= 1e-12 * sum.abs().max(1.0)).then_some(sum)
  }
}

/// A score `s` in `t = asinh(s/√ν)`, the coordinate the bivariate t's conditional law reads: `|t|` (rounded where huge),
/// `ν|t|` exactly, and what an exact `|t_b| − |t_a|` needs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Feature {
  pub(crate) sign: f64,
  pub(crate) abs_t: f64,
  pub(crate) nu_abs_t: f64,
  ln_abs: f64,
  tail: f64,
  /// `|t| − (ln|s| − ½ ln ν)` where that is positive, else NaN.
  corr: f64,
}

impl Feature {
  fn new(sign: f64, ln_abs: f64, nu_ln_abs: f64, tail: f64, nu: f64) -> Self {
    if sign == 0.0 {
      return Self {
        sign,
        abs_t: 0.0,
        nu_abs_t: 0.0,
        ln_abs,
        tail,
        corr: f64::NAN,
      };
    }
    let half_ln_nu = 0.5 * nu.ln();
    let q = ln_abs - half_ln_nu;
    let (abs_t, nu_abs_t, corr) = if q > 0.0 {
      let corr = (1.0 + (-2.0 * q).exp()).sqrt().ln_1p();
      (q + corr, nu_ln_abs - nu * half_ln_nu + nu * corr, corr)
    } else {
      let abs_t = q.exp().asinh();
      (abs_t, nu * abs_t, f64::NAN)
    };
    Self {
      sign,
      abs_t,
      nu_abs_t,
      ln_abs,
      tail,
      corr,
    }
  }

  pub(crate) fn of_value(s: f64, nu: f64) -> Self {
    let ln_abs = s.abs().ln();
    let sign = if s == 0.0 { 0.0 } else { s.signum() };
    Self::new(sign, ln_abs, nu * ln_abs, f64::NAN, nu)
  }
}

impl Law {
  pub(crate) fn feature(&self, s: Score) -> Feature {
    Feature::new(s.sign(), s.ln_abs(), self.nu_ln_abs(s), s.tail, self.nu)
  }

  /// `|t_b| − |t_a|`, from the tail probabilities where both scores have one, so it stays exact past `f64::MAX`.
  pub(crate) fn abs_t_gap(&self, a: &Feature, b: &Feature) -> f64 {
    if a.corr.is_nan() || b.corr.is_nan() {
      return b.abs_t - a.abs_t;
    }
    let ln_gap = if a.tail.is_nan() || b.tail.is_nan() {
      b.ln_abs - a.ln_abs
    } else {
      ln_ratio(a.tail, b.tail) / self.nu
    };
    ln_gap + (b.corr - a.corr)
  }

  /// `x + y` of the quantiles of `u` and `v` from `x̂`, the exact gap `g = u + v − 1` and `start`: Newton on
  /// `T(x̂) − T(x̂ − d) = g + r`, `r = T(x̂) − u`, the mass by Gauss–Legendre of the density, less `r/f(x̂)`.
  pub(crate) fn quantile_sum(&self, x: f64, u: f64, g: f64, start: f64) -> f64 {
    if g == 0.0 {
      return 0.0;
    }
    // `ln Γ((ν + 1)/2) − ln Γ(ν/2) − ½ ln(νπ) = −½(ln 2π + G)`, free of the gammas' cancellation at large `ν`.
    let ln_norm = -0.5 * (std::f64::consts::TAU.ln() + ln_gamma_constant(self.nu));
    let pdf = |s: f64| (ln_norm - 0.5 * (self.nu + 1.0) * (s * s / self.nu).ln_1p()).exp();
    // `T(x) − T(x − d)` over a `d` within a quarter of the density's scale `√(ν + s²)`, as the caller keeps it.
    let mass = |d: f64| {
      let (mid, half) = (x - 0.5 * d, 0.5 * d);
      half
        * rule()
          .iter()
          .map(|&(t, w)| w * pdf(mid + half * t))
          .sum::<f64>()
    };
    let r = if x <= 0.0 {
      self.cdf_at(x) - u
    } else {
      (1.0 - u) - self.cdf_at(-x)
    };
    let (target, mut d) = (g + r, start);
    for _ in 0..8 {
      let step = (target - mass(d)) / pdf(x - d);
      d += step;
      if step.abs() <= 1e-16 * d.abs() {
        break;
      }
    }
    d - r / pdf(x)
  }
}

/// `asinh(a + δ) − asinh(a)` for `a, a + δ ≥ 0`, without the cancellation of the two `asinh`; past `a = 1e8` the ratio's
/// logarithm, which it equals to under `1/a²`.
pub(crate) fn asinh_shift(a: f64, delta: f64) -> f64 {
  if a > 1e8 {
    return (delta / a).ln_1p();
  }
  let b = a + delta;
  (delta * (1.0 + (a + b) / (a.hypot(1.0) + b.hypot(1.0))) / (a + a.hypot(1.0))).ln_1p()
}

/// `ln cosh t`, by `ln_1p(2 sinh²(t/2))` below 1, where `|t| − ln 2 + ln_1p(e^{−2|t|})` cancels.
pub(crate) fn ln_cosh(t: f64) -> f64 {
  let a = t.abs();
  if a < 1.0 {
    (2.0 * (0.5 * a).sinh().powi(2)).ln_1p()
  } else {
    a - LN_2 + (-2.0 * a).exp().ln_1p()
  }
}

/// `ln(Γ(ν/2 + 1)Γ(ν/2)/Γ(ν/2 + ½)²)`: Stirling's series from `ν = 40`, where the gammas' logarithms cancel, its
/// truncation under `4e-17`; `1/(4z) − 1/(96z³) + 1/(320z⁵) − 17/(7168z⁷) + 31/(9216z⁹)`, `z = ν/2`, proved in sympy.
/// Reference: NIST Digital Library of Mathematical Functions, eq. 5.11.1 (Stirling's series), <https://dlmf.nist.gov/5.11.1>.
pub(crate) fn ln_gamma_constant(nu: f64) -> f64 {
  let z = 0.5 * nu;
  if z < 20.0 {
    // `ln Γ(z) = ln Γ(z + 1) − ln z`, finite where `Γ(z)` overflows.
    return 2.0 * ln_gamma(z + 1.0) - z.ln() - 2.0 * ln_gamma(z + 0.5);
  }
  let w = 1.0 / (z * z);
  (0.25 + w * (-1.0 / 96.0 + w * (1.0 / 320.0 + w * (-17.0 / 7168.0 + w * 31.0 / 9216.0)))) / z
}

/// `A = c₀(sinh t_y sech t − ρ tanh t)`, `c₀ = √((ν + 1)/(1 − ρ²))`: `y` standardized in its law given `X = √ν sinh t`,
/// `t_{ν+1}`. Read from `|t|`, its sign and the exact `|t_y| − |t|`; the centre crossing by `expm1`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Conditional {
  pub(crate) ln_c0: f64,
  rho: f64,
  ln_abs_rho: f64,
  sign_y: f64,
  /// `ln(1 − e^{−2|t_y|})`.
  ln_y: f64,
  /// `expm1(2|t_y|)`.
  e2y: f64,
}

impl Conditional {
  pub(crate) fn new(rho: f64, nu: f64, y: &Feature) -> Self {
    Self {
      ln_c0: 0.5 * ((nu + 1.0).ln() - ((1.0 - rho) * (1.0 + rho)).ln()),
      rho,
      ln_abs_rho: rho.abs().ln(),
      sign_y: y.sign,
      ln_y: (-(-2.0 * y.abs_t).exp_m1()).ln(),
      e2y: (2.0 * y.abs_t).exp_m1(),
    }
  }

  /// `(sign A, ln|A|)` at a point of sign `sign_t`, `|t| = abs_t`, `|t_y| − |t| = gap`.
  pub(crate) fn argument(&self, sign_t: f64, abs_t: f64, gap: f64) -> (f64, f64) {
    let ln_1p_e = (-2.0 * abs_t).exp().ln_1p();
    let ln_y = gap + self.ln_y - ln_1p_e;
    let ln_r = self.ln_abs_rho + (-(-2.0 * abs_t).exp_m1()).ln() - ln_1p_e;
    let sign_r = if sign_t == 0.0 {
      0.0
    } else {
      self.rho.signum() * sign_t
    };
    let (sign, ln) = if self.sign_y == 0.0 || ln_y == f64::NEG_INFINITY {
      (-sign_r, ln_r)
    } else if sign_r == 0.0 || ln_r == f64::NEG_INFINITY {
      (self.sign_y, ln_y)
    } else if self.sign_y == sign_r {
      // Near the crossing `ln(1 − e^{−2|t_y|}) − ln(1 − e^{−2|t|})` from the exact `gap` alone, where the two would cancel.
      let z = if gap.abs() < 1.0 {
        gap - self.ln_abs_rho - (-(2.0 * gap).exp_m1() / self.e2y).ln_1p()
      } else {
        ln_y - ln_r
      };
      (self.sign_y * z.signum(), ln_r + ln_abs_expm1(z))
    } else {
      (self.sign_y, ln_add_exp(ln_y, ln_r))
    };
    (sign, self.ln_c0 + ln)
  }
}

/// `ln|eᶻ − 1|`, `−∞` at 0.
fn ln_abs_expm1(z: f64) -> f64 {
  if z > 1.0 {
    z + (-(-z).exp()).ln_1p()
  } else if z < -1.0 {
    (-z.exp()).ln_1p()
  } else {
    z.exp_m1().abs().ln()
  }
}

/// `ln c = G − ½ ln(1 − ρ²) − ((ν + 2)/2) ln(1 + A²/(ν + 1)) − ln cosh t_x + (ν + 1) ln cosh t_y`, `A` at `t_x`, from the
/// features and the exact `|t_y| − |t_x|`: `(ν + 1)|t_y| − |t_x| = ν|t_y| + (|t_y| − |t_x|)` keeps huge scores' terms exact.
pub(crate) fn ln_density(fx: &Feature, fy: &Feature, gap: f64, rho: f64, nu: f64) -> f64 {
  if gap.is_infinite() {
    return f64::NEG_INFINITY;
  }
  let (_, ln_a) = Conditional::new(rho, nu, fy).argument(fx.sign, fx.abs_t, gap);
  let ln_1p_a2 = if ln_a < 300.0 {
    ((2.0 * ln_a).exp() / (nu + 1.0)).ln_1p()
  } else {
    2.0 * ln_a - (nu + 1.0).ln() + ((nu + 1.0) * (-2.0 * ln_a).exp()).ln_1p()
  };
  let cosh_terms = if fx.abs_t >= 1.0 && fy.abs_t >= 1.0 {
    let l = |a: f64| (-2.0 * a).exp().ln_1p();
    fy.nu_abs_t + gap - nu * LN_2 + (nu + 1.0) * l(fy.abs_t) - l(fx.abs_t)
  } else {
    (nu + 1.0) * ln_cosh(fy.abs_t) - ln_cosh(fx.abs_t)
  };
  ln_gamma_constant(nu) - 0.5 * ((1.0 - rho) * (1.0 + rho)).ln() - 0.5 * (nu + 2.0) * ln_1p_a2
    + cosh_terms
}
