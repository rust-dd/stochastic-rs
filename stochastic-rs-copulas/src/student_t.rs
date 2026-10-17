//! The Student-t law every t copula and the t pair copula evaluate through: `SimdStudentT`'s cdf and quantile, the
//! log-density its `pdf` exponentiates, the t copula's log-density at t scores, and the one rule for `nu`.
//! Reference: Abramowitz, M., Stegun, I.A. (1964), "Handbook of Mathematical Functions", NBS AMS 55, 26.7.5 (the Cornish–Fisher expansion of the t quantile).

use std::f64::consts::LN_2;
use std::f64::consts::PI;

use stochastic_rs_distributions::DistributionExt;
use stochastic_rs_distributions::special::ln_gamma;
use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_distributions::studentt::SimdStudentT;

use crate::bivariate::conditional::ln_norm_cdf;
use crate::bivariate::conditional::mills;
use crate::error::CopulaError;

/// Above this `ν` the law is normal to rounding: the first term `(x³ + x)/(4ν)` of the expansion is below `2⁻⁵⁴` of `Φ`
/// wherever `Φ(x) ≥ 2⁻¹⁰⁷⁴`, `|x| ≤ 38.5`, as `38.5⁴/4 · 2⁵⁴ < 10²²`.
const NORMAL_ABOVE: f64 = 1e22;

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

/// A quantile `t` with `ln|t|`, `+∞` where it overflows, and in the tail `ln P(T ≤ −|t|)`, which stays finite there.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Score {
  pub(crate) value: f64,
  pub(crate) ln_abs: f64,
  ln_tail: f64,
}

impl Score {
  fn of(value: f64) -> Self {
    Self {
      value,
      ln_abs: value.abs().ln(),
      ln_tail: f64::NAN,
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
      let mut t = ndtri(p);
      if tail < f64::MIN_POSITIVE {
        let z = -t.abs();
        t = sign * -(z - (ln_norm_cdf(z) - tail.ln()) * mills(-z));
      }
      return Score::of(t);
    }
    if tail <= self.tail_level {
      let ln_tail = tail.ln();
      let ln_abs = (self.ln_scale - ln_tail) / self.nu;
      return Score {
        value: sign * ln_abs.exp(),
        ln_abs,
        ln_tail,
      };
    }
    Score::of(
      self
        .inner
        .quantile(p)
        .expect("the Student-t law has a quantile"),
    )
  }

  /// `ln(|a|/|b|)` of two of this law's quantiles: from their tail probabilities when both have one, so the ratio of two
  /// quantiles past `f64::MAX` stays exact.
  pub(crate) fn ln_ratio(&self, a: Score, b: Score) -> f64 {
    if a.ln_tail.is_nan() || b.ln_tail.is_nan() {
      a.ln_abs - b.ln_abs
    } else {
      (b.ln_tail - a.ln_tail) / self.nu
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
      return self.cdf(sign, t.ln_abs + ln_factor);
    }
    if t.ln_abs + ln_factor >= self.ln_tail_start {
      return self.tail(sign, t.ln_tail - self.nu * ln_factor);
    }
    self.cdf_at(sign * (t.ln_abs + ln_factor).exp())
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

/// The exponent `SimdStudentT::pdf` takes, so `ln_pdf(x, nu).exp()` is that pdf bit for bit.
pub(crate) fn ln_pdf(x: f64, nu: f64) -> f64 {
  let log_norm =
    ln_gamma(0.5 * (nu + 1.0)) - 0.5 * (nu * std::f64::consts::PI).ln() - ln_gamma(0.5 * nu);
  let log_kernel = -0.5 * (nu + 1.0) * (1.0 + x * x / nu).ln();
  log_norm + log_kernel
}

/// `log c` of the t copula at the scores `x = t_ν⁻¹(u)`, `y = t_ν⁻¹(v)`: `c = Γ((ν+2)/2) Γ(ν/2) / (Γ((ν+1)/2)² √(1−ρ²))`
/// `· (1 + (x² − 2ρxy + y²)/(ν(1−ρ²)))^{−(ν+2)/2} · ((1 + x²/ν)(1 + y²/ν))^{(ν+1)/2}`.
pub(crate) fn ln_copula_density(x: f64, y: f64, rho: f64, nu: f64) -> f64 {
  let one_minus_rho2 = 1.0 - rho * rho;
  let log_norm = ln_gamma(0.5 * (nu + 2.0)) + ln_gamma(0.5 * nu)
    - 2.0 * ln_gamma(0.5 * (nu + 1.0))
    - 0.5 * one_minus_rho2.ln();
  let log_joint =
    -0.5 * (nu + 2.0) * (1.0 + (x * x - 2.0 * rho * x * y + y * y) / (nu * one_minus_rho2)).ln();
  let log_marg = 0.5 * (nu + 1.0) * ((1.0 + x * x / nu).ln() + (1.0 + y * y / nu).ln());
  log_norm + log_joint + log_marg
}
