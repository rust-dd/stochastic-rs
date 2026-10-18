//! The bivariate normal cdf to a relative `1e-12`: the conditional integral over the rotated `V`, of positive terms only,
//! by Gauss–Kronrod panels marched out of its maximum.
//! Reference: Genz, A. (2004), "Numerical computation of rectangular bivariate and trivariate normal and t probabilities", *Statistics and Computing* 14(3), 251-260, DOI 10.1023/B:STCO.0000035304.20635.31 (the conditional representation).

use stochastic_rs_distributions::special::norm_cdf;

use crate::bivariate::conditional::ln_norm_cdf;
use crate::bivariate::conditional::ln_one_minus_exp;
use crate::bivariate::integral::ln_integral_concave;
use crate::bivariate::integral::rule;

fn ln_pdf(x: f64) -> f64 {
  -0.5 * x * x - 0.5 * std::f64::consts::TAU.ln()
}

/// `ln(Φ(c + d/2) − Φ(c − d/2))`, `d > 0`, to a relative rounding: the density by Gauss–Legendre about `c` while
/// `d max(1, |c|) ≤ 1`, where the tails' logarithms would cancel, else those logarithms on the side of `0` both lie.
pub(crate) fn ln_normal_interval(c: f64, d: f64) -> f64 {
  if d.is_nan() || d <= 0.0 {
    return f64::NEG_INFINITY;
  }
  if d * c.abs().max(1.0) <= 1.0 {
    let sum = rule()
      .iter()
      .map(|&(x, w)| w * (-0.5 * c * d * x - 0.125 * d * d * x * x).exp())
      .sum::<f64>();
    return (0.5 * d).ln() + ln_pdf(c) + sum.ln();
  }
  let (a, b) = (c + 0.5 * d, c - 0.5 * d);
  if a <= 0.0 {
    let (ln_a, ln_b) = (ln_norm_cdf(a), ln_norm_cdf(b));
    ln_a + ln_one_minus_exp(ln_b - ln_a)
  } else if b >= 0.0 {
    let (ln_a, ln_b) = (ln_norm_cdf(-b), ln_norm_cdf(-a));
    ln_a + ln_one_minus_exp(ln_b - ln_a)
  } else {
    (-(norm_cdf(b) + norm_cdf(-a))).ln_1p()
  }
}

/// `Φ⁻¹(p) − Φ⁻¹(p − g)` from `x = Φ⁻¹(p)` and the exact gap `g`: Newton on `Φ(x) − Φ(x − D) = g`, so two near quantiles
/// differ to a relative rounding, which their separately rounded values lose.
pub(crate) fn quantile_gap(x: f64, g: f64) -> f64 {
  if g == 0.0 {
    return 0.0;
  }
  let mut d = g / ln_pdf(x).exp();
  for _ in 0..8 {
    let mass = ln_normal_interval(x - 0.5 * d, d.abs()).exp().copysign(d);
    let step = (g - mass) / ln_pdf(x - d).exp();
    d += step;
    if step.abs() <= 1e-16 * d.abs() {
      break;
    }
  }
  d
}

/// `Φ₂(h, k; ρ) = P(X ≤ h, Y ≤ k)`, `(X, Y)` standard normal with correlation `ρ`, to a relative `1e-12`: the conditional
/// integral over `V`, `X = (βU + αV)/2`, `sY = (βU − αV)/2`, `α, β = √(2(1 ∓ |ρ|))`, `s = sign ρ`, `U, V` independent.
/// It has no step as `|ρ| → 1`; `ρ ≥ 0` gives `∫ φ(v)Φ(min((2h − αv)/β, (2k + αv)/β)) dv`, `ρ < 0` the interval
/// `∫_{v ≤ (h + k)/α} φ(v)(Φ((2h − αv)/β) − Φ((αv − 2k)/β)) dv`, both of positive terms only.
pub(crate) fn bivariate_normal_cdf(h: f64, k: f64, sum: f64, rho: f64) -> f64 {
  if h == f64::NEG_INFINITY || k == f64::NEG_INFINITY {
    return 0.0;
  }
  if h == f64::INFINITY {
    return norm_cdf(k);
  }
  if k == f64::INFINITY {
    return norm_cdf(h);
  }
  let r = rho.abs();
  let (alpha, beta) = ((2.0 * (1.0 - r)).sqrt(), (2.0 * (1.0 + r)).sqrt());
  let ln_c = if rho >= 0.0 {
    let kink = (h - k) / alpha;
    ln_integral_concave(
      |v| ln_pdf(v) + ln_norm_cdf(((2.0 * h - alpha * v) / beta).min((2.0 * k + alpha * v) / beta)),
      f64::NEG_INFINITY,
      f64::INFINITY,
      0.0,
      &[kink],
    )
  } else {
    // The interval's centre is `(h − k)/β` at every `v`, its width `2(h + k − αv)/β`, from the given `h + k`.
    let (centre, end) = ((h - k) / beta, sum / alpha);
    ln_integral_concave(
      |v| ln_pdf(v) + ln_normal_interval(centre, 2.0 * (sum - alpha * v) / beta),
      f64::NEG_INFINITY,
      end,
      (end - 1.0).min(0.0),
      &[],
    )
  };
  ln_c.exp()
}
