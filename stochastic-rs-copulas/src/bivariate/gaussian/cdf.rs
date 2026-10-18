//! The bivariate normal cdf to a relative `1e-12`: Owen's `T` form where its terms barely cancel, else the conditional
//! integral over the rotated `V`, of positive terms only, by Gauss–Kronrod panels marched out of its maximum.
//! Reference: Genz, A. (2004), "Numerical computation of rectangular bivariate and trivariate normal and t probabilities", *Statistics and Computing* 14(3), 251-260, DOI 10.1023/B:STCO.0000035304.20635.31 (the conditional representation).

use std::f64::consts::FRAC_1_SQRT_2;
use std::f64::consts::FRAC_2_SQRT_PI;
use std::f64::consts::TAU;

use owens_t::owens_t;
use stochastic_rs_distributions::special::erf;
use stochastic_rs_distributions::special::erfc;
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

/// `Φ⁻¹(p) − Φ⁻¹(p − g)` from `x ≈ Φ⁻¹(p)`, its residual `r = Φ(x) − p` and the exact gap `g`: Newton on `Φ(x) − Φ(x − D) =
/// g + r` from `start`, less `r/φ(x)`, so two near quantiles differ to a few roundings, which their rounded values lose.
pub(crate) fn quantile_gap(x: f64, p: f64, g: f64, start: f64) -> f64 {
  if g == 0.0 {
    return 0.0;
  }
  let r = if x <= 0.0 {
    upper_tail(-x) - p
  } else {
    (1.0 - p) - upper_tail(x)
  };
  let (target, mut d) = (g + r, start);
  for _ in 0..8 {
    let step = (target - normal_mass(x, d)) / pdf(x - d);
    d += step;
    if step.abs() <= 1e-16 * d.abs() {
      break;
    }
  }
  d - r / pdf(x)
}

/// `φ(x)`, `x²` split by one fma, so `e^{−x²/2}` keeps the relative digits its rounded square would cost.
fn pdf(x: f64) -> f64 {
  let square = x * x;
  let low = x.mul_add(x, -square);
  (-0.5 * square).exp() * (1.0 - 0.5 * low) * FRAC_1_SQRT_TAU
}

/// `1/√(2π)`.
const FRAC_1_SQRT_TAU: f64 = 0.398_942_280_401_432_7;

/// `Φ(s + e)`, `e` the rounding of the sum `s`, corrected to first order, by the tail on the side of 0 `s` lies.
fn phi_of_sum(s: f64, e: f64) -> f64 {
  if s <= 0.0 {
    upper_tail(-s) + e * pdf(s)
  } else {
    1.0 - (upper_tail(s) - e * pdf(s))
  }
}

/// `Φ(x) − Φ(x − d)` at exact `x`, `d` to a few roundings: `dφ(x)∫₀¹ e^{tdx − t²d²/2} dt` by Gauss–Legendre while
/// `|d| max(1, |x|) ≤ 1`, else the two tails on the side of 0 both ends lie, with `x − d` by TwoSum.
fn normal_mass(x: f64, d: f64) -> f64 {
  if d.abs() * x.abs().max(1.0) <= 1.0 {
    let sum = rule()
      .iter()
      .map(|&(t, w)| {
        let s = 0.5 * (1.0 + t);
        w * (s * d * x - 0.5 * s * s * d * d).exp()
      })
      .sum::<f64>();
    return 0.5 * d * pdf(x) * sum;
  }
  let z = x - d;
  let back = z - x;
  let e = (x - (z - back)) + (-d - back);
  if x <= 0.0 && z <= 0.0 {
    upper_tail(-x) - (upper_tail(-z) + e * pdf(z))
  } else if x >= 0.0 && z >= 0.0 {
    (upper_tail(z) - e * pdf(z)) - upper_tail(x)
  } else {
    phi_of_sum(x, 0.0) - phi_of_sum(z, e)
  }
}

/// `Φ₂(h, k; ρ) = P(X ≤ h, Y ≤ k)` to a relative `1e-12`: `owen` where it holds, else `conditional_integral`, which
/// reads `h + k` from `sum` only at `ρ < 0`.
pub(crate) fn bivariate_normal_cdf(h: f64, k: f64, rho: f64, sum: impl FnOnce() -> f64) -> f64 {
  if h == f64::NEG_INFINITY || k == f64::NEG_INFINITY {
    return 0.0;
  }
  if h == f64::INFINITY {
    return norm_cdf(k);
  }
  if k == f64::INFINITY {
    return norm_cdf(h);
  }
  owen(h, k, rho).unwrap_or_else(|| conditional_integral(h, k, rho, sum))
}

/// `Φ₂` as the integral over `V` of `X = (βU + αV)/2`, `sY = (βU − αV)/2`, `α, β = √(2(1 ∓ |ρ|))`, `s = sign ρ`, `U, V`
/// independent: positive terms only, and no step as `|ρ| → 1`.
fn conditional_integral(h: f64, k: f64, rho: f64, sum: impl FnOnce() -> f64) -> f64 {
  let r = rho.abs();
  let (alpha, beta) = ((2.0 * (1.0 - r)).sqrt(), (2.0 * (1.0 + r)).sqrt());
  let ln_c = if rho >= 0.0 {
    // `∫ φ(v)Φ(min((2h − αv)/β, (2k + αv)/β)) dv`, kinked where the two arguments meet.
    let kink = (h - k) / alpha;
    ln_integral_concave(
      |v| ln_pdf(v) + ln_norm_cdf(((2.0 * h - alpha * v) / beta).min((2.0 * k + alpha * v) / beta)),
      f64::NEG_INFINITY,
      f64::INFINITY,
      0.0,
      &[kink],
    )
  } else {
    // `∫_{v ≤ (h + k)/α} φ(v)(Φ((2h − αv)/β) − Φ((αv − 2k)/β)) dv`: the interval's centre is `(h − k)/β` at every `v`, its
    // width `2(h + k − αv)/β`, from the given `h + k`.
    let sum = sum();
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

/// What a closed form may leave of `C`, relative: Owen's `Φ₂` at `S ≤ 14C`, its terms within `188ε`, reaches `5.8e-13`.
pub(crate) const CLOSED_FORM_BUDGET: f64 = 6e-13;

/// Owen's `Φ₂` is taken where its terms' absolute sum `S` is within this of `C`: Patefield & Tandy's `T`, measured to
/// `179ε` at `a ≤ 1`, `h ≤ 38` (eq. (11) past `a = 1`), the tails `Φ` to `2ε` and the sums to `7ε` keep each within `188ε`.
/// Reference: Patefield, M., Tandy, D. (2000), "Fast and accurate calculation of Owen's T-function", *Journal of Statistical Software* 5(5), 1-25, eqs. (2), (10)-(11).
const OWEN_SPREAD: f64 = 14.0;

/// The low half of `1/√2`, `1/√2 − FRAC_1_SQRT_2` to 17 digits.
const FRAC_1_SQRT_2_LO: f64 = -4.833_646_656_726_457e-17;

/// `Φ(−x)`, `x ≥ 0`, with the rounding of `x/√2` corrected to first order, which `erfc` would amplify `x²`-fold.
fn upper_tail(x: f64) -> f64 {
  let z = x * FRAC_1_SQRT_2;
  let lo = x.mul_add(FRAC_1_SQRT_2, -z) + x * FRAC_1_SQRT_2_LO;
  0.5 * erfc(z) - lo * (-z * z).exp() * 0.5 * FRAC_2_SQRT_PI
}

/// Owen's `Q(x, a) = ½Φ(−x) − T(x, a)`, `a = num/(xs)`, `x ≥ 0` (0 at `x = 0`), past `|a| = 1` by eq. (2) in `T(|a|x, 1/|a|)`
/// so `T` sees `a ≤ 1` only; `spread` gathers its terms' absolute values.
fn owen_q(x: f64, num: f64, s: f64, spread: &mut f64) -> f64 {
  if x == 0.0 {
    return 0.0;
  }
  let den = x * s;
  if num.abs() <= den {
    let (half, t) = (0.5 * upper_tail(x), owens_t(x, num / den));
    *spread += half + t.abs();
    return half - t;
  }
  let w = num.abs() / s;
  let (t, product) = (
    owens_t(w, den / num.abs()),
    0.5 * erf(x * FRAC_1_SQRT_2) * upper_tail(w),
  );
  if num > 0.0 {
    *spread += t + product;
    t - product
  } else {
    let tail = upper_tail(x);
    *spread += tail + product + t;
    tail + product - t
  }
}

/// `Φ₂(h, k; ρ)` by Owen's `P(X > x, Y > y) = Q(x, a_x) + Q(y, a_y)` (P&T eq. (10)) in the orthant of `(h, k)`, `None`
/// where its spread passes `OWEN_SPREAD` or `C < 1e-290`, below which a term short of the normal range could move it.
fn owen(h: f64, k: f64, rho: f64) -> Option<f64> {
  let s = ((1.0 - rho) * (1.0 + rho)).sqrt();
  let (a, b) = ((-rho).mul_add(h, k), (-rho).mul_add(k, h));
  let (x, y) = (h.abs(), k.abs());
  let sign = |z: f64| if z > 0.0 { 1.0 } else { -1.0 };
  let mut spread = 0.0;
  let orthant = if x == 0.0 && y == 0.0 {
    let t = rho.asin() / TAU;
    spread += 0.25 + t.abs();
    0.25 + t
  } else {
    owen_q(x, sign(k) * a, s, &mut spread) + owen_q(y, sign(h) * b, s, &mut spread)
  };
  let c = match (h > 0.0, k > 0.0) {
    (false, false) => orthant,
    (true, false) => {
      let tail = upper_tail(y);
      spread += tail;
      tail - orthant
    }
    (false, true) => {
      let tail = upper_tail(x);
      spread += tail;
      tail - orthant
    }
    (true, true) => {
      let (tail_h, tail_k) = (upper_tail(x), upper_tail(y));
      spread += 1.0 + tail_h + tail_k;
      1.0 - tail_h - tail_k + orthant
    }
  };
  (c >= 1e-290 && spread <= OWEN_SPREAD * c).then_some(c)
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Where Owen's form is taken it agrees with the conditional integral to the closed forms' budget, and the bench's 16
  /// queries at `ρ = 0.5` all take it.
  #[test]
  fn owen_agrees_with_the_integral_where_taken() {
    let scores = [
      -37.0, -20.0, -8.0, -3.0, -1.0, -0.3, 0.0, 0.4, 1.0, 2.5, 6.0, 9.0,
    ];
    let mut taken = 0;
    for rho in [-0.999999, -0.9, -0.5, -0.1, 0.2, 0.5, 0.9, 0.999999] {
      for &h in &scores {
        for &k in &scores {
          if let Some(c) = owen(h, k, rho) {
            taken += 1;
            let want = conditional_integral(h, k, rho, || h + k);
            assert!(
              (c - want).abs() <= CLOSED_FORM_BUDGET * want + 1e-15 * want,
              "Φ₂({h}, {k}; {rho}) = {c:e}, integral {want:e}"
            );
          }
        }
      }
    }
    assert!(taken > 400, "{taken} cells took Owen's form");
    for i in 0..16 {
      let (u, v) = (
        0.1 + 0.05 * (i % 16) as f64,
        0.1 + 0.05 * ((i + 3) % 16) as f64,
      );
      let (h, k) = (
        crate::bivariate::conditional::normal_quantile(u),
        crate::bivariate::conditional::normal_quantile(v),
      );
      assert!(owen(h, k, 0.5).is_some(), "bench query ({u}, {v})");
    }
  }
}
