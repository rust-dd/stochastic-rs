//! `K_ν(x)` for real order: `bessik` (Temme below `x = 2`, Steed's CF2 above), Debye past order 50,
//! DLMF 10.30 near 0; Temme's `γ₁`, `γ₂` from `ln Γ`'s series, which does not cancel at small `μ`:
//!
//! $$
//! \ln\Gamma(1+\mu)^{-1} = \gamma_E\mu - \sum_{k\ge2}(-1)^k\zeta(k)\mu^k/k = S_e + S_o,
//! $$
//! $$
//! \gamma_1 = \frac{1/\Gamma(1-\mu) - 1/\Gamma(1+\mu)}{2\mu} = -\frac{e^{S_e}\sinh S_o}{\mu},\qquad
//! \gamma_2 = \frac{1/\Gamma(1-\mu) + 1/\Gamma(1+\mu)}{2} = e^{S_e}\cosh S_o.
//! $$
//!
//! Reference: Temme, N.M. (1975), "On the numerical evaluation of the modified Bessel function of the third kind", *Journal of Computational Physics* 19(3), 324-337, DOI 10.1016/0021-9991(75)90082-0.
//! Reference: Thompson, I.J., Barnett, A.R. (1987), "Modified Bessel functions I_ν(z) and K_ν(z) of real order and complex argument, to selected accuracy", *Computer Physics Communications* 47(2-3), 245-257, DOI 10.1016/0010-4655(87)90111-1.
//! Reference: Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P. (2007), *Numerical Recipes* 3rd ed., Cambridge UP, §6.6 (`bessik`).
//! Reference: Olver, F. W. J. (1954), "The asymptotic expansion of Bessel functions of large order", Phil. Trans. R. Soc. A 247, 328–368, DOI 10.1098/rsta.1954.0021 (Debye's uniform expansion, DLMF 10.41.4, above order 50).
//! Reference: NIST Digital Library of Mathematical Functions, eq. 10.30.2–10.30.3 (`K_ν` as `x → 0`, below `x = 1e-307`), <https://dlmf.nist.gov/10.30>.
//! Reference: NIST Digital Library of Mathematical Functions, eq. 10.29.1 (the recurrence whose ratio form sums `ln K` where `K` overflows), <https://dlmf.nist.gov/10.29.1>.

use std::f64::consts::LN_2;
use std::f64::consts::PI;
use std::f64::consts::TAU;
use std::sync::OnceLock;

use super::bessel_i::Debye;
use super::bessel_i::UNIFORM_MIN_ORDER;
use super::bessel_i::debye;
use super::bessel_i::debye_sum;
use super::bessel_i::exp_times;
use super::ln_parts;

mod parts;

pub(crate) use parts::ln_bessel_ke_parts;
pub(crate) use parts::ln_half_gamma_parts;

const EPS: f64 = 1e-16;
const MAX_ITER: usize = 10_000;
const X_MIN: f64 = 2.0;
const X_TINY: f64 = 1e-307;
const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;
const ZETA_TERMS: usize = 64;

/// `2¹⁰²³`: from here Steed's `2(1 + x)` overflows, and to order 50 Hankel's first correction
/// `(4ν² − 1)/8x` is under `2⁻¹⁰¹⁰`, so `e^x K_ν(x)` is `√(π/2x)` to the last bit.
const HANKEL_FROM: f64 = 8.988_465_674_311_58e307;

const SQRT_HALF_PI: f64 = 1.253_314_137_315_500_3;

const HALF_LN_HALF_PI: f64 = 0.225_791_352_644_727_44;

/// ζ(k) for k = 2..=65 by direct summation (smallest terms first) plus an
/// Euler–Maclaurin tail, computed once.
fn zeta_table() -> &'static [f64; ZETA_TERMS] {
  static TABLE: OnceLock<[f64; ZETA_TERMS]> = OnceLock::new();
  TABLE.get_or_init(|| {
    let mut table = [0.0; ZETA_TERMS];
    let n = 2_000.0_f64;
    for (slot, k) in table.iter_mut().zip(2..) {
      let kf = k as f64;
      let mut sum = 0.0;
      for i in (1..=2_000u32).rev() {
        sum += (i as f64).powf(-kf);
      }
      *slot = sum + n.powf(1.0 - kf) / (kf - 1.0) - 0.5 * n.powf(-kf)
        + kf * n.powf(-kf - 1.0) / 12.0
        - kf * (kf + 1.0) * (kf + 2.0) * n.powf(-kf - 3.0) / 720.0;
    }
    table
  })
}

/// Temme's $(\gamma_1, \gamma_2)$ for $|\mu| \le 1/2$.
fn temme_gammas(mu: f64) -> (f64, f64) {
  let zeta = zeta_table();
  let mut odd = EULER_MASCHERONI * mu;
  let mut even = 0.0;
  let mut power = mu;
  for (slot, k) in zeta.iter().zip(2..) {
    power *= mu;
    let term = slot * power / k as f64;
    if k % 2 == 0 {
      even -= term;
    } else {
      odd += term;
    }
    if term.abs() < 1e-18 {
      break;
    }
  }
  let scale = even.exp();
  let gam1 = if mu.abs() < 1e-300 {
    -scale * EULER_MASCHERONI
  } else {
    -scale * odd.sinh() / mu
  };
  (gam1, scale * odd.cosh())
}

/// Exponentially scaled $e^{x} K_\nu(x)$ for real `nu`; NaN unless `x > 0` and `nu` is finite, 0 at `x = ∞`.
pub fn bessel_ke(nu: f64, x: f64) -> f64 {
  if !nu.is_finite() || x.is_nan() || x <= 0.0 {
    return f64::NAN;
  }
  if x == f64::INFINITY {
    return 0.0;
  }
  let nu = nu.abs();
  if x < X_TINY {
    return tiny_argument(nu, x, false);
  }
  if nu > UNIFORM_MIN_ORDER {
    return debye_k(nu, x, true);
  }
  if x >= HANKEL_FROM {
    return SQRT_HALF_PI / x.sqrt();
  }
  ke_pair(nu, x).0
}

/// $\ln(e^{x} K_\nu(x))$ for real `nu`, finite wherever the logarithm is representable; NaN unless
/// `x > 0` and `nu` is finite, `−∞` at `x = ∞`.
pub fn ln_bessel_ke(nu: f64, x: f64) -> f64 {
  if !nu.is_finite() || x.is_nan() || x <= 0.0 {
    return f64::NAN;
  }
  if x == f64::INFINITY {
    return f64::NEG_INFINITY;
  }
  let nu = nu.abs();
  if x < X_TINY {
    return tiny_argument(nu, x, true);
  }
  if nu > UNIFORM_MIN_ORDER {
    return ln_debye_k(nu, x);
  }
  if x >= HANKEL_FROM {
    return HALF_LN_HALF_PI - 0.5 * x.ln();
  }
  let ke = ke_pair(nu, x).0;
  if ke.is_finite() {
    ke.ln()
  } else {
    ln_recurrence(nu, x)
  }
}

/// Scaled $(e^x K_\nu(x), e^x K_{\nu+1}(x))$ for `nu ≥ 0` and `x > 0`; both infinite once the
/// upward recurrence overflows, since `K` grows with the order.
pub(super) fn ke_pair(nu: f64, x: f64) -> (f64, f64) {
  let nl = (nu + 0.5).floor() as usize;
  let mu = nu - nl as f64;
  let mu2 = mu * mu;
  let xi = 1.0 / x;
  let xi2 = 2.0 * xi;
  let (mut rkmu, mut rk1) = if x < X_MIN {
    temme_pair(x, mu, mu2, xi2)
  } else {
    steed_pair(x, mu, mu2, xi)
  };
  for i in 1..=nl {
    if !rk1.is_finite() {
      return (rk1, rk1);
    }
    let rktemp = (mu + i as f64) * xi2 * rk1 + rkmu;
    rkmu = rk1;
    rk1 = rktemp;
  }
  (rkmu, rk1)
}

/// $K_\nu(x)$ for real `nu`; NaN unless `x > 0` and `nu` is finite, 0 at `x = ∞`.
pub fn bessel_k(nu: f64, x: f64) -> f64 {
  let nu = nu.abs();
  if nu > UNIFORM_MIN_ORDER && nu.is_finite() && x.is_finite() && x >= X_TINY {
    return debye_k(nu, x, false);
  }
  bessel_ke(nu, x) * (-x).exp()
}

/// DLMF 10.41.4, `K_ν(νz) ~ √(π/2ν) e^{−νη} (1+z²)^{−1/4} Σ(−1)^k U_k(p)/ν^k`, out of line so the
/// small-order path keeps its size.
#[inline(never)]
fn debye_k(nu: f64, x: f64, scaled: bool) -> f64 {
  let (m, d) = debye_k_terms(nu, x);
  exp_times(if scaled { -d.eta_minus_z } else { -d.eta }, m)
}

/// `ln K_ν(x) + νη(x/ν) + ½ ln ν` by Debye, `½ ln(πp/2) + ln Σ(−1)^k U_k(p)/ν^k`, finite at
/// `x = 0`; `½ ln ν` stays apart so two orders' Stirling remainders meet exactly.
pub(crate) fn debye_shape(nu: f64, x: f64) -> f64 {
  debye_factor(nu, x).ln()
}

/// [`debye_shape`] in two parts, its logarithm libm-free, so a large `½ ln p` carries no rounding of its own.
pub(crate) fn debye_shape_parts(nu: f64, x: f64) -> (f64, f64) {
  ln_parts(debye_factor(nu, x), 0.0, 0)
}

fn debye_factor(nu: f64, x: f64) -> f64 {
  let p = debye(nu, x).p;
  PI * (p / TAU).sqrt() * debye_sum(p, nu, true)
}

/// The logarithm of [`debye_k`]'s scaled value, out of line like it.
#[inline(never)]
fn ln_debye_k(nu: f64, x: f64) -> f64 {
  let (m, d) = debye_k_terms(nu, x);
  m.ln() - d.eta_minus_z
}

/// Debye's exponents and the factor before `e^{−νη}`, its root split so that `2πν` cannot overflow at a finite order.
fn debye_k_terms(nu: f64, x: f64) -> (f64, Debye) {
  let d = debye(nu, x);
  (
    PI * ((d.p / TAU).sqrt() / nu.sqrt()) * debye_sum(d.p, nu, true),
    d,
  )
}

/// `ln(e^x K_ν(x))` for `ν ≤ 50` below `x = 2`, where `K` overflows: the recurrence on
/// `g = (x/2)K_{n+1}/K_n` keeps `g` near `n`, its logarithms summed apart from the large terms.
#[cold]
#[inline(never)]
fn ln_recurrence(nu: f64, x: f64) -> f64 {
  let nl = (nu + 0.5).floor() as usize;
  let mu = nu - nl as f64;
  let (sum, sum1) = temme_sums(x, mu, mu * mu);
  let mut g = sum1 / sum;
  let mut small = 0.0;
  for i in 1..=nl {
    small += g.ln();
    g = (mu + i as f64) + 0.25 * x * x / g;
  }
  small + (sum.ln() + x + nl as f64 * (LN_2 - x.ln()))
}

/// DLMF 10.30.2's `½Γ(ν)(2/x)^ν` in logs from `ν = ½` (error `O(x^{2ν})` below `ν = 1`, `O(x²)`
/// above), Temme's `f₀` below, as that term is not uniform in `ν → 0`; scaled is plain here.
#[cold]
#[inline(never)]
fn tiny_argument(nu: f64, x: f64, log: bool) -> f64 {
  let d = LN_2 - x.ln();
  if nu < 0.5 {
    let f0 = temme_head(nu, d).0;
    if log { f0.ln() } else { f0 }
  } else {
    let ln = super::ln_gamma(nu) - LN_2 + nu * d;
    if log { ln } else { ln.exp() }
  }
}

/// Temme's `f₀`, `K_μ(x)` to `O(x²)` for `|μ| ≤ ½`, and the `μ d`, `γ₁`, `γ₂` his series goes on with, from `d = ln(2/x)`.
fn temme_head(mu: f64, d: f64) -> (f64, f64, f64, f64) {
  let pimu = PI * mu;
  let fact = if pimu.abs() < EPS {
    1.0
  } else {
    pimu / pimu.sin()
  };
  let e = mu * d;
  let fact2 = if e.abs() < EPS { 1.0 } else { e.sinh() / e };
  let (gam1, gam2) = temme_gammas(mu);
  (fact * (gam1 * e.cosh() + gam2 * fact2 * d), e, gam1, gam2)
}

/// Temme's series: scaled $(K_\mu, K_{\mu+1})$ for `x < 2`, NaN where it does not converge.
fn temme_pair(x: f64, mu: f64, mu2: f64, xi2: f64) -> (f64, f64) {
  let (sum, sum1) = temme_sums(x, mu, mu2);
  let scale = x.exp();
  (sum * scale, sum1 * xi2 * scale)
}

/// Temme's series: $(K_\mu, (x/2) K_{\mu+1})$ for `x < 2`, NaN where it does not converge.
pub(super) fn temme_sums(x: f64, mu: f64, mu2: f64) -> (f64, f64) {
  temme_sums_at(x, mu, mu2, -(0.5 * x).ln())
}

/// [`temme_sums`] with its `ln(2/x)` given.
pub(super) fn temme_sums_at(x: f64, mu: f64, mu2: f64, ln_2_x: f64) -> (f64, f64) {
  let x2 = 0.5 * x;
  let (mut ff, e, gam1, gam2) = temme_head(mu, ln_2_x);
  let gampl = gam2 - mu * gam1;
  let gammi = gam2 + mu * gam1;
  let mut sum = ff;
  let e = e.exp();
  let mut p = 0.5 * e / gampl;
  let mut q = 0.5 / (e * gammi);
  let mut c = 1.0;
  let d = x2 * x2;
  let mut sum1 = p;
  for i in 1..=MAX_ITER {
    let fi = i as f64;
    ff = (fi * ff + p + q) / (fi * fi - mu2);
    c *= d / fi;
    p /= fi - mu;
    q /= fi + mu;
    let del = c * ff;
    sum += del;
    let del1 = c * (p - fi * ff);
    sum1 += del1;
    if del.abs() < sum.abs() * EPS {
      return (sum, sum1);
    }
  }
  (f64::NAN, f64::NAN)
}

/// Steed's CF2: scaled $(K_\mu, K_{\mu+1})$ for `x ≥ 2`, NaN where it does not converge.
pub(super) fn steed_pair(x: f64, mu: f64, mu2: f64, xi: f64) -> (f64, f64) {
  let mut b = 2.0 * (1.0 + x);
  let mut d = 1.0 / b;
  let mut h = d;
  let mut delh = d;
  let mut q1 = 0.0;
  let mut q2 = 1.0;
  let a1 = 0.25 - mu2;
  let mut q = a1;
  let mut c = a1;
  let mut a = -a1;
  let mut s = 1.0 + q * delh;
  let mut converged = false;
  for i in 2..=MAX_ITER {
    let fi = i as f64;
    a -= 2.0 * (fi - 1.0);
    c = -a * c / fi;
    let qnew = (q1 - b * q2) / a;
    q1 = q2;
    q2 = qnew;
    q += c * qnew;
    b += 2.0;
    d = 1.0 / (b + a * d);
    delh *= b * d - 1.0;
    h += delh;
    let dels = q * delh;
    s += dels;
    if (dels / s).abs() < EPS {
      converged = true;
      break;
    }
  }
  if !converged {
    return (f64::NAN, f64::NAN);
  }
  h *= a1;
  let rkmu = (std::f64::consts::PI / (2.0 * x)).sqrt() / s;
  let rk1 = rkmu * (mu + x + 0.5 - h) * xi;
  (rkmu, rk1)
}

#[cfg(test)]
mod tests;
