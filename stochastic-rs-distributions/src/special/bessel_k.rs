//! Modified Bessel function of the second kind `K_ν(x)` for real order.
//!
//! Numerical Recipes' `bessik` (3rd ed., §6.6) to order 50: Temme's series below `x = 2` or Steed's CF2 at `|μ| ≤ ½`,
//! then the upward recurrence; Debye's uniform expansion above order 50, the small-argument forms below `x = 1e-307`.
//!
//! The reciprocal-gamma pieces of Temme's method,
//!
//! $$
//! \gamma_1 = \frac{1/\Gamma(1-\mu) - 1/\Gamma(1+\mu)}{2\mu},\qquad
//! \gamma_2 = \frac{1/\Gamma(1-\mu) + 1/\Gamma(1+\mu)}{2},
//! $$
//!
//! come from $\log 1/\Gamma(1+x) = \gamma_E x - \sum_{k\ge2}(-1)^k\zeta(k)x^k/k$,
//! so $\gamma_1 = -e^{S_e}\sinh(S_o)/\mu$ and $\gamma_2 = e^{S_e}\cosh(S_o)$
//! with the odd and even partial sums $S_o$, $S_e$ — no Chebyshev fit and
//! no cancellation at small $\mu$.
//!
//! References:
//! - Temme, N.M. (1975), "On the numerical evaluation of the modified
//!   Bessel function of the third kind", *Journal of Computational
//!   Physics* 19(3), 324-337. DOI: 10.1016/0021-9991(75)90082-0
//! - Thompson, I.J., Barnett, A.R. (1987), "Modified Bessel functions
//!   I_ν(z) and K_ν(z) of real order and complex argument, to selected
//!   accuracy", *Computer Physics Communications* 47(2-3), 245-257.
//!   DOI: 10.1016/0010-4655(87)90111-1
//! - Press, Teukolsky, Vetterling, Flannery (2007), *Numerical Recipes*,
//!   3rd ed., §6.6.
//!
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

const EPS: f64 = 1e-16;
const MAX_ITER: usize = 10_000;
const X_MIN: f64 = 2.0;
const X_TINY: f64 = 1e-307;
const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;
const ZETA_TERMS: usize = 64;

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
  ke_pair(nu, x).0
}

/// $\ln(e^{x} K_\nu(x))$ for real `nu`, finite wherever the logarithm is representable; NaN unless `x > 0` and `nu` is
/// finite, `−∞` at `x = ∞`.
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
    let (m, d) = debye_k_terms(nu, x);
    return m.ln() - d.eta_minus_z;
  }
  let ke = ke_pair(nu, x).0;
  if ke.is_finite() {
    ke.ln()
  } else {
    ln_recurrence(nu, x)
  }
}

/// Scaled $(e^x K_\nu(x), e^x K_{\nu+1}(x))$ for `nu ≥ 0` and `x > 0`; both infinite once the upward recurrence
/// overflows, since `K` grows with the order.
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

/// DLMF 10.41.4, `K_ν(νz) ~ √(π/2ν) e^{−νη} (1+z²)^{−1/4} Σ(−1)^k U_k(p)/ν^k`, out of line so the small-order path keeps
/// its size.
#[inline(never)]
fn debye_k(nu: f64, x: f64, scaled: bool) -> f64 {
  let (m, d) = debye_k_terms(nu, x);
  exp_times(if scaled { -d.eta_minus_z } else { -d.eta }, m)
}

/// Debye's exponents and the factor before `e^{−νη}`, its root split so that `2πν` cannot overflow at a finite order.
fn debye_k_terms(nu: f64, x: f64) -> (f64, Debye) {
  let d = debye(nu, x);
  (
    PI * ((d.p / TAU).sqrt() / nu.sqrt()) * debye_sum(d.p, nu, true),
    d,
  )
}

/// `ln(e^x K_ν(x))` for `ν ≤ 50` below `x = 2`, where alone `K` overflows: the recurrence on `g = (x/2)K_{n+1}/K_n` keeps
/// `g` near `n`, and its logarithms are summed apart from the large terms they would otherwise round against.
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

/// DLMF 10.30.2's `½Γ(ν)(2/x)^ν` in log space from `ν = ½`, off by `O(x^{2ν})` below `ν = 1` and `O(x²)` above; Temme's `f₀`
/// below `ν = ½`, where that term is not uniform as `ν → 0`. `e^x = 1` here, so scaled and plain agree.
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
fn temme_sums(x: f64, mu: f64, mu2: f64) -> (f64, f64) {
  let x2 = 0.5 * x;
  let (mut ff, e, gam1, gam2) = temme_head(mu, -x2.ln());
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
fn steed_pair(x: f64, mu: f64, mu2: f64, xi: f64) -> (f64, f64) {
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
mod tests {
  use super::*;

  fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs()
  }

  /// `scipy.special.kv` / `kve` on a grid spanning both branches, integer
  /// and fractional orders, and the upward recurrence.
  #[test]
  fn matches_scipy_kv() {
    let cases = [
      (0.0, 0.01, 4.721_244_730_161_095, 4.768_694_028_544_461),
      (0.0, 1.0, 0.421_024_438_240_708_34, 1.144_463_079_806_894_9),
      (0.5, 0.3, 1.695_161_056_339_283_4, 2.288_228_082_159_423),
      (
        0.5,
        5.0,
        0.003_776_613_374_642_881_7,
        0.560_499_121_639_792_8,
      ),
      (1.0, 2.0, 0.139_865_881_816_522_46, 1.033_476_847_068_688_8),
      (1.5, 0.001, 39_633.253_172_629_775, 39_672.906_249_036_18),
      (2.3, 0.5, 13.509_653_881_303_644, 22.273_653_713_901_86),
      (2.3, 3.0, 0.073_627_459_986_590_29, 1.478_847_066_121_181_9),
      (
        3.7,
        10.0,
        3.397_938_590_173_589_4e-5,
        0.748_445_781_293_123_3,
      ),
      (0.25, 1.9, 0.130_600_563_447_076_52, 0.873_181_581_309_349_9),
      (7.5, 0.7, 2_412_246.300_819_027_2, 4_857_667.519_359_958),
      (
        7.5,
        25.0,
        1.036_599_319_047_804_4e-11,
        0.746_402_296_212_398_8,
      ),
      (
        12.0,
        60.0,
        4.630_938_063_529_372e-27,
        0.528_856_549_032_927_8,
      ),
      (4.0, 1e-4, 4.799_999_996e17, 4.800_480_020_000_401e17),
      (0.99, 2.01, 0.137_484_978_072_787_1, 1.026_094_021_846_430_7),
      (1.0, 1.9999, 0.139_884_265_831_690_97, 1.033_509_331_487_225),
      (0.5, 2.0, 0.119_937_771_968_061_45, 0.886_226_925_452_757_9),
    ];
    for (nu, x, kv, kve) in cases {
      assert!(
        close(bessel_k(nu, x), kv, 1e-13),
        "K_{nu}({x}) = {}",
        bessel_k(nu, x)
      );
      assert!(
        close(bessel_ke(nu, x), kve, 1e-13),
        "Ke_{nu}({x}) = {}",
        bessel_ke(nu, x)
      );
    }
    // The scaled form stays O(1) where the plain one is deep in the
    // subnormal range.
    assert!(close(
      bessel_ke(0.1, 700.0),
      0.047_362_707_517_186_83,
      1e-13
    ));
    assert!(bessel_k(0.1, 700.0) < 1e-300);
  }

  /// Integer orders agree with the Cephes `K₀` / `K₁` already in the crate.
  #[test]
  fn integer_orders_match_cephes() {
    for x in [0.05, 0.5, 1.5, 2.5, 7.0, 30.0] {
      assert!(close(bessel_k(0.0, x), super::super::bessel_k0(x), 1e-13));
      assert!(close(bessel_k(1.0, x), super::super::bessel_k1(x), 1e-13));
    }
  }

  /// $K_{1/2}(x) = \sqrt{\pi/(2x)}\,e^{-x}$ exactly, and the order symmetry.
  #[test]
  fn half_order_closed_form_and_symmetry() {
    for x in [0.1, 1.0, 3.0, 12.0] {
      let want = (std::f64::consts::PI / (2.0 * x)).sqrt() * (-x).exp();
      assert!(close(bessel_k(0.5, x), want, 1e-14));
      assert_eq!(bessel_k(-2.3, x), bessel_k(2.3, x));
    }
  }

  #[test]
  fn k_nu_is_nan_outside_its_domain() {
    assert!(bessel_ke(0.5, 0.0).is_nan());
    assert!(bessel_k(0.5, -2.0).is_nan());
    assert!(bessel_k(0.5, f64::NAN).is_nan());
  }

  #[test]
  fn k_nu_is_nan_for_a_non_finite_order_and_zero_at_infinity() {
    assert!(bessel_ke(f64::NAN, 1.0).is_nan());
    assert!(bessel_k(f64::INFINITY, 1.0).is_nan());
    assert!(bessel_k(f64::NEG_INFINITY, 3.0).is_nan());
    assert_eq!(bessel_k(0.5, f64::INFINITY), 0.0);
    assert_eq!(bessel_ke(2.5, f64::INFINITY), 0.0);
  }

  /// mpmath at the f64 arguments: below `x = 1e-307` the small-argument forms (DLMF 10.30.2–3; Temme's `f₀` below
  /// `ν = ½`) in log space; `(0.5, 1e-300)` is the Temme path.
  #[test]
  fn k_nu_at_subnormal_arguments() {
    assert!(close(
      bessel_ke(0.5, 1e-323),
      3.987_058_540_015_018e161,
      1e-13
    ));
    assert!(close(
      bessel_ke(0.5, 5e-324),
      5.638_552_261_264_71e161,
      1e-13
    ));
    assert!(close(bessel_ke(0.0, 1e-310), 713.917_310_343_812_6, 1e-13));
    assert!(close(bessel_ke(1e-3, 1e-310), 776.126_946_984_750_5, 1e-13));
    assert!(close(bessel_ke(1e-9, 1e-310), 713.917_310_343_873_2, 1e-13));
    assert!(close(
      bessel_ke(0.25, 5e-324),
      1.445_980_426_164_288_5e81,
      1e-13
    ));
    assert!(close(
      bessel_ke(0.4999, 1e-308),
      1.167_656_551_441_422_3e154,
      1e-13
    ));
    assert!(close(
      bessel_ke(0.5001, 1e-308),
      1.345_255_505_457_928e154,
      1e-13
    ));
    assert!(close(
      bessel_ke(0.5, 1e-300),
      1.253_314_137_315_500_2e150,
      1e-13
    ));
    assert_eq!(bessel_ke(2.5, 1e-310), f64::INFINITY);
    assert_eq!(bessel_k(0.5, 1e-323), bessel_ke(0.5, 1e-323));
  }

  /// mpmath `besselk` above order 50, where `K` is Debye's expansion; each side of the switch against its own value, and
  /// orders and arguments past `1e17` answer `∞` or `0` without a recurrence.
  #[test]
  fn k_nu_at_large_orders() {
    assert!(close(bessel_k(60.0, 1.0), 7.960_735_226_220_905e97, 1e-12));
    assert!(close(
      bessel_ke(60.0, 1.0),
      2.163_952_190_661_009_3e98,
      1e-12
    ));
    assert!(close(bessel_ke(60.0, 100.0), 4_667_765.644_383_589, 1e-12));
    assert!(close(
      bessel_ke(200.0, 150.0),
      7.832_615_732_173_226e50,
      1e-12
    ));
    assert!(close(
      bessel_ke(1000.0, 1000.0),
      2.557_507_414_201_933_5e201,
      1e-12
    ));
    assert!(close(
      bessel_k(1000.0, 1000.0),
      1.298_180_251_466_701e-233,
      1e-12
    ));
    assert_eq!(bessel_k(2000.0, 10.0), f64::INFINITY);
    assert!(close(
      bessel_ke(50.0 + 1e-9, 7.0),
      1.627_200_869_285_169_9e38,
      1e-12
    ));
    assert!(close(
      bessel_ke(50.0 - 1e-9, 7.0),
      1.627_200_860_647_077e38,
      1e-12
    ));
    assert_eq!(bessel_ke(1e17, 1.5), f64::INFINITY);
    assert_eq!(bessel_ke(1e300, 1.5), f64::INFINITY);
    assert_eq!(bessel_ke(1e18, 1e18), f64::INFINITY);
    assert_eq!(bessel_k(1e18, 1e18), 0.0);
  }

  /// mpmath at 120 and 240 digits on the two floats where `e^x K_50(x)` overflows: the recurrence side and the `ke.ln()`
  /// side each within an ulp of the logarithm.
  #[test]
  fn ln_bessel_ke_is_continuous_where_ke_overflows() {
    let (rec, direct) = (2.429_950_310_934_274e-5, 2.429_950_310_934_274_3e-5);
    assert_eq!(bessel_ke(50.0, rec), f64::INFINITY);
    assert!(bessel_ke(50.0, direct).is_finite());
    for x in [rec, direct] {
      let got = ln_bessel_ke(50.0, x);
      assert!(
        (got - 709.782_712_893_384).abs() <= 1.2e-13,
        "ln_bessel_ke(50, {x}) = {got}"
      );
    }
  }

  /// Past `2πν`'s overflow at `ν = 2.86e307` an order still answers: `∞` where `K` overflows, the logarithm where it is
  /// representable (against Debye's leading terms in mpmath).
  #[test]
  fn k_nu_past_the_overflow_of_two_pi_nu() {
    assert_eq!(bessel_ke(2.9e307, 1.5), f64::INFINITY);
    assert_eq!(bessel_k(2.9e307, 1.5), f64::INFINITY);
    assert_eq!(ln_bessel_ke(2.9e307, 1.5), f64::INFINITY);
    assert!(close(
      ln_bessel_ke(2.9e307, 1e305),
      1.556_277_287_959_050_6e308,
      1e-15
    ));
    assert!(close(
      ln_bessel_ke(1e300, 1.5),
      6.900_632_099_706_655e302,
      1e-15
    ));
  }

  /// mpmath `ln(e^x K_ν(x))` where `K` overflows: Debye's form above order 50, the logarithmic recurrence below it, the
  /// small-argument forms below `1e-307`; elsewhere the logarithm of [`bessel_ke`].
  #[test]
  fn ln_bessel_ke_is_finite_where_k_overflows() {
    for (nu, x, want) in [
      (60.0, 1e-3, 639.895_829_249_177_2),
      (60.0, 1.0, 226.425_275_381_116_45),
      (60.0, 50.0, 30.902_637_246_772_64),
      (172.0, 1e-3, 2_018.377_801_661_506_3),
      (172.0, 1.0, 831.241_431_696_023),
      (172.0, 50.0, 203.758_167_641_667_34),
      (1000.0, 1e-3, 13_505.430_735_570_453),
      (1000.0, 1.0, 6_598.674_206_338_348),
      (1000.0, 50.0, 2_735.026_021_466_822_4),
      (30.0, 1e-20, 1_472.909_362_999_833_8),
      (10.0, 1e-35, 824.944_934_653_037),
      (2.0, 1e-200, 921.727_184_378_178_2),
      (50.0, 3e-307, 35_468.280_518_818_98),
      (0.5, 5e-324, 372.445_827_313_335_37),
      (2.5, 1e-310, 1_785.827_850_711_698_1),
      (0.0, 1e-310, 6.570_767_143_789_475_5),
    ] {
      let got = ln_bessel_ke(nu, x);
      assert!(close(got, want, 1e-12), "ln_bessel_ke({nu}, {x}) = {got}");
    }
    for (nu, x) in [(0.5, 2.0), (12.0, 0.7), (7.5, 25.0)] {
      assert_eq!(ln_bessel_ke(nu, x), bessel_ke(nu, x).ln());
    }
    assert!(ln_bessel_ke(f64::NAN, 1.0).is_nan() && ln_bessel_ke(1.0, 0.0).is_nan());
    assert_eq!(ln_bessel_ke(3.0, f64::INFINITY), f64::NEG_INFINITY);
  }
}
