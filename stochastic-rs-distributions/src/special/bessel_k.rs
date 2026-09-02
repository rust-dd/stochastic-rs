//! Modified Bessel function of the second kind `K_ν(x)` for real order.
//!
//! Numerical Recipes' `bessik` scheme (3rd ed., §6.6): Temme's series for
//! `x ≤ 2` and Steed's continued fraction CF2 for `x > 2` give
//! `K_μ`, `K_{μ+1}` at the fractional part `|μ| ≤ 1/2` of the order, then
//! the stable upward recurrence `K_{ν+1} = K_{ν-1} + (2ν/x) K_ν` climbs to
//! `ν`. The reciprocal-gamma pieces of Temme's method,
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
//!   DOI: 10.1016/0010-4655(87)90107-8
//! - Press, Teukolsky, Vetterling, Flannery (2007), *Numerical Recipes*,
//!   3rd ed., §6.6.

use std::sync::OnceLock;

const EPS: f64 = 1e-16;
const MAX_ITER: usize = 10_000;
const X_MIN: f64 = 2.0;
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

/// Exponentially scaled $e^{x} K_\nu(x)$ for real `nu` and `x > 0`.
///
/// # Panics
///
/// If `x` is not positive.
pub fn bessel_ke(nu: f64, x: f64) -> f64 {
  assert!(x > 0.0, "bessel_ke needs x > 0, got {x}");
  let nu = nu.abs();
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
    let rktemp = (mu + i as f64) * xi2 * rk1 + rkmu;
    rkmu = rk1;
    rk1 = rktemp;
  }
  rkmu
}

/// $K_\nu(x)$ for real `nu` and `x > 0`.
pub fn bessel_k(nu: f64, x: f64) -> f64 {
  bessel_ke(nu, x) * (-x).exp()
}

/// Temme's series: scaled $(K_\mu, K_{\mu+1})$ for `x < 2`.
fn temme_pair(x: f64, mu: f64, mu2: f64, xi2: f64) -> (f64, f64) {
  let x2 = 0.5 * x;
  let pimu = std::f64::consts::PI * mu;
  let fact = if pimu.abs() < EPS {
    1.0
  } else {
    pimu / pimu.sin()
  };
  let d = -x2.ln();
  let e = mu * d;
  let fact2 = if e.abs() < EPS { 1.0 } else { e.sinh() / e };
  let (gam1, gam2) = temme_gammas(mu);
  let gampl = gam2 - mu * gam1;
  let gammi = gam2 + mu * gam1;
  let mut ff = fact * (gam1 * e.cosh() + gam2 * fact2 * d);
  let mut sum = ff;
  let e = e.exp();
  let mut p = 0.5 * e / gampl;
  let mut q = 0.5 / (e * gammi);
  let mut c = 1.0;
  let d = x2 * x2;
  let mut sum1 = p;
  let mut converged = false;
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
      converged = true;
      break;
    }
  }
  assert!(converged, "Temme series for K_nu did not converge");
  let scale = x.exp();
  (sum * scale, sum1 * xi2 * scale)
}

/// Steed's CF2: scaled $(K_\mu, K_{\mu+1})$ for `x ≥ 2`.
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
  assert!(
    converged,
    "Steed continued fraction for K_nu did not converge"
  );
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
  #[should_panic(expected = "bessel_ke needs x > 0")]
  fn rejects_non_positive_argument() {
    let _ = bessel_k(1.0, 0.0);
  }
}
