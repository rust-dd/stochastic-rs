//! Special functions used by the closed-form `DistributionExt` impls.
//!
//! All implementations are self-contained — no dependency on `statrs` or any
//! other distribution / special-function crate. Algorithms are textbook
//! (Numerical Recipes 3e; Abramowitz & Stegun; Acklam 1998 for `ndtri`;
//! Lanczos for `ln_gamma`).

const LANCZOS_G: f64 = 7.0;
const LANCZOS_C: [f64; 9] = [
  0.999_999_999_999_809_93,
  676.520_368_121_885_1,
  -1_259.139_216_722_402_8,
  771.323_428_777_653_13,
  -176.615_029_162_140_59,
  12.507_343_278_686_905,
  -0.138_571_095_265_720_12,
  9.984_369_578_019_571_6e-6,
  1.505_632_735_149_311_6e-7,
];

/// Logarithm of the gamma function, accurate to ~14 decimal digits.
///
/// Lanczos approximation, g = 7, n = 9 (Press et al., Numerical Recipes 3e).
#[inline]
pub fn ln_gamma(x: f64) -> f64 {
  if x < 0.5 {
    // Reflection: ln Γ(x) = ln(π / sin(πx)) − ln Γ(1−x)
    return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
  }
  let z = x - 1.0;
  let mut a = LANCZOS_C[0];
  for (i, c) in LANCZOS_C.iter().enumerate().skip(1) {
    a += c / (z + i as f64);
  }
  let t = z + LANCZOS_G + 0.5;
  0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + a.ln()
}

/// Gamma function. Euler reflection for x < 0.5; Lanczos otherwise.
pub fn gamma(x: f64) -> f64 {
  if x <= 0.0 && x.fract() == 0.0 {
    return f64::NAN; // poles at non-positive integers
  }
  if x < 0.5 {
    // Γ(x) = π / (sin(πx) · Γ(1−x))
    std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma(1.0 - x))
  } else {
    ln_gamma(x).exp()
  }
}

/// Digamma function ψ(x) = Γ'(x)/Γ(x).
///
/// Recurrence (ψ(x) = ψ(x+1) − 1/x) lifts the argument above 6, then an
/// asymptotic expansion in 1/x.
pub fn digamma(x: f64) -> f64 {
  if x <= 0.0 && x.fract() == 0.0 {
    return f64::NAN;
  }
  // For x ≤ 0 use the reflection formula ψ(1−x) = ψ(x) + π cot(πx)
  if x < 0.5 {
    return digamma(1.0 - x) - std::f64::consts::PI * (std::f64::consts::PI * x).tan().recip();
  }
  let mut y = x;
  let mut sum = 0.0;
  while y < 6.0 {
    sum -= 1.0 / y;
    y += 1.0;
  }
  // Asymptotic expansion for y ≥ 6.
  let inv = 1.0 / y;
  let inv2 = inv * inv;
  sum + y.ln()
    - 0.5 * inv
    - inv2
      * (1.0 / 12.0
        - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0 - inv2 * 1.0 / 132.0))))
}

/// Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b).
#[inline]
pub fn beta(a: f64, b: f64) -> f64 {
  ln_beta(a, b).exp()
}

/// Logarithm of the beta function.
#[inline]
pub fn ln_beta(a: f64, b: f64) -> f64 {
  ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Error function `erf(x)`.
///
/// Cody (1969) rational approximation transformed to the form used by W. J.
/// Cody, *Rational Chebyshev approximations for the error function*,
/// Math. Comp. **23** (1969) 631–637. Maximum relative error ~1.5e-7.
/// Sufficient for distribution work here; for higher precision swap in
/// `libm::erf` later if desired.
pub fn erf(x: f64) -> f64 {
  // Abramowitz & Stegun 7.1.26 — relative error < 1.5e-7.
  let sign = if x < 0.0 { -1.0 } else { 1.0 };
  let ax = x.abs();
  let p = 0.327_591_1_f64;
  let a1 = 0.254_829_592_f64;
  let a2 = -0.284_496_736_f64;
  let a3 = 1.421_413_741_f64;
  let a4 = -1.453_152_027_f64;
  let a5 = 1.061_405_429_f64;
  let t = 1.0 / (1.0 + p * ax);
  let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
  sign * y
}

/// Complementary error function `erfc(x) = 1 − erf(x)`.
#[inline]
pub fn erfc(x: f64) -> f64 {
  1.0 - erf(x)
}

/// Standard-normal quantile (inverse CDF).
///
/// P. J. Acklam, *An algorithm for computing the inverse normal cumulative
/// distribution function*, 2003. Maximum absolute error ≈ 1.15e-9 over
/// `(0, 1)`; rebuilt without external lookup tables.
pub fn ndtri(p: f64) -> f64 {
  if !(0.0..=1.0).contains(&p) {
    return f64::NAN;
  }
  if p == 0.0 {
    return f64::NEG_INFINITY;
  }
  if p == 1.0 {
    return f64::INFINITY;
  }

  // Coefficients from Acklam (2003).
  const A: [f64; 6] = [
    -3.969_683_028_665_376_e1,
    2.209_460_984_245_205_e2,
    -2.759_285_104_469_687_e2,
    1.383_577_518_672_69_e2,
    -3.066_479_806_614_716_e1,
    2.506_628_277_459_239,
  ];
  const B: [f64; 5] = [
    -5.447_609_879_822_406_e1,
    1.615_858_368_580_409_e2,
    -1.556_989_798_598_866_e2,
    6.680_131_188_771_972_e1,
    -1.328_068_155_288_572_e1,
  ];
  const C: [f64; 6] = [
    -7.784_894_002_430_293_e-3,
    -3.223_964_580_411_365_e-1,
    -2.400_758_277_161_838,
    -2.549_732_539_343_734,
    4.374_664_141_464_968,
    2.938_163_982_698_783,
  ];
  const D: [f64; 4] = [
    7.784_695_709_041_462_e-3,
    3.224_671_290_700_398_e-1,
    2.445_134_137_142_996,
    3.754_408_661_907_416,
  ];

  let p_low = 0.025;
  let p_high = 1.0 - p_low;

  if p < p_low {
    let q = (-2.0 * p.ln()).sqrt();
    (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
      / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
  } else if p <= p_high {
    let q = p - 0.5;
    let r = q * q;
    (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
      / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
  } else {
    let q = (-2.0 * (1.0 - p).ln()).sqrt();
    -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
      / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
  }
}

/// Standard-normal pdf φ(x) = (2π)^{-½} exp(−x²/2).
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
  (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard-normal cdf Φ(x) = ½(1 + erf(x/√2)).
#[inline]
pub fn norm_cdf(x: f64) -> f64 {
  0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Regularised lower incomplete gamma P(a, x) = γ(a,x)/Γ(a).
///
/// Numerical Recipes 3e §6.2: series for x < a+1, continued fraction otherwise.
pub fn gamma_p(a: f64, x: f64) -> f64 {
  if x < 0.0 || a <= 0.0 {
    return f64::NAN;
  }
  if x == 0.0 {
    return 0.0;
  }
  if x < a + 1.0 {
    gser(a, x)
  } else {
    1.0 - gcf(a, x)
  }
}

/// Regularised upper incomplete gamma Q(a, x) = Γ(a,x)/Γ(a) = 1 − P(a,x).
#[inline]
pub fn gamma_q(a: f64, x: f64) -> f64 {
  1.0 - gamma_p(a, x)
}

fn gser(a: f64, x: f64) -> f64 {
  // Series: P(a,x) = x^a e^{-x} / Γ(a+1) · Σ_{n=0}^∞ x^n / Π_{k=0}^n (a+k)
  let gln = ln_gamma(a);
  let mut ap = a;
  let mut sum = 1.0 / a;
  let mut del = sum;
  for _ in 0..200 {
    ap += 1.0;
    del *= x / ap;
    sum += del;
    if del.abs() < sum.abs() * 1e-15 {
      return sum * (-x + a * x.ln() - gln).exp();
    }
  }
  sum * (-x + a * x.ln() - gln).exp()
}

fn gcf(a: f64, x: f64) -> f64 {
  // Continued fraction (Lentz's method) for Q(a,x).
  let gln = ln_gamma(a);
  let fpmin = 1e-300_f64;
  let mut b = x + 1.0 - a;
  let mut c = 1.0 / fpmin;
  let mut d = 1.0 / b;
  let mut h = d;
  for i in 1..=200 {
    let an = -(i as f64) * (i as f64 - a);
    b += 2.0;
    d = an * d + b;
    if d.abs() < fpmin {
      d = fpmin;
    }
    c = b + an / c;
    if c.abs() < fpmin {
      c = fpmin;
    }
    d = 1.0 / d;
    let del = d * c;
    h *= del;
    if (del - 1.0).abs() < 1e-15 {
      break;
    }
  }
  (-x + a * x.ln() - gln).exp() * h
}

/// Regularised incomplete beta `I_x(a, b) = B(x; a, b) / B(a, b)`.
///
/// Numerical Recipes 3e §6.4: continued fraction with Lentz's method, plus
/// the symmetry `I_x(a,b) = 1 − I_{1−x}(b,a)` for tail-side stability.
pub fn beta_i(a: f64, b: f64, x: f64) -> f64 {
  if !(0.0..=1.0).contains(&x) {
    return f64::NAN;
  }
  if x == 0.0 || x == 1.0 {
    return x;
  }
  let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
  if x < (a + 1.0) / (a + b + 2.0) {
    bt * betacf(a, b, x) / a
  } else {
    1.0 - bt * betacf(b, a, 1.0 - x) / b
  }
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
  let fpmin = 1e-300_f64;
  let qab = a + b;
  let qap = a + 1.0;
  let qam = a - 1.0;
  let mut c = 1.0;
  let mut d = 1.0 - qab * x / qap;
  if d.abs() < fpmin {
    d = fpmin;
  }
  d = 1.0 / d;
  let mut h = d;
  for m in 1..=200 {
    let m_f = m as f64;
    let m2 = 2.0 * m_f;
    let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if d.abs() < fpmin {
      d = fpmin;
    }
    c = 1.0 + aa / c;
    if c.abs() < fpmin {
      c = fpmin;
    }
    d = 1.0 / d;
    h *= d * c;
    let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if d.abs() < fpmin {
      d = fpmin;
    }
    c = 1.0 + aa / c;
    if c.abs() < fpmin {
      c = fpmin;
    }
    d = 1.0 / d;
    let del = d * c;
    h *= del;
    if (del - 1.0).abs() < 1e-15 {
      break;
    }
  }
  h
}

#[cfg(test)]
mod tests {
  use super::*;

  fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol || ((a - b) / b).abs() < tol
  }

  #[test]
  fn ln_gamma_known_values() {
    // Γ(½) = √π ⟹ ln Γ(½) = ½ ln π
    assert!(close(ln_gamma(0.5), 0.5 * std::f64::consts::PI.ln(), 1e-10));
    // Γ(n) = (n−1)! for positive integer n.
    assert!(close(ln_gamma(5.0), 24.0_f64.ln(), 1e-10));
    assert!(close(ln_gamma(10.0), 362880.0_f64.ln(), 1e-9));
  }

  #[test]
  fn digamma_known_values() {
    // ψ(1) = −γ_Euler ≈ −0.577215...
    assert!(close(digamma(1.0), -0.577_215_664_901_532_9, 1e-9));
    // ψ(½) = −2 ln 2 − γ ≈ −1.96351...
    assert!(close(
      digamma(0.5),
      -2.0_f64.ln() * 2.0 - 0.577_215_664_901_532_9,
      1e-9
    ));
  }

  #[test]
  fn erf_known_values() {
    assert!(close(erf(0.0), 0.0, 1e-9));
    assert!(close(erf(1.0), 0.842_700_792_949_715, 1e-6));
    assert!(close(erf(-1.0), -0.842_700_792_949_715, 1e-6));
  }

  #[test]
  fn ndtri_round_trip() {
    for &p in &[0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999] {
      let z = ndtri(p);
      let back = norm_cdf(z);
      assert!(close(back, p, 1e-7), "p={p}, z={z}, back={back}");
    }
  }

  #[test]
  fn gamma_p_known_values() {
    // P(1, x) = 1 − e^{−x}
    assert!(close(gamma_p(1.0, 1.0), 1.0 - (-1.0_f64).exp(), 1e-12));
    // P(½, x) = erf(√x)
    assert!(close(gamma_p(0.5, 1.0), erf(1.0), 1e-6));
  }

  #[test]
  fn beta_i_known_values() {
    // I_{0.5}(1, 1) = 0.5
    assert!(close(beta_i(1.0, 1.0, 0.5), 0.5, 1e-10));
    // I_x(a, a) symmetric: I_{0.7}(2, 2) and I_{0.3}(2,2) sum to 1.
    let a = beta_i(2.0, 2.0, 0.7);
    let b = beta_i(2.0, 2.0, 0.3);
    assert!(close(a + b, 1.0, 1e-10));
  }
}
