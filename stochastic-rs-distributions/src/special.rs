//! Special functions behind the closed-form `DistributionExt` impls: Lanczos `gamma`/`ln_gamma`,
//! Acklam's `ndtri`, `libm` `erf`/`erfc`; Bessel functions in [`bessel`], [`bessel_i`](mod@bessel_i), [`bessel_k`](mod@bessel_k).

pub mod bessel;
pub mod bessel_i;
pub mod bessel_k;

pub use bessel::bessel_i0;
pub use bessel::bessel_i1;
pub use bessel::bessel_k0;
pub use bessel::bessel_k1;
pub use bessel::bessel_k1e;
pub use bessel_i::bessel_i;
pub use bessel_i::bessel_ie;
pub use bessel_i::ln_bessel_ie;
pub use bessel_k::bessel_k;
pub use bessel_k::bessel_ke;

const LANCZOS_G: f64 = 7.0;
const LANCZOS_C: [f64; 9] = [
  0.999_999_999_999_809_9,
  676.520_368_121_885_1,
  -1_259.139_216_722_402_8,
  771.323_428_777_653_1,
  -176.615_029_162_140_6,
  12.507_343_278_686_905,
  -0.138_571_095_265_720_12,
  9.984_369_578_019_572e-6,
  1.505_632_735_149_311_6e-7,
];

/// Logarithm of the gamma function, accurate to ~14 decimal digits.
/// +∞ at the poles, NaN where Γ(x) < 0.
///
/// Lanczos (1964, DOI 10.1137/0701008) with Godfrey's g = 7, n = 9 coefficients.
#[inline]
pub fn ln_gamma(x: f64) -> f64 {
  if x <= 0.0 && x.fract() == 0.0 {
    return f64::INFINITY;
  }
  if x < 0.5 {
    // Reflection: ln Γ(x) = ln(π / sin(πx)) − ln Γ(1−x)
    return (std::f64::consts::PI / sinpi(x)).ln() - ln_gamma(1.0 - x);
  }
  let z = x - 1.0;
  let mut a = LANCZOS_C[0];
  for (i, c) in LANCZOS_C.iter().enumerate().skip(1) {
    a += c / (z + i as f64);
  }
  let t = z + LANCZOS_G + 0.5;
  0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + a.ln()
}

/// Gamma function: Lanczos for `x ≥ 0.5`, Euler's reflection below; NaN at the
/// poles, the non-positive integers.
pub fn gamma(x: f64) -> f64 {
  if x <= 0.0 && x.fract() == 0.0 {
    return f64::NAN;
  }
  if x < 0.5 {
    std::f64::consts::PI / (sinpi(x) * gamma(1.0 - x))
  } else {
    ln_gamma(x).exp()
  }
}

/// `sin(πx)` with `x` reduced to `[-½, ½]` before scaling, so it keeps its
/// relative accuracy next to the integers.
pub(crate) fn sinpi(x: f64) -> f64 {
  let n = x.round();
  let s = (std::f64::consts::PI * (x - n)).sin();
  if n % 2.0 == 0.0 { s } else { -s }
}

/// Digamma function ψ(x) = Γ'(x)/Γ(x).
///
/// Recurrence (ψ(x) = ψ(x+1) − 1/x) lifts the argument above 6, then an
/// asymptotic expansion in 1/x.
///
/// Returns NaN at the poles, the non-positive integers.
pub fn digamma(x: f64) -> f64 {
  if x <= 0.0 && x.fract() == 0.0 {
    return f64::NAN;
  }
  // Reflection ψ(1−x) = ψ(x) + π cot(πx); since cot has period π, x − round(x) can stand in
  // for x and keeps cot accurate next to the poles.
  if x < 0.5 {
    let reduced = x - x.round();
    return digamma(1.0 - x)
      - std::f64::consts::PI * (std::f64::consts::PI * reduced).tan().recip();
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

/// Error function `erf(x)`, from the `libm` crate's port of fdlibm (the
/// implementation in musl and FreeBSD libc).
#[inline]
pub fn erf(x: f64) -> f64 {
  libm::erf(x)
}

/// Complementary error function `erfc(x) = 1 − erf(x)`, computed directly
/// rather than as the difference, so the upper tail keeps its relative
/// precision: `erfc(10)` is `2.09e-45`, where `1 − erf(10)` rounds to zero.
#[inline]
pub fn erfc(x: f64) -> f64 {
  libm::erfc(x)
}

/// Standard-normal quantile (inverse CDF).
///
/// P. J. Acklam, *An algorithm for computing the inverse normal cumulative
/// distribution function*, 2003: the rational approximation (absolute error
/// ≈ 1.15e-9) followed by the single Halley step on `Φ(x) − p` that Acklam
/// gives for refining it, which brings the result to full double precision.
/// The step runs on the lower half, where `Φ(x) − p` is a difference of small
/// numbers; an upper-half `p` is mirrored through `1 − p`, which is exact for
/// `p ≥ ½`.
///
/// Returns NaN for a `p` outside `[0, 1]`.
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
  if p > 0.5 {
    return -ndtri_lower(1.0 - p);
  }
  ndtri_lower(p)
}

/// [`ndtri`] on `(0, ½]`.
fn ndtri_lower(p: f64) -> f64 {
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

  let x = if p < 0.025 {
    let q = (-2.0 * p.ln()).sqrt();
    (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
      / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
  } else {
    let q = p - 0.5;
    let r = q * q;
    (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
      / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
  };

  let e = 0.5 * erfc(-x / std::f64::consts::SQRT_2) - p;
  let u = e * (2.0 * std::f64::consts::PI).sqrt() * (0.5 * x * x).exp();
  // Past p ≈ 1e-308 the exponential overflows; the approximation stands alone.
  if u.is_finite() {
    x - u / (1.0 + 0.5 * x * u)
  } else {
    x
  }
}

/// Standard-normal pdf φ(x) = (2π)^{-½} exp(−x²/2).
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
  (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard-normal cdf Φ(x) = ½ erfc(−x/√2), which keeps its relative
/// precision in the lower tail, where ½(1 + erf(x/√2)) cancels to zero.
#[inline]
pub fn norm_cdf(x: f64) -> f64 {
  0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Regularised lower incomplete gamma P(a, x) = γ(a,x)/Γ(a).
///
/// Numerical Recipes 3e §6.2: series for x < a+1, continued fraction otherwise.
///
/// Returns NaN for a negative `x` or a non-positive `a`.
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

/// Unregularised lower incomplete gamma γ(a, x) = ∫₀ˣ tᵃ⁻¹ e⁻ᵗ dt = P(a,x)·Γ(a).
#[inline]
pub fn gamma_li(a: f64, x: f64) -> f64 {
  gamma_p(a, x) * gamma(a)
}

/// Unregularised upper incomplete gamma Γ(a, x) = ∫ₓ^∞ tᵃ⁻¹ e⁻ᵗ dt = Q(a,x)·Γ(a).
#[inline]
pub fn gamma_ui(a: f64, x: f64) -> f64 {
  gamma_q(a, x) * gamma(a)
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
///
/// Returns NaN for an `x` outside `[0, 1]`.
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
  fn ln_gamma_is_infinite_at_every_pole() {
    for pole in [0.0, -0.0, -1.0, -2.0, -7.0] {
      assert_eq!(ln_gamma(pole), f64::INFINITY, "ln_gamma({pole})");
    }
  }

  /// mpmath at 60 digits; on (−1, 2), poles included, the error is a few ulp,
  /// above it grows with `ln Γ(x)`, the exponent the Lanczos form exponentiates.
  #[test]
  fn gamma_matches_high_precision_references() {
    for (x, want) in [
      (0.5, 1.772_453_850_905_516),
      (1.0, 1.0),
      (1.4, 0.887_263_817_503_075_3),
      (0.4, 2.218_159_543_757_688),
      (-0.4, -3.722_980_622_032_042_5),
      (-0.999, -1_000.424_196_681_275_8),
      (-0.999_999, -1_000_000.422_756_991_2),
      (1.0e-10, 9_999_999_999.422_785),
      (1.2, 0.918_168_742_399_760_7),
      (1.9, 0.961_765_831_907_387_4),
      (-2.5, -0.945_308_720_482_941_9),
    ] {
      let got = gamma(x);
      assert!(
        ((got - want) / want).abs() < 4e-15,
        "gamma({x}) = {got}, want {want}"
      );
    }
    for (x, want) in [
      (30.5, 4.822_696_933_490_909e31),
      (170.5, 5.562_092_414_56e305),
    ] {
      let got = gamma(x);
      assert!(
        ((got - want) / want).abs() < 1e-15 * want.ln(),
        "gamma({x}) = {got}, want {want}"
      );
    }
    for pole in [0.0, -0.0, -1.0, -50.0] {
      assert!(gamma(pole).is_nan(), "gamma({pole}) must be NaN");
    }
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

  /// mpmath at 60 digits next to the poles, where the reflection's `π cot(πx)` dominates `ψ`.
  #[test]
  fn digamma_keeps_its_accuracy_next_to_the_poles() {
    for (x, want) in [
      (-0.999_999, -999_999.577_184_264_3),
      (-0.999_999_999, -1_000_000_027.859_147_9),
      (-10.000_001, 1_000_002.352_497_794_8),
      (-29.999_999, -999_996.581_197_320_5),
    ] {
      let got = digamma(x);
      assert!(
        ((got - want) / want).abs() < 1e-15,
        "digamma({x}) = {got}, want {want}"
      );
    }
    for pole in [0.0, -1.0, -7.0] {
      assert!(digamma(pole).is_nan(), "digamma({pole}) must be NaN");
    }
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

  /// References from mpmath at 50 significant digits.
  #[test]
  fn erf_and_erfc_match_high_precision_references() {
    let rel = |a: f64, b: f64| ((a - b) / b).abs();
    assert!(rel(erf(0.5), 0.520_499_877_813_046_5) < 1e-15);
    assert!(rel(erf(-1.2), -0.910_313_978_229_635_3) < 1e-15);
    assert!(rel(erfc(0.5), 0.479_500_122_186_953_5) < 1e-15);
    assert!(rel(erfc(5.0), 1.537_459_794_428_035e-12) < 1e-14);
    assert!(rel(erfc(10.0), 2.088_487_583_762_545e-45) < 1e-14);
    assert!(rel(erfc(26.0), 5.663_192_408_856_143e-296) < 1e-13);
  }

  #[test]
  fn norm_cdf_keeps_the_lower_tail() {
    let rel = |a: f64, b: f64| ((a - b) / b).abs();
    assert!(rel(norm_cdf(-8.0), 6.220_960_574_271_784e-16) < 1e-14);
    assert!(rel(norm_cdf(-1.0), 0.158_655_253_931_457_05) < 1e-15);
    assert!(rel(norm_cdf(1.96), 0.975_002_104_851_779_5) < 1e-15);
    assert!(rel(norm_cdf(6.0), 0.999_999_999_013_412_3) < 1e-15);
    assert!(norm_cdf(-38.0) > 0.0);
  }

  #[test]
  fn ndtri_matches_high_precision_references() {
    for &(p, x) in &[
      (1e-300, -37.047_096_299_361_2),
      (1e-10, -6.361_340_902_404_057),
      (0.001, -3.090_232_306_167_813_6),
      (0.025, -1.959_963_984_540_054_3),
      (0.3, -0.524_400_512_708_040_8),
      (0.975, 1.959_963_984_540_053_8),
      (0.999, 3.090_232_306_167_813),
      (0.999_999_999_9, 6.361_340_889_697_422),
    ] {
      let got = ndtri(p);
      assert!(
        ((got - x) / x).abs() < 1e-14,
        "ndtri({p}) = {got}, want {x}"
      );
    }
    assert_eq!(ndtri(0.5), 0.0);
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
