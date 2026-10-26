//! Special functions behind the closed-form `DistributionExt` impls: Lanczos `gamma`/`ln_gamma`, Acklam's `ndtri`,
//! `libm` `erf`/`erfc`, Mills' ratio, the regularised incomplete gamma and beta functions and the modified Bessel `I`, `K`.
//! Reference: NIST Digital Library of Mathematical Functions, eq. 7.9.3 (the continued fraction of `w(z)`), <https://dlmf.nist.gov/7.9.3>.
//! Reference: NIST Digital Library of Mathematical Functions, eq. 5.11.8 (Stirling's series for `ln Γ(z + h)`, at `h = ½` less `h = 0` the ratio of eq. 5.11.13), <https://dlmf.nist.gov/5.11.8>.
//! Reference: Acklam, P.J. (2003), "An algorithm for computing the inverse normal cumulative distribution function" (the rational approximation and Halley step of `ndtri`).

mod bessel;
mod bessel_i;
mod bessel_k;
mod erfc;
mod estimate;
mod incomplete;
mod k_ratio;
mod quotient;

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
pub use bessel_k::ln_bessel_ke;
pub(crate) use estimate::Estimate;
pub(crate) use estimate::exp_of_log_sum;
pub use incomplete::beta_i;
pub use incomplete::gamma_li;
pub use incomplete::gamma_p;
pub(crate) use incomplete::gamma_p_inv;
pub use incomplete::gamma_q;
pub(crate) use incomplete::gamma_q_inv;
pub use incomplete::gamma_ui;
pub use incomplete::ln_gamma_p;
pub use incomplete::ln_gamma_q;
pub(crate) use incomplete::student_tail;
pub(crate) use incomplete::student_xi;
pub(crate) use k_ratio::KDenominator;
pub(crate) use k_ratio::k_ratio;
#[cfg(test)]
pub(crate) use k_ratio::log_regime;
#[cfg(test)]
pub(crate) use k_ratio::ratio_regime;
pub(crate) use quotient::SHAPE_CUT;
pub use quotient::expm1_over;
pub use quotient::ln_1p_over;
pub(crate) use quotient::ln_1p_over_xi;
pub(crate) use quotient::pow_m1_over_xi;

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

/// Logarithm of the gamma function to ~14 digits; +∞ at the poles, NaN where Γ(x) < 0.
/// Lanczos (1964, DOI 10.1137/0701008) with Godfrey's g = 7, n = 9 coefficients.
#[inline]
pub fn ln_gamma(x: f64) -> f64 {
  if is_pole(x) {
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
  if is_pole(x) {
    return f64::NAN;
  }
  if x < 0.5 {
    std::f64::consts::PI / (sinpi(x) * gamma(1.0 - x))
  } else {
    ln_gamma(x).exp()
  }
}

/// `Γ(a + ½)/(Γ(a)√a)`, which tends to 1, free of the cancellation of the two logarithms: Stirling's difference series
/// from `a = 8`, below it `Γ(a + 1) = aΓ(a)` lifting `a` past 8.
pub(crate) fn gamma_ratio_half(a: f64) -> f64 {
  const LIFT: f64 = 8.0;

  /// `−(2 − 2^{1−2m})B_{2m}/(2m(2m − 1))`, `m = 1, 2, …`: the coefficient of `a^{1−2m}` in `ln Γ(a + ½) − ln Γ(a) − ½ ln a`.
  const C: [f64; 11] = [
    -1.0 / 8.0,
    1.0 / 192.0,
    -1.0 / 640.0,
    17.0 / 14336.0,
    -31.0 / 18432.0,
    691.0 / 180224.0,
    -5461.0 / 425984.0,
    929569.0 / 15728640.0,
    -3202291.0 / 8912896.0,
    221930581.0 / 79691776.0,
    -4722116521.0 / 176160768.0,
  ];
  let series = |b: f64| {
    let r = b.recip();
    let r2 = r * r;
    (C.iter().rev().fold(0.0, |acc, &c| acc * r2 + c) * r).exp()
  };
  if a >= LIFT {
    return series(a);
  }
  let n = (LIFT - a).ceil();
  let (mut num, mut den) = (1.0, a + 0.5);
  let mut k = 1.0;
  while k < n {
    num *= a + k;
    den *= a + k + 0.5;
    k += 1.0;
  }
  let b = a + n;
  series(b) * b.sqrt() * a.sqrt() * (num / den)
}

/// The non-positive integers, where Γ, ln Γ and ψ have their poles.
#[inline]
fn is_pole(x: f64) -> bool {
  x <= 0.0 && x.fract() == 0.0
}

/// `sin(πx)` with `x` reduced to `[-½, ½]` before scaling, so it keeps its
/// relative accuracy next to the integers.
pub(crate) fn sinpi(x: f64) -> f64 {
  let n = x.round();
  let s = (std::f64::consts::PI * (x - n)).sin();
  if n % 2.0 == 0.0 { s } else { -s }
}

/// Digamma ψ(x) = Γ'(x)/Γ(x): the recurrence ψ(x) = ψ(x+1) − 1/x lifts x above 6, then an
/// asymptotic series in 1/x; NaN at the poles.
pub fn digamma(x: f64) -> f64 {
  if is_pole(x) {
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
  erfc::erfc(x)
}

/// Standard-normal quantile: Acklam's (2003) rational approximation and Halley step on `(0, ½]`, mirrored through the exact
/// `1 − p` above, and a Newton step on `ln Φ` at a subnormal `p`, where Halley's step loses its digits; NaN off `[0, 1]`.
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
  let t = ndtri_lower(p);
  if p < f64::MIN_POSITIVE {
    return subnormal_quantile(p, t);
  }
  t
}

/// The Newton step on `ln Φ` from the approximation `t` at a subnormal `p`.
#[cold]
#[inline(never)]
fn subnormal_quantile(p: f64, t: f64) -> f64 {
  let z = -t.abs();
  t.signum() * -(z - (ln_norm_cdf(z) - p.ln()) * mills_ratio(-z))
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

/// Mills' ratio `R(z) = Φ(−z)/φ(z)` for `z ≥ 0`: the quotient below 3, else Laplace's continued fraction
/// `1/(z + 1/(z + 2/(z + …)))` (DLMF 7.9.3 at `iz/√2`) by Lentz's method, free of the `e^{−z²/2}` both factors carry.
pub fn mills_ratio(z: f64) -> f64 {
  if z < 3.0 {
    return norm_cdf(-z) / norm_pdf(z);
  }
  if z > 1e8 {
    return 1.0 / z;
  }
  let (mut f, mut c, mut d) = (z, z, 0.0);
  for n in 1..300 {
    let a = f64::from(n);
    d = 1.0 / (z + a * d);
    c = z + a / c;
    let step = c * d;
    f *= step;
    if (step - 1.0).abs() <= f64::EPSILON {
      break;
    }
  }
  1.0 / f
}

/// `ln Φ(z)`: `ln_1p(−Φ(−z))` above 0, `ln Φ(z)` to −30 and `−z²/2 − ln √(2π) + ln R(−z)` below, the switch a point of
/// `[−37.5, −3]`, where `Φ` is still normal and the continued fraction converged, so both forms hold to 3 ulps.
pub fn ln_norm_cdf(z: f64) -> f64 {
  if z > 0.0 {
    (-norm_cdf(-z)).ln_1p()
  } else if z > -30.0 {
    norm_cdf(z).ln()
  } else {
    -0.5 * z * z - 0.5 * std::f64::consts::TAU.ln() + mills_ratio(-z).ln()
  }
}

#[cfg(test)]
mod tests;
