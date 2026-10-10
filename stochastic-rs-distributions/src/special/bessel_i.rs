//! $I_\nu(x)$ for real order: Debye's uniform expansion above order 50, the ascending series
//! for `x < 2`, Hankel's expansion for `x ≥ max(50, ν²/2)`, Temme/Steed with the Wronskian between.
//!
//! References:
//! - Temme (1975), "On the numerical evaluation of the modified Bessel function of the third kind", J. Comput. Phys. 19(3), DOI 10.1016/0021-9991(75)90082-0
//! - Thompson, Barnett (1987), "Modified Bessel functions I_ν(z) and K_ν(z) of real order and complex argument, to selected accuracy", Comput. Phys. Commun. 47(2-3), DOI 10.1016/0010-4655(87)90111-1
//! - Thompson, Barnett (1986), "Coulomb and Bessel functions of complex arguments and order", J. Comput. Phys. 64(2), DOI 10.1016/0021-9991(86)90046-X (Lentz's method for CF1)
//! - Olver (1954), "The asymptotic expansion of Bessel functions of large order", Phil. Trans. R. Soc. A 247, DOI 10.1098/rsta.1954.0021
//! - NIST DLMF §§10.25–10.41, <https://dlmf.nist.gov/10> (10.25.2 series, 10.27.1–2 negative orders, 10.28.2 Wronskian, 10.40.1 Hankel, 10.41.3/10.41.9 uniform)

use std::f64::consts::FRAC_2_PI;
use std::f64::consts::LN_2;
use std::f64::consts::TAU;

use super::bessel_k::bessel_ke;
use super::bessel_k::ke_pair;
use super::gamma;
use super::sinpi;

const EPS: f64 = f64::EPSILON;
const UNIFORM_MIN_ORDER: f64 = 50.0;
const SERIES_MAX_X: f64 = 2.0;
const HANKEL_MIN_X: f64 = 50.0;
const MAX_TERMS: usize = 64;
const CF1_MAX_ITER: usize = 100_000;

/// Coefficients of $U_k(p) = p^k \sum_j c_{k,j} p^{2j}$, `k = 1..=10`, from DLMF 10.41.9 (recursion; 10.41.10 lists U₁–U₃).
const DEBYE: [&[f64]; 10] = [
  &[0.125, -0.20833333333333334],
  &[0.0703125, -0.4010416666666667, 0.3342013888888889],
  &[
    0.0732421875,
    -0.8912109375,
    1.8464626736111112,
    -1.0258125964506173,
  ],
  &[
    0.112152099609375,
    -2.3640869140625,
    8.78912353515625,
    -11.207002616222994,
    4.669584423426247,
  ],
  &[
    0.22710800170898438,
    -7.368794359479632,
    42.53499874538846,
    -91.81824154324002,
    84.63621767460073,
    -28.212072558200244,
  ],
  &[
    0.5725014209747314,
    -26.491430486951554,
    218.1905117442116,
    -699.5796273761325,
    1059.9904525279999,
    -765.2524681411817,
    212.57013003921713,
  ],
  &[
    1.7277275025844574,
    -108.09091978839466,
    1200.9029132163525,
    -5305.646978613403,
    11655.393336864534,
    -13586.550006434138,
    8061.722181737309,
    -1919.457662318407,
  ],
  &[
    6.074042001273483,
    -493.915304773088,
    7109.514302489364,
    -41192.65496889755,
    122200.46498301746,
    -203400.17728041555,
    192547.00123253153,
    -96980.59838863752,
    20204.29133096615,
  ],
  &[
    24.380529699556064,
    -2499.8304818112097,
    45218.76898136273,
    -331645.1724845636,
    1268365.2733216248,
    -2813563.226586534,
    3763271.297656404,
    -2998015.9185381066,
    1311763.6146629772,
    -242919.18790055133,
  ],
  &[
    110.01714026924674,
    -13886.08975371704,
    308186.4046126624,
    -2785618.1280864547,
    13288767.166421818,
    -37567176.66076335,
    66344512.27472903,
    -74105148.21153265,
    50952602.49266464,
    -19706819.118432228,
    3284469.853072038,
  ],
];

/// $I_\nu(x)$ for real `nu`; NaN when `x < 0` and `nu` is not an integer.
pub fn bessel_i(nu: f64, x: f64) -> f64 {
  evaluate(nu, x, false)
}

/// $e^{-|x|} I_\nu(x)$, finite for every `x` where [`bessel_i`] overflows.
pub fn bessel_ie(nu: f64, x: f64) -> f64 {
  evaluate(nu, x, true)
}

/// $\ln(e^{-x} I_\nu(x))$, finite for `x > 0` and `nu > -1` even where [`bessel_ie`]
/// underflows; NaN for `x < 0` or where $I_\nu(x) < 0$.
pub fn ln_bessel_ie(nu: f64, x: f64) -> f64 {
  if nu.is_nan() || nu.is_infinite() || x.is_nan() || x < 0.0 {
    return f64::NAN;
  }
  let nu = integer_symmetric(nu);
  if x == 0.0 || x.is_infinite() {
    return evaluate(nu, x, true).ln();
  }
  if nu > UNIFORM_MIN_ORDER {
    let (_, eta_minus_z, prefactor) = uniform(nu, x);
    return eta_minus_z + prefactor.ln();
  }
  if x < SERIES_MAX_X {
    let g = gamma_succ(nu);
    let s = series(nu, x);
    if g.is_sign_negative() != s.is_sign_negative() {
      return f64::NAN;
    }
    return nu * (x.ln() - LN_2) - g.abs().ln() + s.abs().ln() - x;
  }
  evaluate(nu, x, true).ln()
}

fn evaluate(nu: f64, x: f64, scaled: bool) -> f64 {
  if nu.is_nan() || nu.is_infinite() || x.is_nan() {
    return f64::NAN;
  }
  let nu = integer_symmetric(nu);
  if x < 0.0 {
    if nu.fract() != 0.0 {
      return f64::NAN;
    }
    let sign = if nu % 2.0 == 0.0 { 1.0 } else { -1.0 };
    return sign * evaluate(nu, -x, scaled);
  }
  if x == 0.0 {
    return at_zero(nu);
  }
  if x.is_infinite() {
    return if scaled { 0.0 } else { f64::INFINITY };
  }
  if nu > UNIFORM_MIN_ORDER || (nu >= 0.0 && x >= SERIES_MAX_X) {
    return positive_order(nu, x, scaled);
  }
  if x < SERIES_MAX_X {
    let value = x.powf(nu) * (-nu).exp2() / gamma_succ(nu) * series(nu, x);
    return if scaled { value * (-x).exp() } else { value };
  }
  let a = -nu;
  let k_term = FRAC_2_PI * sinpi(a) * bessel_ke(a, x);
  let decay = if scaled { -2.0 * x } else { -x };
  positive_order(a, x, scaled) + k_term * decay.exp()
}

/// DLMF 10.27.1: $I_{-n} = I_n$ for integer `n`.
fn integer_symmetric(nu: f64) -> f64 {
  if nu < 0.0 && nu.fract() == 0.0 {
    -nu
  } else {
    nu
  }
}

/// The limit at `x = 0` (DLMF 10.30.1).
fn at_zero(nu: f64) -> f64 {
  if nu == 0.0 {
    1.0
  } else if nu > 0.0 {
    0.0
  } else {
    f64::INFINITY.copysign(gamma(nu + 1.0))
  }
}

/// $\Gamma(\nu + 1)$, as $\nu\,\Gamma(\nu)$ for `ν > 0` so `ν + 1` is never rounded.
fn gamma_succ(nu: f64) -> f64 {
  if nu > 0.0 {
    nu * gamma(nu)
  } else {
    gamma(nu + 1.0)
  }
}

/// $I_\nu(x)$ or $e^{-x} I_\nu(x)$ for `nu ≥ 0`, with `x ≥ 2` unless `nu > 50`.
fn positive_order(nu: f64, x: f64, scaled: bool) -> f64 {
  if nu > UNIFORM_MIN_ORDER {
    let (eta, eta_minus_z, prefactor) = uniform(nu, x);
    return exp_times(if scaled { eta_minus_z } else { eta }, prefactor);
  }
  let ie = if x >= HANKEL_MIN_X.max(0.5 * nu * nu) {
    hankel(nu, x)
  } else {
    let (kv, kv1) = ke_pair(nu, x);
    (1.0 / x) / (kv1 + cf1(nu, x) * kv)
  };
  if scaled { ie } else { exp_times(x, ie) }
}

/// $e^a m$, split so a large `a` does not overflow before the product would.
fn exp_times(a: f64, m: f64) -> f64 {
  let half = (0.5 * a).exp();
  half * (m * half)
}

/// DLMF 10.41.3 at `z = x / ν`: `(ν η, ν (η − z), (p / 2πν)^{1/2} Σ U_k(p) / ν^k)`.
fn uniform(nu: f64, x: f64) -> (f64, f64, f64) {
  let z = x / nu;
  let root = z.hypot(1.0);
  let p = 1.0 / root;
  // `asinh(1/z)` overflows for tiny `z`, the log form cancels for large `z`; `ln x − ln ν` stands
  // in for `ln z` only once `z = x / ν` underflows, as it loses bits next to `z = 1`.
  let tail = if z < 1.0 {
    let ln_z = if z.is_normal() {
      z.ln()
    } else {
      x.ln() - nu.ln()
    };
    (1.0 + root).ln() - ln_z
  } else {
    (1.0 / z).asinh()
  };
  let p2 = p * p;
  let mut sum = 1.0;
  let mut scale = 1.0;
  for coefficients in DEBYE {
    scale *= p / nu;
    sum += scale * coefficients.iter().rev().fold(0.0, |acc, &c| acc * p2 + c);
  }
  (
    nu * (root - tail),
    nu * (1.0 / (root + z) - tail),
    (p / (TAU * nu)).sqrt() * sum,
  )
}

/// DLMF 10.25.2 divided by its leading term $(x/2)^\nu / \Gamma(\nu + 1)$.
fn series(nu: f64, x: f64) -> f64 {
  let q = 0.25 * x * x;
  let mut term = 1.0;
  let mut sum = 1.0;
  for k in 1..MAX_TERMS {
    let k = k as f64;
    term *= q / (k * (nu + k));
    sum += term;
    if term.abs() <= EPS * sum.abs() {
      break;
    }
  }
  sum
}

/// DLMF 10.40.1 for $e^{-x} I_\nu(x)$; from `x = max(50, ν²/2)` on, the terms
/// reach ε within about 20 of them, long before they start to grow.
fn hankel(nu: f64, x: f64) -> f64 {
  let mu = 4.0 * nu * nu;
  let mut term = 1.0;
  let mut sum = 1.0;
  for k in 1..MAX_TERMS {
    let odd = (2 * k - 1) as f64;
    term *= -(mu - odd * odd) / (8.0 * k as f64 * x);
    sum += term;
    if term.abs() <= EPS * sum.abs() {
      break;
    }
  }
  sum / (TAU.sqrt() * x.sqrt())
}

/// CF1, $I_{\nu+1}(x) / I_\nu(x)$, by Lentz's method; every partial denominator is positive.
fn cf1(nu: f64, x: f64) -> f64 {
  let tiny = f64::MIN_POSITIVE.sqrt();
  let xi2 = 2.0 / x;
  let mut f = tiny;
  let mut c = tiny;
  let mut d = 0.0;
  for k in 1..=CF1_MAX_ITER {
    let b = (nu + k as f64) * xi2;
    c = b + 1.0 / c;
    d = 1.0 / (b + d);
    let delta = c * d;
    f *= delta;
    if (delta - 1.0).abs() <= EPS {
      return f;
    }
  }
  f64::NAN
}

#[cfg(test)]
mod tests;
