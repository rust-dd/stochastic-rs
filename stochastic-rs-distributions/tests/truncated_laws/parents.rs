//! The truncated laws' parents from the crate's special functions: both tails at a point, each with
//! a relative error bound, so a survey compares a cdf only where its reference holds digits.

use std::f64::consts::FRAC_1_SQRT_2;
use std::f64::consts::FRAC_2_SQRT_PI;

use stochastic_rs_distributions::beta::SimdBeta;
use stochastic_rs_distributions::special::beta_i;
use stochastic_rs_distributions::special::erfc;
use stochastic_rs_distributions::special::gamma_p;
use stochastic_rs_distributions::special::gamma_q;
use stochastic_rs_distributions::special::ln_gamma;
use stochastic_rs_distributions::traits::DistributionExt;

const EPS: f64 = f64::EPSILON;

/// The 16-point Gauss–Legendre rule's positive nodes and weights (mpmath at 60 and 120 digits).
const GL16: [(f64, f64); 8] = [
  (0.989_400_934_991_649_9, 0.027_152_459_411_754_096),
  (0.944_575_023_073_232_6, 0.062_253_523_938_647_894),
  (0.865_631_202_387_831_8, 0.095_158_511_682_492_79),
  (0.755_404_408_355_003, 0.124_628_971_255_533_88),
  (0.617_876_244_402_643_8, 0.149_595_988_816_576_74),
  (0.458_016_777_657_227_37, 0.169_156_519_395_002_54),
  (0.281_603_550_779_258_9, 0.182_603_415_044_923_58),
  (0.095_012_509_837_637_44, 0.189_450_610_455_068_5),
];

/// `1/√2 − FRAC_1_SQRT_2`, from mpmath.
const FRAC_1_SQRT_2_LO: f64 = -4.833646656726457e-17;

/// `P` and `Q = 1 − P` of a parent law at a point, each with a bound on its relative error.
#[derive(Clone, Copy, Debug)]
pub struct Tails {
  pub p: f64,
  pub q: f64,
  pub err_p: f64,
  pub err_q: f64,
}

impl Tails {
  fn limit(p: f64) -> Self {
    Self {
      p,
      q: 1.0 - p,
      err_p: 0.0,
      err_q: 0.0,
    }
  }

  fn unknown() -> Self {
    Self {
      p: f64::NAN,
      q: f64::NAN,
      err_p: f64::INFINITY,
      err_q: f64::INFINITY,
    }
  }
}

/// The parent's mass on `[l, r]` with its error bound, from the tail whose difference keeps more
/// digits: a cancelling difference carries its terms' errors over its size; below 1e-290, none.
pub fn between(l: Tails, r: Tails) -> (f64, f64) {
  let form = |big: f64, e_big: f64, small: f64, e_small: f64| {
    let v = big - small;
    if v >= 1e-290 {
      (v, (big.abs() * e_big + small.abs() * e_small) / v + EPS)
    } else {
      (v, f64::INFINITY)
    }
  };
  let by_p = form(r.p, r.err_p, l.p, l.err_p);
  let by_q = form(l.q, l.err_q, r.q, r.err_q);
  if by_p.1 <= by_q.1 { by_p } else { by_q }
}

/// A tail's error bound, unbounded where the value or the bound left its range; an underflowed 0
/// holds no digit worth a bound.
fn bounded(v: f64, err: f64) -> f64 {
  if v == 0.0 {
    0.0
  } else if (0.0..=1.0 + 1e-15).contains(&v) && err >= 0.0 {
    err
  } else {
    f64::INFINITY
  }
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
  let s = a + b;
  let bb = s - a;
  (s, (a - (s - bb)) + (b - bb))
}

/// `ln Γ(z)`'s absolute error: Lanczos' terms each round on the scale of `|ln Γ(z)| + z`.
fn ln_gamma_err(z: f64) -> f64 {
  2.0 * EPS * (ln_gamma(z).abs() + z + 1.0) + 1e-15
}

/// `N(μ, σ²)` at `x`: `z = (x − μ)/σ` to double-double, its rest carried into `erfc` by its derivative.
pub fn normal(mu: f64, sigma: f64, x: f64) -> Tails {
  if x.is_infinite() {
    return Tails::limit(if x > 0.0 { 1.0 } else { 0.0 });
  }
  let (mut d, mut d_lo) = two_sum(x, -mu);
  let mut s = sigma;
  if d.is_infinite() {
    (d, d_lo) = two_sum(0.5 * x, -0.5 * mu);
    s = 0.5 * sigma;
  }
  let z = d / s;
  if !z.is_finite() {
    return Tails::limit(if z > 0.0 { 1.0 } else { 0.0 });
  }
  let z_lo = ((-z).mul_add(s, d) + d_lo) / s;
  let w = z * FRAC_1_SQRT_2;
  let w_lo = z.mul_add(FRAC_1_SQRT_2, -w) + z * FRAC_1_SQRT_2_LO + z_lo * FRAC_1_SQRT_2;
  let g = FRAC_2_SQRT_PI * (-w * w).exp() * w_lo;
  Tails {
    p: 0.5 * (erfc(-w) + g),
    q: 0.5 * (erfc(w) - g),
    err_p: 4.0 * EPS,
    err_q: 4.0 * EPS,
  }
}

/// `Gamma(k, θ)` at `x`: `u = x/θ` in double-double, its rest through the density, each tail's
/// error by the branch `gamma_p` and `gamma_q` take (one tail as one minus the other).
pub fn gamma(k: f64, theta: f64, x: f64) -> Tails {
  if x <= 0.0 {
    return Tails::limit(0.0);
  }
  let u = x / theta;
  if u == f64::INFINITY {
    return Tails::limit(1.0);
  }
  if u < f64::MIN_POSITIVE {
    return Tails::unknown();
  }
  let u_lo = (-u).mul_add(theta, x) / theta;
  let (mut p, mut q) = (gamma_p(k, u), gamma_q(k, u));
  let ln_f = (k - 1.0) * u.ln() - u - ln_gamma(k);
  let err_f = 4.0 * EPS * ((k - 1.0).abs() * u.ln().abs() + u + ln_gamma(k).abs()) + 1e-15;
  // The density's exponent loses its digits with the shape: past 1e-3 it corrects nothing.
  let corrected = u_lo != 0.0 && err_f <= 1e-3;
  if corrected {
    let shift = u_lo * ln_f.exp();
    p += shift;
    q -= shift;
  }
  // `d ln P/d ln u` and `d ln Q/d ln u` stay below this: `k` in the lower tail, `u` in the upper.
  let slope = (u_lo / u).abs() * (k.max(u) + 1.0);
  let err_u = if corrected {
    slope * err_f + slope * slope
  } else {
    slope
  };
  let base = if k < 10.0 {
    4.0 * EPS * (u + (k * u.ln()).abs() + ln_gamma(k).abs() + 1.0) + 1e-14
  } else {
    1e-14
  };
  let upper = if k < 100.0 { u >= k + 1.0 } else { u >= k };
  let q_direct = upper || (k < 1.0 && u < k + 1.0);
  Tails {
    p,
    q,
    err_p: bounded(p, if upper { (EPS + base * q) / p } else { base } + err_u),
    err_q: bounded(
      q,
      if q_direct { base } else { (EPS + base * p) / q } + err_u,
    ),
  }
}

/// `Beta(a, b)` at `x`: `beta_i(a, b, x)` and `beta_i(b, a, 1 − x)`, each prefactor erring by its
/// terms' rounding, and `Q` by the mass between `x` and the point a rounded `1 − x` stands for.
pub fn beta(a: f64, b: f64, x: f64) -> Tails {
  if x <= 0.0 {
    return Tails::limit(0.0);
  }
  if x >= 1.0 {
    return Tails::limit(1.0);
  }
  let (y, y_lo) = two_sum(1.0, -x);
  let (p, q) = (beta_i(a, b, x), beta_i(b, a, y));
  let terms = (a * x.ln()).abs() + (b * y.ln()).abs();
  let prefactor =
    ln_gamma_err(a + b) + ln_gamma_err(a) + ln_gamma_err(b) + 2.0 * EPS * terms + 1e-15;
  let fraction = if a.max(b) <= 1e3 {
    4e-15
  } else {
    f64::INFINITY
  };
  let base = prefactor + fraction;
  let argument = if x < 0.5 { EPS * (a.max(b) + 1.0) } else { 0.0 };
  // `y` is the exact `1 − x'`, `x' = x + y_lo`; the mass between them is under the larger density
  // at an end where it is monotone across them, and all of `P(x)` where `y` rounds to 1.
  let shift = if y == 1.0 {
    p
  } else {
    let law = SimdBeta::<f64>::new(a, b);
    let at = |t: f64| law.pdf(t).unwrap_or(f64::NAN);
    at(x).max(at(1.0 - y)) * y_lo.abs()
  };
  let p_direct = x < (a + 1.0) / (a + b + 2.0);
  let q_direct = y < (b + 1.0) / (a + b + 2.0);
  Tails {
    p,
    q,
    err_p: bounded(
      p,
      if p_direct {
        base
      } else {
        (EPS + base * q + shift) / p
      } + argument,
    ),
    err_q: bounded(
      q,
      if q_direct { base } else { (EPS + base * p) / q } + argument + shift / q,
    ),
  }
}

/// `Beta(a, b)`'s mass on `[l, r]` and its relative error bound: the tails' where they hold `gate`,
/// else, where they hold more, the law's own density in panels from each end (`0 < l`, `r < 1`).
pub fn beta_between(a: f64, b: f64, l: f64, r: f64, gate: f64) -> (f64, f64) {
  let forms = between(beta(a, b, l), beta(a, b, r));
  if forms.1 <= gate || l <= 0.0 || r >= 1.0 || r <= l {
    return forms;
  }
  let panels = beta_panels(a, b, l, r);
  if panels.1 < forms.1 { panels } else { forms }
}

/// Below ½ `Beta(a, b)`'s density in `ln t` from `l`, above it `Beta(b, a)`'s in `ln(1 − t)` from
/// `1 − r`, no singular end or rounded `1 − t` in a node: the rule on n and 2n panels, n ≥ span.
fn beta_panels(a: f64, b: f64, l: f64, r: f64) -> (f64, f64) {
  let (below, above) = (r.min(0.5), l.max(0.5));
  let (mut coarse, mut fine, mut held) = (0.0, 0.0, 0.0);
  for (p, q, near, far) in [(a, b, l, below), (b, a, 1.0 - r, 1.0 - above)] {
    if near >= far {
      continue;
    }
    let law = SimdBeta::<f64>::new(p, q);
    // `ln(far/near)` from the exact `far − near` and each node as `near·e^v`, where a difference of
    // logs and `e^{ln near + v}` round at an ulp of `ln near`, a share of a narrow interval.
    let span = ((far - near) / near).ln_1p();
    let sum = |n: f64| {
      let h = span / n;
      let (mut s, mut c, mut e) = (0.0, 0.0, 0.0);
      for k in 0..n as usize {
        let mid = h * (k as f64 + 0.5);
        for (x, w) in GL16 {
          for v in [mid - 0.5 * h * x, mid + 0.5 * h * x] {
            let t = near + near * v.exp_m1();
            let term = 0.5 * h * w * t * law.pdf(t).unwrap_or(f64::NAN);
            let (next, err) = two_sum(s, term);
            (s, c) = (next, c + err);
            // The density's log terms round once each, `1 − t` as `t/(1 − t)` of its log.
            let logs = (p - 1.0).abs() * t.ln().abs()
              + (q - 1.0).abs() * ((-t).ln_1p().abs() + t / (1.0 - t));
            e += term.abs() * EPS * (logs + 4.0);
          }
        }
      }
      (s + c, e)
    };
    let n = span.ceil().max(1.0);
    let ((c1, _), (f2, e2)) = (sum(n), sum(2.0 * n));
    (coarse, fine, held) = (coarse + c1, fine + f2, held + e2);
  }
  let err =
    ((coarse - fine).abs() + held) / fine + ln_gamma_err(a) + ln_gamma_err(b) + ln_gamma_err(a + b);
  (fine, if err.is_nan() { f64::INFINITY } else { err })
}

/// Survey law 475's points (seed `0x5eed_2026`) from the panels and law 3652's (seed 11) from its P
/// form, where `1 − x` rounds the tails' digits off: mpmath at 60 and 120 digits, panels beside it.
#[test]
fn the_parent_resolves_what_its_tails_round_away() {
  let ratio = |(a, b, lo, up): (f64, f64, f64, f64), x: f64| {
    let ((mass, err_mass), (part, err_part)) = (
      beta_between(a, b, lo, up, 2e-13),
      beta_between(a, b, lo, x, 2e-13),
    );
    assert!(err_mass + err_part <= 2e-13, "TB({a:e}, {b:e}) at {x:e}");
    part / mass
  };
  let law_475 = (
    4.695248705244798e-23,
    3.312248097819112e-8,
    6.543618931704305e-119,
    0.7908536524565434,
  );
  let law_3652 = (4.936562329156904e-3, 20.56978573562787, 0.0, 1.0);
  for (law, x, want) in [
    (law_475, 5.028784517286747e-12, 0.899_999_999_999_999_7),
    (law_475, 0.7420439923424864, 0.999_000_000_000_000_1),
    (law_475, 0.7908084175669646, 0.999_999_000_000_000_1),
    (law_3652, 1.5112500415742897e-11, 0.900_000_000_000_000_1),
  ] {
    let got = ratio(law, x);
    assert!(
      (got - want).abs() <= 2e-13 * want,
      "the parent at {x:e}: {got:e}, want {want:e}"
    );
  }
}
