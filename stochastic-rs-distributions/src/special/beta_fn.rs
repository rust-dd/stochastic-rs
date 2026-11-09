//! `ln B(a, b)` in two parts and the beta entropy free of the `ln Γ` cancellation: Stirling's
//! difference `ln Γ(z + p) − ln Γ(z)` from 16, `ln Γ` below 32 by `1/Γ(1 + f)`'s series.
//! Reference: NIST Digital Library of Mathematical Functions, eq. 5.11.1 (Stirling's series), <https://dlmf.nist.gov/5.11.1>; eq. 5.11.2 (`ψ`'s series), <https://dlmf.nist.gov/5.11.2>.
//! Reference: Abramowitz, M., Stegun, I.A. (1964), "Handbook of Mathematical Functions", NBS AMS 55, 6.1.34 (the power series of `1/Γ`).

use super::HALF_LN_TAU;
use super::STIRLING_FROM;
use super::add_parts;
use super::digamma;
use super::digamma_tail;
use super::kl_divergence;
use super::ln_gamma_star;
use super::ln_parts;
use super::mul_parts;
use super::recip_gamma_1p_m1;
use super::sub_parts;
use super::two_sum;

/// From this shape, both, the entropy and `beta_i`'s prefactor read Stirling's `Γ*` in place of `ln
/// Γ`.
pub(crate) const STIRLING_SHAPES: f64 = 10.0;

/// From this shape, either, the beta law reads this module: the plain closed forms' `ln Γ` sums
/// pass the bar from `a + b ≈ 1200` (`I_0.26(323, 879)` 2e-12 off, `pdf(981, 69, 0.59)` 1.1e-12).
pub(crate) const BETA_LARGE_SHAPE: f64 = 100.0;

/// `ln Γ(p + p_lo)` in two parts for `0 < p ≤ 32`, `p_lo` a shape sum's rounding (`ψ(p)·p_lo` is
/// 1e-14 at 32): `ln Γ(1 + f)` by A&S 6.1.34 and the exact product `Π (f + k)`, `p = n + f`.
pub(crate) fn ln_gamma_parts(p: f64, p_lo: f64) -> (f64, f64) {
  // `ln Γ(q + p_lo) = ln Γ(q) + ψ(q)·p_lo`; `−0.0` keeps a zero part's sign at an exact sum.
  let shift = |q: f64| if p_lo == 0.0 { -0.0 } else { digamma(q) * p_lo };
  if p <= 1.0 {
    let (l, l_lo) = ln_parts(p, p_lo, 0);
    let g = -recip_gamma_1p_m1(p).ln_1p() + shift(1.0 + p);
    return two_sum(-l, g - l_lo);
  }
  let n = p.floor();
  let f = p - n;
  let mut product = (1.0, 0.0);
  let mut k = 1.0;
  while k < n {
    let (t, t_lo) = two_sum(k, f);
    product = mul_parts(product, (t, t_lo + p_lo));
    k += 1.0;
  }
  let (l, l_lo) = ln_parts(product.0, product.1, 0);
  two_sum(l, l_lo - recip_gamma_1p_m1(f).ln_1p() + shift(1.0 + f))
}

/// `ln Γ(z + p) − ln Γ(z)` in two parts for `z ≥ 16`: `p ln z − z·g(p/z) + (p − ½) ln(1 + p/z)`
/// plus the `ln Γ*` difference, `g(u) = u − ln(1 + u)`, every term of the size of the difference.
pub(crate) fn ln_gamma_diff(z: f64, p: f64) -> (f64, f64) {
  let first = mul_parts((p, 0.0), ln_parts(z, 0.0, 0));
  let (w, w_err) = two_sum(z, p);
  let q = w / z;
  let q_lo = ((-q).mul_add(z, w) + w_err) / z;
  let third = mul_parts((p - 0.5, 0.0), ln_parts(q, q_lo, 0));
  let (s, s_lo) = add_parts(
    sub_parts(first, kl_divergence(z, (p, 0.0), (w, w_err))),
    third,
  );
  two_sum(s, s_lo + (ln_gamma_star(w) - ln_gamma_star(z)))
}

/// `ln B(a, b)` in two parts, the smaller shape `p` to 32: `ln Γ(p) − [ln Γ(q + p) − ln Γ(q)]`,
/// the bracket Stirling's difference from 16, else three logs, `p + q` two-part where `exact_sums`.
pub(crate) fn ln_beta_parts(a: f64, b: f64, exact_sums: bool) -> (f64, f64) {
  let (p, q) = if a <= b { (a, b) } else { (b, a) };
  debug_assert!(p <= 32.0, "ln_beta_parts takes a shape at most 32, got {p}");
  let lg = ln_gamma_parts(p, 0.0);
  if q >= STIRLING_FROM {
    return sub_parts(lg, ln_gamma_diff(q, p));
  }
  // The shapes' sum in two parts: rounded, `ψ(p + q)` times its half ulp reaches 6e-15 below 32.
  let (s, s_lo) = two_sum(p, q);
  sub_parts(
    add_parts(lg, ln_gamma_parts(q, 0.0)),
    ln_gamma_parts(s, if exact_sums { s_lo } else { 0.0 }),
  )
}

/// `ln(s·B(a, b))`, `s` one of the shapes, in two parts with the smaller shape to 32: `ln Γ(s + 1)`
/// replaces `ln Γ(s)`, so a tiny `s` cancels no `ln s`; the shape sums two-part where `exact_sums`.
pub(crate) fn ln_shape_beta_parts(s: f64, other: f64, exact_sums: bool) -> (f64, f64) {
  let low = |lo: f64| if exact_sums { lo } else { 0.0 };
  if s <= other {
    let (s1, s1_lo) = two_sum(s, 1.0);
    let lg = ln_gamma_parts(s1, low(s1_lo));
    if other >= STIRLING_FROM {
      return sub_parts(lg, ln_gamma_diff(other, s));
    }
    let (t, t_lo) = two_sum(s, other);
    return sub_parts(
      add_parts(lg, ln_gamma_parts(other, 0.0)),
      ln_gamma_parts(t, low(t_lo)),
    );
  }
  add_parts(ln_beta_parts(s, other, exact_sums), ln_parts(s, 0.0, 0))
}

/// `ψ(a + b) − ψ(b)` for `b ≥ 10`, from `ψ`'s series: `ln(1 + a/b) + 1/(2b) − 1/(2(a + b))` less
/// the tails' difference.
fn digamma_step(b: f64, a: f64) -> f64 {
  let s = a + b;
  (a / b).ln_1p() + (0.5 / b - 0.5 / s) + (digamma_tail(b) - digamma_tail(s))
}

/// `ln B(a, b) − (a − 1)ψ(a) − (b − 1)ψ(b) + (a + b − 2)ψ(a + b)`: both shapes from 10 by `Γ*` and
/// `ψ`'s tail, where terms of order `(a + b) ln(a + b)` would cancel, else by differences.
pub(crate) fn beta_entropy(a: f64, b: f64) -> f64 {
  let (p, q) = if a <= b { (a, b) } else { (b, a) };
  if p >= STIRLING_SHAPES {
    let half = 0.5 * p + 0.5 * q;
    // `½ ln(pq/(p + q)³)` by its logarithms, since the quotient underflows at huge shapes.
    let spread = 0.5 * (p.ln() + q.ln() - 3.0 * (half.ln() + std::f64::consts::LN_2));
    let star = ln_gamma_star(p) + ln_gamma_star(q) - ln_gamma_star(p + q);
    // The sum's tail, `(p + q − 2)/(12(p + q)) + …`, is below `ε` of the value where `p + q`
    // overflows.
    let sum_tail = if (p + q).is_finite() {
      (p + q - 2.0) * digamma_tail(p + q)
    } else {
      0.0
    };
    let tails = (p - 1.0) * digamma_tail(p) + (q - 1.0) * digamma_tail(q) - sum_tail;
    return 0.5 + HALF_LN_TAU.0 + spread - 0.5 / p - 0.5 / q
      + 0.5 / half
      + star
      + tails
      + HALF_LN_TAU.1;
  }
  // `ln_gamma_diff` needs `q ≥ 16`, `digamma_step` `q ≥ 10`: the caller's larger shape is ≥ 100.
  debug_assert!(q >= STIRLING_FROM);
  let (lb, lb_lo) = sub_parts(ln_gamma_parts(p, 0.0), ln_gamma_diff(q, p));
  let step = digamma_step(q, p);
  lb + ((q - 1.0) * step + (p - 1.0) * (digamma(q) + step - digamma(p)) + lb_lo)
}
