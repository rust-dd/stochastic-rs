//! The density from `r = √(λ² + δ²γ²) = 50`: the normal mixture at its GIG clock's saddles, as the Debye forms meet.

use crate::special::add_parts;
use crate::special::debye_log_sum;
use crate::special::divergence;
use crate::special::exp_times_over_pow2;
use crate::special::ldexp;
use crate::special::ln_saddle;
use crate::special::mul_parts;
use crate::special::offset;
use crate::special::saddle;
use crate::special::saddle_parts;
use crate::special::two_prod;
use crate::special::two_sum;

/// The clock's saddles the concentrated density takes as they are, `2^{±512}`.
const SADDLE_RANGE: (f64, f64) = (7.458_340_731_200_207e-155, 1.340_780_792_994_259_7e154);

/// Where the clock's saddle is past `2^{±512}` or a square of its clock leaves the normal range, the law of `X/2ᵏ` with that
/// saddle near 1, whose squares are below `2r` and move the clock by `ω²/r` if subnormal.
pub(super) fn density(
  (lambda, alpha, beta, delta): (f64, f64, f64, f64),
  d: (f64, f64),
  (gamma, r0): (f64, f64),
  p: i32,
) -> f64 {
  let w0 = saddle(lambda, delta, gamma);
  if w0 > SADDLE_RANGE.0
    && w0 < SADDLE_RANGE.1
    && (delta * delta).is_normal()
    && (gamma * gamma).is_normal()
  {
    return at_saddle((lambda, alpha, beta, delta), d, (w0, r0), p);
  }
  rescaled((lambda, alpha, beta, delta), d, (gamma, r0), p)
}

#[cold]
#[inline(never)]
fn rescaled(
  (lambda, alpha, beta, delta): (f64, f64, f64, f64),
  d: (f64, f64),
  (gamma, r0): (f64, f64),
  p: i32,
) -> f64 {
  let k = (0.5 * ln_saddle(lambda, delta, gamma) / std::f64::consts::LN_2).round() as i32;
  let (alpha, beta, delta) = (ldexp(alpha, k), ldexp(beta, k), ldexp(delta, -k));
  at_saddle(
    (lambda, alpha, beta, delta),
    (ldexp(d.0, -k), ldexp(d.1, -k)),
    (saddle(lambda, delta, ldexp(gamma, k)), r0),
    p - k,
  )
}

/// `βd + ln N(λ − ½, δ² + d², α²) − ln N(λ, δ², γ²)` meets as `H_λ(w₁) − H_λ(w₀) − ½ ln w₁ − (d − βw₁)²/2w₁`
/// (`α² = γ² + β²`), both maximisers and `d` in two parts, plus `T₁ − T₀`, times `2ᵖ`.
fn at_saddle(
  (lambda, alpha, beta, delta): (f64, f64, f64, f64),
  d: (f64, f64),
  (w0, r0): (f64, f64),
  p: i32,
) -> f64 {
  let (root, dd) = (delta.hypot(d.0), mul_parts(d, d));
  // Past `|d| = 2^{512}` the point lies `2^{256}` deviations off a saddle within `2^{±512}`.
  if !dd.0.is_finite() {
    return 0.0;
  }
  let chi = two_prod(delta, delta);
  let psi = mul_parts(two_sum(alpha, -beta), two_sum(alpha, beta));
  let order = two_sum(lambda, -0.5);
  let w1 = saddle(order.0, root, alpha);
  // Past the normal range `w₁` puts `d` beyond the law's reach: the divergence from `w₀` passes `e^{−10⁴}`.
  if !w1.is_normal() {
    return 0.0;
  }
  let (w0, w1) = (
    saddle_parts(w0, (lambda, 0.0), chi, psi),
    saddle_parts(w1, order, add_parts(chi, dd), two_prod(alpha, alpha)),
  );
  let (bw, bw_lo) = two_prod(beta, w1.0);
  let (g, g_lo) = two_sum(d.0, -bw);
  let gap = g + (g_lo + d.1 - bw_lo - beta * w1.1);
  let r1 = order.0.hypot(alpha * root);
  let a = divergence(lambda, (chi.0, psi.0), w0.0, offset(w1, w0)) - 0.5 * (gap / w1.0) * gap
    + debye_log_sum(order.0.abs(), r1)
    - debye_log_sum(lambda.abs(), r0)
    - 0.5 * std::f64::consts::TAU.ln();
  exp_times_over_pow2(a, ((r0 / r1).sqrt(), w1.0.sqrt()), p)
}
