//! The density from `r = √(λ² + δ²γ²) = 50`: the normal mixture at its GIG clock's saddles, as the Debye forms meet.

use crate::special::Scaled;
use crate::special::add_parts;
use crate::special::debye_log_sum;
use crate::special::divergence;
use crate::special::exact_saddle;
use crate::special::exp_parts;
use crate::special::exp_times_over_pow2;
use crate::special::ldexp;
use crate::special::ln_saddle;
use crate::special::mul_parts;
use crate::special::offset;
use crate::special::quarter;
use crate::special::saddle;
use crate::special::saddle_parts;
use crate::special::two_prod;
use crate::special::two_sum;
use crate::special::two_sum_far;

/// `2¹⁶⁰`: from this `r` the clock's skewness, under `3/√r`, moves a representable density by
/// under `2⁻⁶³`, so the law is normal there, as VG's is from a gamma clock of that shape.
pub(super) const NORMAL_CLOCK: f64 = 1.461_501_637_330_903e48;

/// `2¹⁰²²`, under which a sum of two such parts stays finite.
const HALF_TOP: f64 = 4.494_232_837_155_79e307;

const SQRT_TAU: f64 = 2.506_628_274_631_000_7;

/// The clock's saddles the concentrated density takes as they are, `2^{±512}`.
const SADDLE_RANGE: (f64, f64) = (7.458_340_731_200_207e-155, 1.340_780_792_994_259_7e154);

/// What the concentrated density reads from the parameters alone, formed once a law: the clock's
/// saddle and squares in two parts, the order `λ − ½`, `r₀`, `δγ` and Debye's sum at `(|λ|, r₀)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Clock {
  w0: (f64, f64),
  chi: (f64, f64),
  psi: (f64, f64),
  order: (f64, f64),
  r0: f64,
  omega: f64,
  debye: f64,
}

/// The [`Clock`] of a law whose saddle the density takes as it is: within `2^{±512}`, both squares normal.
pub(super) fn clock(
  (lambda, alpha, beta, delta): (f64, f64, f64, f64),
  (gamma, r0): (f64, f64),
) -> Option<Clock> {
  let w0 = saddle(lambda, delta, gamma);
  let plain = w0 > SADDLE_RANGE.0
    && w0 < SADDLE_RANGE.1
    && (delta * delta).is_normal()
    && (gamma * gamma).is_normal();
  plain.then(|| clock_at((lambda, alpha, beta, delta), (w0, r0, delta * gamma)))
}

fn clock_at(
  (lambda, alpha, beta, delta): (f64, f64, f64, f64),
  (w0, r0, omega): (f64, f64, f64),
) -> Clock {
  let chi = two_prod(delta, delta);
  let psi = mul_parts(two_sum(alpha, -beta), two_sum(alpha, beta));
  Clock {
    w0: saddle_parts(w0, (lambda, 0.0), chi, psi),
    chi,
    psi,
    order: two_sum(lambda, -0.5),
    r0,
    omega,
    debye: debye_log_sum(lambda.abs(), r0),
  }
}

/// Where the clock's saddle is past `2^{±512}` or a square of its clock leaves the normal range, the law of `X/2ᵏ` with that
/// saddle near 1, whose squares are below `2r` and move the clock by `ω²/r` if subnormal.
pub(super) fn density(
  (lambda, alpha, beta, delta): (f64, f64, f64, f64),
  d: (f64, f64),
  (gamma, r0): (f64, f64),
  p: i32,
) -> f64 {
  match clock((lambda, alpha, beta, delta), (gamma, r0)) {
    Some(c) => at_saddle((lambda, alpha, beta, delta), d, &c, p),
    None => rescaled((lambda, alpha, beta, delta), d, (gamma, r0), p),
  }
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
  let omega = delta * gamma;
  let (alpha, beta, delta) = (ldexp(alpha, k), ldexp(beta, k), ldexp(delta, -k));
  let scaled = (lambda, alpha, beta, delta);
  let c = clock_at(scaled, (saddle(lambda, delta, ldexp(gamma, k)), r0, omega));
  at_saddle(scaled, (ldexp(d.0, -k), ldexp(d.1, -k)), &c, p - k)
}

/// `βd + ln N(λ − ½, δ² + d², α²) − ln N(λ, δ², γ²)` meets as `H_λ(w₁) − H_λ(w₀) − ½ ln w₁ − (d − βw₁)²/2w₁`
/// (`α² = γ² + β²`), both maximisers and `d` in two parts, plus `T₁ − T₀`, times `2ᵖ`.
pub(super) fn at_saddle(
  (lambda, alpha, beta, delta): (f64, f64, f64, f64),
  d: (f64, f64),
  c: &Clock,
  p: i32,
) -> f64 {
  let (root, dd) = (delta.hypot(d.0), mul_parts(d, d));
  // Past `|d| = 2^{512}` the point lies `2^{256}` deviations off a saddle within `2^{±512}`.
  if !dd.0.is_finite() {
    return 0.0;
  }
  let (chi, psi, order, w0, r0, omega) = (c.chi, c.psi, c.order, c.w0, c.r0, c.omega);
  let w1 = saddle(order.0, root, alpha);
  // Past the normal range `w₁` puts `d` beyond the law's reach: the divergence from `w₀` passes `e^{−10⁴}`.
  if !w1.is_normal() {
    return 0.0;
  }
  let w1 = saddle_parts(w1, order, add_parts(chi, dd), two_prod(alpha, alpha));
  let (bw, bw_lo) = two_prod(beta, w1.0);
  let (g, g_lo) = two_sum_far(d.0, -bw);
  let gap = g + (g_lo + d.1 - bw_lo - beta * w1.1);
  let r1 = order.0.hypot(alpha * root);
  let a = divergence(lambda, (chi.0, psi.0), w0.0, offset(w1, w0)) - 0.5 * (gap / w1.0) * gap
    + debye_log_sum(order.0.abs(), r1)
    - c.debye
    - 0.5 * std::f64::consts::TAU.ln();
  // Where a root passes MAX its ratio in quarter units, the Debye sums past `1/r ≈ 0` already 1.
  let ratio = if r0.is_finite() && r1.is_finite() {
    r0 / r1
  } else {
    let r0 = (0.25 * lambda).hypot(0.25 * omega);
    let (_, r1) = quarter(order.0, (alpha, root));
    r0 / r1
  };
  exp_times_over_pow2(a, (ratio.sqrt(), w1.0.sqrt()), p)
}

/// `N(βw, w + β²w²/r)` at `d`: within `1/r` of the clock's mean and variance, `w` and `r` from
/// the parameters in two parts, so a mean at a double keeps `d − βw`'s digits; all at `2⁻ᵏ`.
#[cold]
#[inline(never)]
pub(super) fn normal_clock(
  (lambda, alpha, beta, delta): (f64, f64, f64, f64),
  d: (f64, f64),
  p: i32,
) -> f64 {
  let chi = Scaled::new(delta).mul(Scaled::new(delta));
  let psi = positive_sum(alpha, -beta).mul(positive_sum(alpha, beta));
  let (w, r) = exact_saddle(lambda, chi, psi);
  let tilt = (beta != 0.0).then(|| Scaled::new(beta.abs()).mul(w));
  let variance = tilt.map_or(w, |t| w.add(t.mul(t).div(r)));
  let k = tilt
    .map_or(0, |t| t.exponent() - 1021)
    .max(i32::from(d.0.abs() >= HALF_TOP));
  let (d, d_lo) = (ldexp(d.0, -k), ldexp(d.1, -k));
  let gap = tilt.map_or((d, d_lo), |t| {
    let (t, t_lo) = t.shift(-k).value_parts();
    let (g, g_lo) = two_sum(d, -beta.signum() * t);
    two_sum(g, g_lo + (d_lo - beta.signum() * t_lo))
  });
  let spread = variance.sqrt();
  let (e, m) = (spread.exponent(), spread.shift(-spread.exponent()).value());
  let half_square = if gap.0 == 0.0 {
    (0.0, 0.0)
  } else {
    let z = Scaled::parts((gap.0.abs(), gap.1 * gap.0.signum())).div(spread.shift(-k));
    let (zz, zz_lo) = z.mul(z).value_parts();
    (-0.5 * zz, -0.5 * zz_lo)
  };
  exp_parts(half_square, 1.0 / (SQRT_TAU * m), p - e)
}

/// `a + b > 0` in two parts and a power of two, from the halves where the sum may pass MAX.
fn positive_sum(a: f64, b: f64) -> Scaled {
  if a.abs().max(b.abs()) < HALF_TOP {
    Scaled::parts(two_sum(a, b))
  } else {
    Scaled::parts(two_sum(0.5 * a, 0.5 * b)).shift(1)
  }
}
