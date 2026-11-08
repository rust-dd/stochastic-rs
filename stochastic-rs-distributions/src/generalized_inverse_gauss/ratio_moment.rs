//! `E Xᵏ` of GIG(λ, χ, ψ) with an end of its orders under 50: the product of the Bessel ratios over the orders under 50,
//! a run of orders past it as one saddle step, so no `K` and no logarithm of one is formed, at a cost bounded in `k`.
//! Reference: NIST Digital Library of Mathematical Functions, eq. 10.29.1 (`K`'s recurrence, carried as a ratio), <https://dlmf.nist.gov/10.29.1>.

use std::f64::consts::LN_2;

use crate::special::LARGE_ORDER;
use crate::special::Scaled;
use crate::special::k_half_ratio;
use crate::special::ldexp;
use crate::special::ln_two_over;
use crate::special::order_step;
use crate::special::scaled_saddle;
use crate::special::split_pow2;
use crate::special::two_sum;

/// `E Xᵏ = 2ᵉm` as `(m, e)`: for `|ν| < 50` the factor `2g_ν/ψ` from order `−½` on and `χ/(2g_{−ν−1})` below, each
/// `(χ/ψ)^{1/2}R_ν`, `g_ν = (ω/2)R_ν` from `k_half_ratio`; the exponents are kept apart, so no factor leaves the doubles.
#[inline]
pub(super) fn ratio_moment(lambda: f64, roots: (f64, f64), k: u32) -> (f64, i32) {
  let (root_chi, root_psi) = roots;
  let omega = root_chi * root_psi;
  // `ln(2/ω)`, which only Temme's sums below `ω = 2` read.
  let d = if omega < 2.0 {
    ln_two_over(root_chi, root_psi, omega)
  } else {
    0.0
  };
  let (kc, mc) = split_pow2(root_chi);
  let (kp, mp) = split_pow2(root_psi);
  let mut value = (1.0, 0_i64);
  let (mut nu, mut left) = (lambda, f64::from(k));
  if nu <= -LARGE_ORDER {
    let run = (-LARGE_ORDER - nu).floor().min(left);
    saddle_run(&mut value, nu, run, roots);
    nu += run;
    left -= run;
  }
  while left > 0.0 && nu < LARGE_ORDER {
    if nu >= -0.5 {
      let (j, g) = k_half_ratio(nu, omega, d);
      value.0 *= g / (mp * mp);
      value.1 += i64::from(j + 1 - 2 * kp);
    } else {
      let (j, g) = k_half_ratio(-nu - 1.0, omega, d);
      value.0 *= mc * mc / g;
      value.1 += i64::from(2 * kc - 1 - j);
    }
    nu += 1.0;
    left -= 1.0;
  }
  if left > 0.0 {
    saddle_run(&mut value, nu, left, roots);
  }
  let (j, m) = split_pow2(value.0);
  (m, (value.1 + i64::from(j)).clamp(-4000, 4000) as i32)
}

/// `N(ν + run)/N(ν) = wʳᵘⁿe^{rest}` from `order_step`, both orders past 50 on one side, with `w = 2ᵃb` split so only
/// `run·ln b + rest` is exponentiated.
fn saddle_run(value: &mut (f64, i64), nu: f64, run: f64, (root_chi, root_psi): (f64, f64)) {
  if run <= 0.0 {
    return;
  }
  let (_, rest) = order_step(nu, run, (root_chi, root_psi));
  let to = nu + run;
  let (b, a) = scaled_saddle(to, to.hypot(root_chi * root_psi), (root_chi, root_psi));
  let ln = run * b.ln() + rest;
  let n = (ln / LN_2).round();
  value.0 *= (ln - n * LN_2).exp();
  value.1 += i64::from(a) * run as i64 + n as i64;
}

/// `factor·E Xᵏ` from the ratios' limits `g_ν → ν` where each holds within `2⁻¹¹²`: `2ν/ψ` above
/// order `−½`, `χ/(2ν)` below, from exact `χ` and `ψ`, rounded once; `None` elsewhere.
#[cold]
#[inline(never)]
pub(super) fn limit_moment(
  lambda: f64,
  (chi, psi): (f64, f64),
  k: u32,
  factor: f64,
) -> Option<f64> {
  if !(chi > 0.0 && psi > 0.0 && factor != 0.0 && factor.is_finite()) {
    return None;
  }
  // `log₂(ω/2)` from the squares, which the rounded `ω` on the subnormal grid would not give.
  let half_ln = 0.5 * (chi.log2() + psi.log2()) - 1.0;
  let (mut value, mut lower, mut upper) = (Scaled::new(factor.abs()), false, false);
  for i in 0..k {
    let nu = two_sum(lambda, f64::from(i));
    let (order, up) = if nu.0 >= -0.5 {
      (nu, true)
    } else {
      (two_sum(-nu.0 - 1.0, -nu.1), false)
    };
    let g = order.0;
    // `g_ν − ν`: under `(2 + 1/(1 − ν))(ω/2)^{2ν}ν` below order 1, `(ω/2)²/g_{ν−1}` from it.
    let bound = if g < 1.0 {
      2.0 * g * half_ln + (2.0 + 1.0 / (1.0 - g)).log2()
    } else {
      2.0 * half_ln + (1.0 / (g - 1.0)).min(2.0 * (1.0 - half_ln * LN_2)).log2()
    };
    if !(g > 0.0 && bound < -112.0) {
      return None;
    }
    let twice = Scaled::parts(order).shift(1);
    value = if up {
      upper = true;
      value.mul(twice).div(Scaled::new(psi))
    } else {
      lower = true;
      value.mul(Scaled::new(chi)).div(twice)
    };
  }
  // A limit at a midpoint of the subnormal grid: `g_ν > ν` puts every term below (above) its limit.
  let bias = match (lower, upper) {
    (true, false) => -1.0,
    (false, true) => 1.0,
    _ => 0.0,
  };
  Some(factor.signum() * round_once(value, bias))
}

/// `s` rounded once onto the doubles: its low part, or `bias` where that is 0, decides a subnormal
/// tie that the high part alone would round to even.
fn round_once(s: Scaled, bias: f64) -> f64 {
  let e = s.exponent();
  let (m, lo) = s.shift(-e).value_parts();
  let r = ldexp(m, e);
  if e >= -1022 || !r.is_finite() {
    return r;
  }
  let back = ldexp(r, -e);
  let d = m - back;
  let sign = if lo != 0.0 { lo.signum() } else { bias };
  if d.abs() == ldexp(1.0, -1075 - e) && sign == d.signum() {
    return ldexp(back + 2.0 * d, e);
  }
  r
}
