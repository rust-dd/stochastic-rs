//! `E Xᵏ` of GIG(λ, χ, ψ) with an end of its orders under 50: the product of the Bessel ratios over the orders under 50,
//! a run of orders past it as one saddle step, so no `K` and no logarithm of one is formed, at a cost bounded in `k`.
//! Reference: NIST Digital Library of Mathematical Functions, eq. 10.29.1 (`K`'s recurrence, carried as a ratio), <https://dlmf.nist.gov/10.29.1>.

use std::f64::consts::LN_2;

use crate::special::LARGE_ORDER;
use crate::special::k_half_ratio;
use crate::special::ln_two_over;
use crate::special::order_step;
use crate::special::scaled_saddle;
use crate::special::split_pow2;

/// `E Xᵏ = 2ᵉm` as `(m, e)`: for `|ν| < 50` the factor `2g_ν/ψ` from order `−½` on and `χ/(2g_{−ν−1})` below, each
/// `(χ/ψ)^{1/2}R_ν`, `g_ν = (ω/2)R_ν` from `k_half_ratio`; the exponents are kept apart, so no factor leaves the doubles.
pub(super) fn ratio_moment(lambda: f64, roots: (f64, f64), k: u32) -> (f64, i32) {
  let (root_chi, root_psi) = roots;
  let omega = root_chi * root_psi;
  let d = ln_two_over(root_chi, root_psi, omega);
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
