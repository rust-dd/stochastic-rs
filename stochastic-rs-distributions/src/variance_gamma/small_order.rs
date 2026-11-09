//! The density below order `1/ν − ½ = 50` at large logs or where the Bessel argument is not normal:
//! logs, `1/ν` and the order in two parts over scaled `K`, `root` and `|d|·root/σ²` scaled.

use crate::special::HALF_LN_TAU;
use crate::special::LN2_HI;
use crate::special::LN2_LO;
use crate::special::Scaled;
use crate::special::add_parts;
use crate::special::digamma;
use crate::special::exp_parts;
use crate::special::ln_bessel_ke_parts;
use crate::special::ln_half_gamma_parts;
use crate::special::ln_parts;
use crate::special::mul_parts;
use crate::special::sub_parts;

/// `ln(½Γ(s))` at `s` in two parts, the low part through `ψ(s)`, below ½ as `ψ(1 + s) − 1/s`, since `ψ(2⁻¹⁰²⁴)` overflows.
fn ln_half_gamma((s, s_lo): (f64, f64)) -> (f64, f64) {
  let slope = if s < 0.5 {
    digamma(1.0 + s) * s_lo - s_lo / s
  } else {
    digamma(s) * s_lo
  };
  add_parts(ln_half_gamma_parts(s), (slope, 0.0))
}

/// Madan–Carr–Chang eq. 23 in two parts, at `d = 0` the `y^aK_a(y)` limit `2^{a−1}Γ(a)` (`a > 0`);
/// `θd/σ² − |d|·root/σ²` as its one-sign `−2|d|/(ν(root + |θ|)) − |d|(|θ| − θ sgn d)/σ²`.
#[cold]
#[inline(never)]
pub(super) fn density((sigma, nu, theta): (f64, f64, f64), d: (f64, f64), p: i32) -> f64 {
  if d.0.is_nan() {
    return d.0;
  }
  let shape = (1.0 / nu, (-1.0 / nu).mul_add(nu, 1.0) / nu);
  let order = add_parts(shape, (-0.5, 0.0));
  let (ln_sigma, ln_nu) = (ln_parts(sigma, 0.0, 0), ln_parts(nu, 0.0, 0));
  let norm = add_parts(
    add_parts(ln_sigma, HALF_LN_TAU),
    add_parts(mul_parts(shape, ln_nu), ln_half_gamma(shape)),
  );
  let (ss, sn, st) = (
    Scaled::new(sigma),
    Scaled::new(nu),
    Scaled::new(theta.abs()),
  );
  let sigma2 = ss.mul(ss);
  let spread = sigma2.mul(Scaled::new(2.0)).div(sn);
  let root = if theta == 0.0 {
    spread.sqrt()
  } else {
    spread.add(st.mul(st)).sqrt()
  };
  if d.0 == 0.0 {
    let ln_ratio = add_parts(
      (LN2_HI, LN2_LO),
      mul_parts((2.0, 0.0), sub_parts(ln_sigma, root.ln())),
    );
    let at_mu = add_parts(ln_half_gamma(order), mul_parts(order, ln_ratio));
    return exp_parts(sub_parts(at_mu, norm), 1.0, p);
  }
  let sign = if d.0 < 0.0 { -1.0 } else { 1.0 };
  let far = Scaled::parts((d.0.abs(), sign * d.1));
  let near = if theta == 0.0 { root } else { root.add(st) };
  let mut drop = far.mul(Scaled::new(2.0)).div(sn.mul(near)).value_parts();
  if theta * sign < 0.0 {
    drop = add_parts(
      drop,
      far.mul(st).mul(Scaled::new(2.0)).div(sigma2).value_parts(),
    );
  }
  if !drop.0.is_finite() {
    return 0.0;
  }
  let power = mul_parts(order, sub_parts(far.ln(), root.ln()));
  let kernel = sub_parts(ln_bessel_ke_parts(order, far.mul(root).div(sigma2)), drop);
  exp_parts(sub_parts(add_parts(power, kernel), norm), 1.0, p)
}

/// `ln b` where `θνt` or `½σ²νt²` overflows: `b = 1 − νt(θ + ½σ²t)`, the bracket in two parts at
/// one power of two, `νt` times it scaled, its `ln` once past `2⁶⁰`; `None` at `b ≤ 0`.
#[cold]
#[inline(never)]
pub(super) fn ln_base_far((sigma, nu, theta): (f64, f64, f64), t: f64) -> Option<f64> {
  if t == 0.0 {
    return Some(0.0);
  }
  let spread = Scaled::new(sigma)
    .mul(Scaled::new(sigma))
    .mul(Scaled::new(t.abs()))
    .div(Scaled::new(2.0));
  let k = if theta == 0.0 {
    spread.exponent()
  } else {
    spread.exponent().max(Scaled::new(theta.abs()).exponent())
  };
  let (s, s_lo) = spread.shift(-k).value_parts();
  let bracket = add_parts(
    (crate::special::ldexp(theta, -k), 0.0),
    (t.signum() * s, t.signum() * s_lo),
  );
  if bracket.0 == 0.0 {
    return Some(0.0);
  }
  let product = Scaled::new(nu)
    .mul(Scaled::new(t.abs()))
    .mul(Scaled::parts((
      bracket.0.abs(),
      bracket.1 * bracket.0.signum(),
    )))
    .shift(k);
  let rising = (t < 0.0) != (bracket.0 < 0.0);
  if product.exponent() > 60 {
    return rising.then(|| {
      let (h, h_lo) = product.ln();
      h + h_lo
    });
  }
  let (v, v_lo) = product.value_parts();
  let sign = if rising { 1.0 } else { -1.0 };
  let (b, b_lo) = add_parts((1.0, 0.0), (sign * v, sign * v_lo));
  (b > 0.0).then(|| {
    let (h, h_lo) = ln_parts(b, b_lo, 0);
    h + h_lo
  })
}
