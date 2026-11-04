//! The density's limits where the gamma clock or the diffusion is negligible at double precision: a clock of shape
//! `1/ν ≥ 2¹⁶⁰` is normal, and where `σ²` is under `2⁻¹⁰⁰νθ²` the law is `μ + θG`, so neither squares `θ/σ` nor forms `1/ν`.

use crate::special::HALF_LN_TAU;
use crate::special::Scaled;
use crate::special::add_parts;
use crate::special::exp_parts;
use crate::special::ln_1p_mx;
use crate::special::ln_parts;
use crate::special::mul_parts;
use crate::special::stirling_tail;
use crate::special::sub_parts;
use crate::special::two_sum;

/// From this rate down the clock's skewness `2√ν` moves a representable density by under `2⁻⁶²`.
pub(super) const NORMAL_CLOCK: f64 = 6.842_277_657_836_021e-49;

/// From this `νθ²/σ²` on the diffusion moves a representable density by under `2⁻⁶⁰`: `σ²/(νθ²)` times `1/(νg)`.
pub(super) const NARROW_DIFFUSION: f64 = 1.267_650_600_228_229_4e30;

/// `N(θ, σ² + νθ²)` at `d` in two parts, `|d − θ|` scaled where it passes the doubles, the variance and `(d − θ)/s` scaled.
#[cold]
#[inline(never)]
pub(super) fn normal_clock((sigma, nu, theta): (f64, f64, f64), (d, d_lo): (f64, f64)) -> f64 {
  let ss = Scaled::new(sigma);
  let variance = if theta == 0.0 {
    ss.mul(ss)
  } else {
    let st = Scaled::new(theta.abs());
    ss.mul(ss).add(Scaled::new(nu).mul(st).mul(st))
  };
  let spread = variance.sqrt();
  let (h, h_lo) = two_sum(d, -theta);
  let distance = if !h.is_finite() {
    Some(Scaled::new(d.abs()).add(Scaled::new(theta.abs())))
  } else {
    let near = two_sum(h, h_lo + d_lo);
    (near.0 != 0.0).then(|| Scaled::parts((near.0.abs(), near.1 * near.0.signum())))
  };
  let quadratic = match distance {
    None => (0.0, 0.0),
    Some(distance) => {
      let z = distance.div(spread).value_parts();
      if !(z.0 * z.0).is_finite() {
        return 0.0;
      }
      let (q, q_lo) = mul_parts(z, z);
      (0.5 * q, 0.5 * q_lo)
    }
  };
  exp_parts(
    sub_parts(
      sub_parts((-HALF_LN_TAU.0, -HALF_LN_TAU.1), spread.ln()),
      quadratic,
    ),
    1.0,
    0,
  )
}

/// `f_G(g)/|θ|`, `G ~ Gamma(1/ν, ν)`, `g = d/θ`: `ln_1p_mx(g − 1)/ν − ln g − ½ ln ν − ½ ln 2π` less Stirling's remainder,
/// `g − 1 = (d − θ)/θ`; below `g = ½` the shape as `(1/ν − 1) ln g − (g − 1)/ν` from `ln g = ln|d| − ln|θ|` in two parts.
#[cold]
#[inline(never)]
pub(super) fn gamma_clock((nu, theta): (f64, f64), (d, d_lo): (f64, f64)) -> f64 {
  // Signs, since `dθ` underflows at a narrow law's mode.
  if d == 0.0 || (d < 0.0) != (theta < 0.0) {
    return 0.0;
  }
  let near = two_sum(d, -theta);
  let near = two_sum(near.0, near.1 + d_lo);
  let gm1 = near.0 / theta;
  let gm1 = gm1 + ((-gm1).mul_add(theta, near.0) + near.1) / theta;
  let (lt, ln_nu) = (ln_parts(theta.abs(), 0.0, 0), ln_parts(nu, 0.0, 0));
  let scale = add_parts(add_parts(lt, HALF_LN_TAU), (0.5 * ln_nu.0, 0.5 * ln_nu.1));
  let shape = if gm1 > -0.5 {
    (ln_1p_mx(gm1) / nu - gm1.ln_1p(), 0.0)
  } else {
    let ln_g = sub_parts(ln_parts(d.abs(), d_lo * d.signum(), 0), lt);
    add_parts(mul_parts(ln_g, (1.0 / nu - 1.0, 0.0)), (-gm1 / nu, 0.0))
  };
  if shape.0 == f64::NEG_INFINITY {
    return 0.0;
  }
  exp_parts(
    sub_parts(add_parts(shape, (-stirling_tail(nu), 0.0)), scale),
    1.0,
    0,
  )
}
