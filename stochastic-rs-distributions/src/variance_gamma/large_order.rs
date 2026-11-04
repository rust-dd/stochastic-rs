//! The density from order `1/ν − ½ = 50` with its exponent summed in two parts past `|ln f| + |ln σ| = 512` or where the
//! clock's saddle `w = 1 + g` falls below `⅔`, there the plain sum's rounding of its large terms or of `g` nearing `1e-12`.

use crate::special::HALF_LN_TAU;
use crate::special::LN2_HI;
use crate::special::LN2_LO;
use crate::special::Scaled;
use crate::special::add_parts;
use crate::special::debye_shape_parts;
use crate::special::exp_parts;
use crate::special::ln_parts;
use crate::special::mul_parts;
use crate::special::stirling_tail;
use crate::special::sub_parts;
use crate::special::two_sum;

/// Up to this `|g|` `ln_1p_mx(g)` is summed as its series, past it as `ln w − g`.
const SERIES_BELOW: f64 = 9.765_625e-4;

/// `(a + a_lo)/(b + b_lo)` in two parts.
fn over((a, a_lo): (f64, f64), (b, b_lo): (f64, f64)) -> (f64, f64) {
  let q = a / b;
  two_sum(q, ((-q).mul_add(b, a) + (a_lo - q * b_lo)) / b)
}

/// The density of `large_order_pdf` with each term in two parts at `d = x − μ` in two parts, `ln(1 + g)` and `ln_1p_mx(g)`
/// from `ln w` in two parts past `|g| = 2⁻¹⁰`, `w = 1 + g` the clock's saddle, and below it from their series in `g`.
#[cold]
#[inline(never)]
pub(super) fn density((sigma, nu, theta): (f64, f64, f64), d: (f64, f64)) -> f64 {
  let order = add_parts((1.0 / nu, (-1.0 / nu).mul_add(nu, 1.0) / nu), (-0.5, 0.0));
  let two_over_nu = Scaled::new(2.0).div(Scaled::new(nu));
  let s = Scaled::new(sigma);
  let ratio = (theta != 0.0).then(|| Scaled::new(theta.abs()).div(s));
  let root = (d.0 != 0.0).then(|| Scaled::parts((d.0.abs(), d.1 * d.0.signum())).div(s));
  let c = ratio.map_or((0.0, 0.0), |r| r.mul(r).value_parts());
  let psi = ratio.map_or(two_over_nu, |r| r.mul(r).add(two_over_nu));
  let so = Scaled::parts(order);
  let (r, t) = match root {
    Some(root) => {
      let t = root.mul(psi.sqrt());
      (so.mul(so).add(t.mul(t)).sqrt().value_parts(), t.value())
    }
    None => (order, 0.0),
  };
  let den = add_parts(add_parts(r, order), add_parts((1.0, 0.0), c));
  let near = {
    let (h, h_lo) = two_sum(d.0, -theta);
    two_sum(h, h_lo + d.1)
  };
  let far = {
    let (s, s_lo) = two_sum(d.0, theta);
    two_sum(s, s_lo + d.1)
  };
  // `g = (χ − c − 1)/D` itself, `χ − c = (d − θ)(d + θ)/σ²`, so a small `g` keeps its relative digits, which `1/ν` multiplies.
  let g = over(
    add_parts(
      mul_parts(over(near, (sigma, 0.0)), over(far, (sigma, 0.0))),
      (-1.0, 0.0),
    ),
    den,
  );
  let w = add_parts(g, (1.0, 0.0));
  let (mx, ln_w) = if g.0.abs() > SERIES_BELOW {
    let ln_w = ln_parts(w.0, w.1, 0);
    (sub_parts(ln_w, g), ln_w)
  } else {
    // `ln_1p_mx(g) = −½g²Σ 2(−g)ᵏ/(k + 2)`, its first term in two parts and the rest under `2⁻¹⁰` of it.
    let (mut tail, mut power) = (0.0, 1.0);
    for k in 1..8 {
      power *= -g.0;
      tail += 2.0 * power / f64::from(k + 2);
    }
    let mx = mul_parts(mul_parts(g, g), two_sum(-0.5, -0.5 * tail));
    (mx, add_parts(g, mx))
  };
  let signed = |v: Option<Scaled>, sign: f64| {
    v.map_or((0.0, 0.0), |v| {
      let (hi, lo) = v.value_parts();
      (sign * hi, sign * lo)
    })
  };
  let (theta_s, d_s) = (signed(ratio, theta.signum()), signed(root, d.0.signum()));
  let q = (theta / sigma) * ((d.0 + theta) / sigma) / den.0;
  // Past `q = θ(d + θ)/σ²D = ½` the gap `(d − θ) − θg` cancels; `((d − θ)(o + 1 + Δ) + θ)/D`, `Δ = R − θd/σ²`, does not.
  let gap = if q < 0.5 {
    over(sub_parts(near, mul_parts((theta, 0.0), g)), (sigma, 0.0))
  } else {
    let tilt = over(mul_parts(theta_s, d_s), r);
    // `Δ = (o² + 2χ/ν)/(R + θd/σ²)` where `θd > 0`, from `R² − (θd/σ²)² = o² + 2χ/ν`.
    let excess = if tilt.0 > 0.0 {
      let root_r = over(d_s, r);
      let squares = add_parts(
        mul_parts(over(order, r), order),
        mul_parts(mul_parts(root_r, d_s), two_over_nu.value_parts()),
      );
      over(squares, add_parts((1.0, 0.0), tilt))
    } else {
      mul_parts(r, sub_parts((1.0, 0.0), tilt))
    };
    let ratio_d = over(add_parts(add_parts(order, (1.0, 0.0)), excess), den);
    add_parts(
      mul_parts(over(near, (sigma, 0.0)), ratio_d),
      over(theta_s, den),
    )
  };
  let quadratic = over(mul_parts(gap, gap), w);
  let rests = add_parts(
    debye_shape_parts(order.0, t),
    (-0.5 * (-0.5 * nu).ln_1p() - stirling_tail(nu), 0.0),
  );
  let ln_sigma = ln_parts(sigma, 0.0, 0);
  let a = add_parts(
    add_parts(sub_parts((LN2_HI, LN2_LO), ln_sigma), over(mx, (nu, 0.0))),
    add_parts(
      (-0.5 * ln_w.0, -0.5 * ln_w.1),
      add_parts(
        (-0.5 * quadratic.0, -0.5 * quadratic.1),
        sub_parts(rests, (2.0 * HALF_LN_TAU.0, 2.0 * HALF_LN_TAU.1)),
      ),
    ),
  );
  exp_parts(a, 1.0, 0)
}
