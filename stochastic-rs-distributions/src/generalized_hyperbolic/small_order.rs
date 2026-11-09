//! The density and mgf below order 50 where their logs are large or `ω = δγ` leaves the normal
//! range, each log in two parts over scaled `K`; from order 50 the mgf at a subnormal `δγ`.

use crate::special::HALF_LN_TAU;
use crate::special::LN2_HI;
use crate::special::LN2_LO;
use crate::special::Scaled;
use crate::special::add_parts;
use crate::special::exp_parts;
use crate::special::ln_bessel_ke_parts;
use crate::special::ln_half_gamma_parts;
use crate::special::ln_parts;
use crate::special::mul_parts;
use crate::special::sub_parts;
use crate::special::two_prod;
use crate::special::two_sum;

/// `γ_t = √((α − β − t)(α + β + t))` as a scaled value, its factors in two parts.
fn clock_root(alpha: f64, beta: f64, t: f64) -> Scaled {
  let below = add_parts(two_sum(alpha, -beta), (-t, 0.0));
  let above = add_parts(two_sum(alpha, beta), (t, 0.0));
  Scaled::parts(below).mul(Scaled::parts(above)).sqrt()
}

/// `ln K_λ(x)` in two parts.
fn ln_k(order: (f64, f64), x: Scaled) -> (f64, f64) {
  sub_parts(ln_bessel_ke_parts(order, x), x.value_parts())
}

/// `λ ln(γ/δ) − ½ ln 2π − ln K_λ(δγ) − (λ − ½) ln(α/root) + ln K_{λ−½}(α·root) + βd` in two parts
/// at a two-part `d = x − μ`, times `2ᵖ`; `βd − α·root` as `−(α − β sgn d)|d| − αδ²/(root + |d|)`.
#[cold]
#[inline(never)]
pub(super) fn density(
  (lambda, alpha, beta, delta): (f64, f64, f64, f64),
  d: (f64, f64),
  p: i32,
) -> f64 {
  if d.0.is_nan() {
    return d.0;
  }
  let sign = if d.0 < 0.0 { -1.0 } else { 1.0 };
  let far = (d.0.abs(), sign * d.1);
  let along = mul_parts(two_sum(alpha, -beta * sign), far);
  if !along.0.is_finite() {
    return 0.0;
  }
  let (sd, sa) = (Scaled::new(delta), Scaled::new(alpha));
  let root = if far.0 == 0.0 {
    sd
  } else {
    let sf = Scaled::parts(far);
    sd.mul(sd).add(sf.mul(sf)).sqrt()
  };
  let across = if far.0 == 0.0 {
    sa.mul(sd)
  } else {
    sa.mul(sd).mul(sd.div(root.add(Scaled::parts(far))))
  };
  let tilt = add_parts(along, across.value_parts());
  if !tilt.0.is_finite() {
    return 0.0;
  }
  let gamma = clock_root(alpha, beta, 0.0);
  let order = two_sum(lambda, -0.5);
  let ln_delta = ln_parts(delta, 0.0, 0);
  let clock = mul_parts((lambda, 0.0), sub_parts(gamma.ln(), ln_delta));
  let mixing = mul_parts(order, sub_parts(ln_parts(alpha, 0.0, 0), root.ln()));
  let ln_kx = sub_parts(ln_bessel_ke_parts(order, sa.mul(root)), tilt);
  let norm = add_parts(ln_k((lambda, 0.0), sd.mul(gamma)), HALF_LN_TAU);
  exp_parts(
    sub_parts(add_parts(clock, ln_kx), add_parts(norm, mixing)),
    1.0,
    p,
  )
}

/// `μt + λ ln(γ/γ_t) + ln K_λ(δγ_t) − ln K_λ(δγ)` in two parts, the `K`s' logarithms apart from
/// `δγ − δγ_t = δt(2β + t)/(γ + γ_t)`, which their two parts lose at `δγ` near MAX.
#[cold]
#[inline(never)]
pub(super) fn mgf((lambda, alpha, beta, delta, mu): (f64, f64, f64, f64, f64), t: f64) -> f64 {
  let location = two_prod(mu, t);
  if !location.0.is_finite() {
    return location.0.exp();
  }
  let (gamma, gamma_t) = (clock_root(alpha, beta, 0.0), clock_root(alpha, beta, t));
  let sd = Scaled::new(delta);
  let power = mul_parts((lambda, 0.0), sub_parts(gamma.ln(), gamma_t.ln()));
  let lift = two_sum(2.0 * beta, t);
  let gap = if lift.0 == 0.0 {
    (0.0, 0.0)
  } else {
    let sign = (t * lift.0).signum();
    let lift = Scaled::parts((lift.0.abs(), lift.1 * lift.0.signum()));
    let (hi, lo) = sd
      .mul(Scaled::new(t.abs()))
      .mul(lift)
      .div(gamma.add(gamma_t))
      .value_parts();
    (sign * hi, sign * lo)
  };
  // `δγ_t − δγ` past MAX puts the mgf under `e^{−MAX}`.
  if !gap.0.is_finite() {
    return 0.0;
  }
  let ratio = sub_parts(
    ln_bessel_ke_parts((lambda, 0.0), sd.mul(gamma_t)),
    ln_bessel_ke_parts((lambda, 0.0), sd.mul(gamma)),
  );
  exp_parts(
    add_parts(add_parts(location, power), add_parts(ratio, gap)),
    1.0,
    0,
  )
}

/// The limit `e^{μt}½Γ(ν)(2/δγ)^ν/K_ν(δγ)` at `γ_t = 0`, `ν = −λ > 0`, in two parts.
#[cold]
#[inline(never)]
pub(super) fn boundary_mgf(
  (lambda, alpha, beta, delta, mu): (f64, f64, f64, f64, f64),
  t: f64,
) -> f64 {
  let location = two_prod(mu, t);
  if !location.0.is_finite() {
    return location.0.exp();
  }
  let nu = -lambda;
  let omega = Scaled::new(delta).mul(clock_root(alpha, beta, 0.0));
  let ln_two_over = sub_parts((LN2_HI, LN2_LO), omega.ln());
  let num = add_parts(ln_half_gamma_parts(nu), mul_parts((nu, 0.0), ln_two_over));
  exp_parts(
    add_parts(location, sub_parts(num, ln_k((nu, 0.0), omega))),
    1.0,
    0,
  )
}

/// `e^{μt}(ψ/ψ_t)^{λ⁺}` from order 50 at a subnormal `δγ`, the clock's `ω²/|λ|` under the doubles:
/// `ψ_t/ψ = 1 − q`, `q = t(2β + t)/((α − β)(α + β))` scaled, no subnormal square entering.
#[cold]
#[inline(never)]
pub(super) fn subnormal_clock_mgf(
  (lambda, alpha, beta, _, mu): (f64, f64, f64, f64, f64),
  t: f64,
) -> f64 {
  let lift = two_sum(2.0 * beta, t);
  if lambda <= 0.0 || lift.0 == 0.0 {
    return (mu * t).exp();
  }
  let num = Scaled::new(t.abs()).mul(Scaled::parts((lift.0.abs(), lift.1 * lift.0.signum())));
  let den = Scaled::parts(two_sum(alpha, -beta)).mul(Scaled::parts(two_sum(alpha, beta)));
  let q = (t * lift.0).signum() * num.div(den).value();
  let ln_ratio = if q.abs() <= 0.25 {
    -(-q).ln_1p()
  } else {
    let (h, h_lo) = sub_parts(
      clock_root(alpha, beta, 0.0).ln(),
      clock_root(alpha, beta, t).ln(),
    );
    2.0 * (h + h_lo)
  };
  (mu * t + lambda * ln_ratio).exp()
}
