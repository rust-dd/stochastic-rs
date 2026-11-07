//! The density and mgf below `r = √(λ² + χψ) = 50` where their logarithms are large or `ω = √(χψ)` leaves the normal range:
//! each logarithm in two parts over `K` in scaled form, so their sum keeps the digits its large terms would round away.

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
use crate::special::two_sum_far;

/// `2¹²⁸`; `χ/x` is finite only for `χ < 4` at a subnormal `x`, so `2¹²⁸χ` is too.
const TWO_128: f64 = 3.402_823_669_209_385e38;

/// `ln K_λ(ω)` in two parts, `ln(e^ω K)` less `ω`.
fn ln_k(lambda: f64, omega: Scaled) -> (f64, f64) {
  sub_parts(
    ln_bessel_ke_parts((lambda, 0.0), omega),
    omega.value_parts(),
  )
}

fn omega(chi: f64, psi: f64) -> Scaled {
  Scaled::new(chi).mul(Scaled::new(psi)).sqrt()
}

/// `½λ ln(ψ/χ) − ln 2 − ln K_λ(ω) + (λ − 1) ln x − ½(χ/x + ψx)` in two parts at `x > 0`.
#[cold]
#[inline(never)]
pub(super) fn density(lambda: f64, chi: f64, psi: f64, x: f64) -> f64 {
  if x.is_nan() {
    return x;
  }
  let q = chi / x;
  let (p, p_lo) = two_prod(psi, x);
  if !(q + p).is_finite() {
    return 0.0;
  }
  // At a subnormal `x` the residual of `χ/x` would round on the subnormal grid; `2¹²⁸x` and `2¹²⁸χ` keep it normal.
  let (xs, cs) = if x.is_normal() {
    (x, chi)
  } else {
    (x * TWO_128, chi * TWO_128)
  };
  let spread = add_parts((q, (-q).mul_add(xs, cs) / xs), (p, p_lo));
  let power = mul_parts(
    (0.5 * lambda, 0.0),
    sub_parts(ln_parts(psi, 0.0, 0), ln_parts(chi, 0.0, 0)),
  );
  let tail = mul_parts(two_sum(lambda, -1.0), ln_parts(x, 0.0, 0));
  let norm = add_parts(ln_k(lambda, omega(chi, psi)), (LN2_HI, LN2_LO));
  let total = sub_parts(
    add_parts(power, tail),
    add_parts(norm, (0.5 * spread.0, 0.5 * spread.1)),
  );
  exp_parts(total, 1.0, 0)
}

/// `(λ/2) ln(ψ/s) + ln K_λ(√(χs)) − ln K_λ(ω)`, `s = ψ − 2t` in two parts, `4(ψ/4 − t/2)` past
/// MAX; the `K`s' logs apart from `ω − √(χs) = √χ·2t/(√ψ + √s)`, which two parts lose there.
#[cold]
#[inline(never)]
pub(super) fn mgf(lambda: f64, chi: f64, psi: f64, t: f64) -> f64 {
  if t.is_nan() {
    return t;
  }
  let shifted = if (psi - 2.0 * t).is_finite() {
    Scaled::parts(two_sum_far(psi, -2.0 * t))
  } else {
    Scaled::parts(two_sum(0.25 * psi, -0.5 * t)).mul(Scaled::new(4.0))
  };
  let power = mul_parts(
    (0.5 * lambda, 0.0),
    sub_parts(ln_parts(psi, 0.0, 0), shifted.ln()),
  );
  let root_chi = Scaled::new(chi).sqrt();
  let gap = root_chi
    .mul(Scaled::new(t.abs()).shift(1))
    .div(Scaled::new(psi).sqrt().add(shifted.sqrt()))
    .value_parts();
  // `√(χs) − ω` past MAX, at `t < 0`, puts the mgf under `e^{−MAX}`.
  if !gap.0.is_finite() {
    return 0.0;
  }
  let dx = if t < 0.0 { (-gap.0, -gap.1) } else { gap };
  let ratio = sub_parts(
    ln_bessel_ke_parts((lambda, 0.0), Scaled::new(chi).mul(shifted).sqrt()),
    ln_bessel_ke_parts((lambda, 0.0), omega(chi, psi)),
  );
  exp_parts(add_parts(power, add_parts(ratio, dx)), 1.0, 0)
}

/// The limit `½Γ(ν)(2/ω)^ν/K_ν(ω)` at `t = ψ/2`, `ν = −λ > 0`, in two parts.
#[cold]
#[inline(never)]
pub(super) fn boundary_mgf(lambda: f64, chi: f64, psi: f64) -> f64 {
  let nu = -lambda;
  let omega = omega(chi, psi);
  let ln_two_over = sub_parts((LN2_HI, LN2_LO), omega.ln());
  let num = add_parts(ln_half_gamma_parts(nu), mul_parts((nu, 0.0), ln_two_over));
  exp_parts(sub_parts(num, ln_k(nu, omega)), 1.0, 0)
}
