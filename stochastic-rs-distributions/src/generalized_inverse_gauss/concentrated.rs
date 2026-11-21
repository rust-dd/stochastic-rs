//! The density from `r = √(λ² + χψ) = 50` at the clock's saddle; from `r = 2¹⁰⁰` or a saddle
//! past the normal range, the law of `X/2ᵏ`, its saddle in `[1, 2)`, in quarter units.

use std::f64::consts::LN_2;
use std::f64::consts::PI;

use crate::special::Scaled;
use crate::special::debye_log_sum;
use crate::special::debye_log_sum_quarter;
use crate::special::divergence;
use crate::special::exact_saddle;
use crate::special::exp_times_over;
use crate::special::exp_times_over_pow2;
use crate::special::ldexp;
use crate::special::offset;
use crate::special::quarter;
use crate::special::saddle;
use crate::special::saddle_parts;

/// `2¹⁰⁰`: from this `r` the plain saddle's `ε²` from `√χ` and `√ψ`, times the divergence's
/// curvature `r`, nears `ε` within the law's width.
const EXACT_FROM: f64 = 1.267_650_600_228_229_4e30;

/// `ln f = D_λ(x/w − 1) − ln x − ln 2 − T`, `e^{−T} = √(2r/π)/S`: the `O(r)` terms meet in the divergence.
pub(super) fn density(
  lambda: f64,
  (chi, psi): (f64, f64),
  (root_chi, root_psi): (f64, f64),
  r: f64,
  x: f64,
) -> f64 {
  let w = saddle(lambda, root_chi, root_psi);
  if !(w.is_normal() && r < EXACT_FROM) {
    return rescaled(lambda, (chi, psi), (root_chi, root_psi), x);
  }
  let w_parts = saddle_parts(w, (lambda, 0.0), (chi, 0.0), (psi, 0.0));
  let a = divergence(lambda, (chi, psi), w, offset((x, 0.0), w_parts))
    - LN_2
    - debye_log_sum(lambda.abs(), r);
  exp_times_over(a, (2.0 * r / PI).sqrt(), x)
}

/// [`density`] on `X/2ᵏ ~ GIG(λ, χ2⁻ᵏ, ψ2ᵏ)`, `χ'/w'` and `ψ'w'` being `ω²/Q` and `Q = |λ| + r`
/// (swapped at `λ < 0`): `(λ, χ', ψ')/4` holds the saddle and a quarter of the divergence.
#[cold]
#[inline(never)]
fn rescaled(lambda: f64, (chi, psi): (f64, f64), (root_chi, root_psi): (f64, f64), x: f64) -> f64 {
  let w = exact_saddle_of(lambda, chi, psi);
  let k = w.exponent();
  let (m, m_lo) = w.shift(-k).value_parts();
  let at = ldexp(x, -k);
  // `x/w` under `2⁻¹⁰⁷⁴` puts the divergence past `−10⁴` (its `χ/2x` or `λ ln(x/w)`), the prefactor under `e^{1100}`.
  if at == 0.0 {
    return 0.0;
  }
  let (l, r) = quarter(lambda, (root_chi, root_psi));
  let reach = l.abs() + r;
  let omega = (0.5 * root_chi) * (0.5 * root_psi);
  let half_c = omega * (omega / reach);
  let (chi, psi) = if lambda >= 0.0 {
    (half_c * m, reach / m)
  } else {
    (reach * m, half_c / m)
  };
  let a = 4.0 * divergence(l, (chi, psi), m, offset((at, 0.0), (m, m_lo)))
    - LN_2
    - debye_log_sum_quarter(l.abs(), r);
  exp_times_over_pow2(a, (2.0 * (2.0 * r / PI).sqrt(), at), -k)
}

/// [`exact_saddle`], or where `χ` or `ψ` is 0 the gamma or inverse-gamma clock's `2λ/ψ` or `χ/2|λ|`.
fn exact_saddle_of(lambda: f64, chi: f64, psi: f64) -> Scaled {
  if chi == 0.0 || psi == 0.0 {
    let two_nu = Scaled::new(lambda.abs()).shift(1);
    return if lambda > 0.0 {
      two_nu.div(Scaled::new(psi))
    } else {
      Scaled::new(chi).div(two_nu)
    };
  }
  exact_saddle(lambda, Scaled::new(chi), Scaled::new(psi)).0
}
