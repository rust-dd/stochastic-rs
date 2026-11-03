//! GIG normalisers `N(λ, χ, ψ) = ∫ w^{λ−1}e^{−(χ/w + ψw)/2} dw = 2(χ/ψ)^{λ/2}K_λ(√(χψ))` from order 50 on as `ln 2 + H_λ(w*)`
//! plus Debye's remainder, `H_λ(w) = λ ln w − ½(χ/w + ψw)` at its maximiser, so a ratio of two meets as one divergence.
//! Reference: NIST Digital Library of Mathematical Functions, eq. 10.41.4 (Debye's expansion of `K_ν(νz)`, whose `−νη` with the power is `H_λ(w*)`), <https://dlmf.nist.gov/10.41.4>.

use super::bessel_i::debye_sum_at;
pub(crate) use super::bessel_k::debye_shape;
use super::twofold::two_prod;
use super::twofold::two_sum;

/// From this `√(ν² + x²)` on Debye's expansion holds to the last bit at any order `ν`; the GH, GIG and VG analytics take their
/// saddle forms there.
pub(crate) const LARGE_ORDER: f64 = 50.0;

/// The maximiser of `H_λ` from `√χ` and `√ψ`, `(λ + √(λ² + χψ))/ψ`, at `λ < 0` as `χ/(|λ| + √(λ² + χψ))`, which does not
/// cancel; neither squares a root, so it holds where `χ` or `ψ` would overflow.
pub(crate) fn saddle(lambda: f64, root_chi: f64, root_psi: f64) -> f64 {
  let root = lambda.hypot(root_chi * root_psi);
  if lambda >= 0.0 {
    (lambda + root) / root_psi / root_psi
  } else {
    root_chi * (root_chi / (root - lambda))
  }
}

/// The maximiser `w ≈ guess` of `H_λ` in two parts, `λ`, `χ` and `ψ` each given in two: a Newton step on `ψw − 2λ − χ/w`, its
/// residual formed exactly, so an offset from `w` keeps its digits where the law is narrower than `w`'s rounding.
pub(crate) fn saddle_parts(
  guess: f64,
  lambda: (f64, f64),
  chi: (f64, f64),
  psi: (f64, f64),
) -> (f64, f64) {
  let w = guess;
  let (p, p_lo) = two_prod(psi.0, w);
  let q = chi.0 / w;
  let q_lo = ((-q).mul_add(w, chi.0) + chi.1) / w;
  let (s, s_lo) = two_sum(p, -2.0 * lambda.0);
  let (g, g_lo) = two_sum(s, -q);
  let residual = g + (g_lo + s_lo + p_lo + psi.1 * w - 2.0 * lambda.1 - q_lo);
  // The slope `ψ + χ/w²` times `w`, which stays finite where `χ/w²` would not.
  let step = residual / (p + q) * w;
  if step.is_finite() {
    (w, -step)
  } else {
    (w, 0.0)
  }
}

/// `x/w − 1` for `x` and `w` in two parts, the difference formed exactly.
pub(crate) fn offset((x, x_lo): (f64, f64), (w, w_lo): (f64, f64)) -> f64 {
  let (d, d_lo) = two_sum(x, -w);
  (d + (d_lo + (x_lo - w_lo))) / w
}

/// `ln S` of Debye's `S = Σ(−1)^k U_k(p)/ν^k`, `p = ν/r`, `r = √(ν² + x²)`: `ln K_ν(x) = ½ ln(π/2r) − r + ν·asinh(ν/x) + ln S`, taken
/// in `1/r` so it holds at `ν = 0`.
pub(crate) fn debye_log_sum(nu: f64, r: f64) -> f64 {
  debye_sum_at(nu / r, -1.0 / r).ln()
}

/// `e^a·m/x`, the factor apart where it is normal and `e^a` too, so a large `|ln x|` does not round into the exponent.
pub(crate) fn exp_times_over(a: f64, m: f64, x: f64) -> f64 {
  let factor = m / x;
  if factor.is_normal() && a > -700.0 {
    a.exp() * factor
  } else {
    (a + m.ln() - x.ln()).exp()
  }
}

/// `H_λ(w(1 + e)) − H_λ(w)` at the maximiser `w`, in terms of one sign: `λ·ln1pmx(e) − (χ/2w)e²/(1 + e)` at `λ ≥ 0`, else
/// `λ·ℓ(e) − (ψw/2)e²/(1 + e)`, `ℓ(e) = −ln1pmx(−e/(1 + e))`, both from the saddle equation `λ + χ/2w = ψw/2`.
pub(crate) fn divergence(lambda: f64, (chi, psi): (f64, f64), w: f64, e: f64) -> f64 {
  let q = e * e / (1.0 + e);
  if lambda >= 0.0 {
    lambda * ln_1p_mx(e) - 0.5 * chi / w * q
  } else {
    -lambda * ln_1p_mx(-e / (1.0 + e)) - 0.5 * psi * w * q
  }
}

/// `ln(1 + x) − x` for `x > −1`: `−x²/(2 + x) + 2s³ Σ s^{2j}/(2j + 3)`, `s = x/(2 + x)`, where `|s| ≤ 0.2`, else directly.
pub(crate) fn ln_1p_mx(x: f64) -> f64 {
  let s = x / (2.0 + x);
  if s.abs() > 0.2 {
    return x.ln_1p() - x;
  }
  let s2 = s * s;
  // To `j = 12`: at `s² ≤ 0.04` the next term is under `1e-18` of the sum.
  let sum = [
    1.0 / 27.0,
    1.0 / 25.0,
    1.0 / 23.0,
    1.0 / 21.0,
    1.0 / 19.0,
    1.0 / 17.0,
    1.0 / 15.0,
    1.0 / 13.0,
    1.0 / 11.0,
    1.0 / 9.0,
    1.0 / 7.0,
    0.2,
    1.0 / 3.0,
  ]
  .iter()
  .fold(0.0_f64, |acc, &c| acc.mul_add(s2, c));
  -x * x / (2.0 + x) + 2.0 * s * s2 * sum
}

/// Debye's remainder at order `to` less that at `from`, the `½ ln ν` parts by `ln_1p` so two near orders keep their gap.
fn rest_step(from: f64, to: f64, (x_from, x_to): (f64, f64)) -> f64 {
  debye_shape(to, x_to) - debye_shape(from, x_from) - 0.5 * ((to - from) / from).ln_1p()
}

/// `ln N(λ + k, χ, ψ) − ln N(λ, χ, ψ)`, `N(λ, χ, ψ) = 2(χ/ψ)^{λ/2}K_λ(√(χψ))`, both orders past `LARGE_ORDER`:
/// `H_λ(w_k) + k ln w_k − H_λ(w)` as a divergence, `w_k` the maximiser at `λ + k`, so `ln E X^k` of the GIG keeps its digits.
pub(crate) fn ln_order_step(lambda: f64, k: f64, (chi, psi): (f64, f64)) -> f64 {
  let (rc, rp) = (chi.sqrt(), psi.sqrt());
  let (w, wk) = (saddle(lambda, rc, rp), saddle(lambda + k, rc, rp));
  let omega = rc * rp;
  divergence(lambda, (chi, psi), w, wk / w - 1.0)
    + k * wk.ln()
    + rest_step(lambda.abs(), (lambda + k).abs(), (omega, omega))
}

/// `ln N(λ, χ, ψ − 2s) − ln N(λ, χ, ψ)` for `ψ − 2s = shifted ≥ 0`, `|λ|` past `LARGE_ORDER`: with `Q = |λ| + √(λ² + χψ)`
/// each maximum is `λ ln(Q/ψ) − Q + λ` (`|λ| ln(Q/χ) − Q + |λ|` at `λ < 0`), so the step is `λ⁺ ln(ψ/ψ_s) + |λ| ln_1p(ΔQ/Q) − ΔQ`.
pub(crate) fn ln_shift(lambda: f64, s: f64, (chi, psi, shifted): (f64, f64, f64)) -> f64 {
  let nu = lambda.abs();
  let (omega, omega_s) = (chi.sqrt() * psi.sqrt(), chi.sqrt() * shifted.sqrt());
  let (r, r_s) = (nu.hypot(omega), nu.hypot(omega_s));
  let dq = -2.0 * s * chi / (r + r_s);
  // `ln(ψ/ψ_s)` from whichever of `2s` and `ψ_s` is the smaller part of `ψ`, both given exactly.
  let power = if lambda <= 0.0 {
    0.0
  } else if 4.0 * s < psi {
    -lambda * (-2.0 * s / psi).ln_1p()
  } else {
    lambda * (psi / shifted).ln()
  };
  power + nu * (dq / (nu + r)).ln_1p() - dq + debye_shape(nu, omega_s) - debye_shape(nu, omega)
}

/// `ln Γ(1/ν) + (ln ν + 1)/ν − ½ ln(2πν)`, Stirling's remainder in `ν`, `Σ B_{2k}ν^{2k−1}/(2k(2k − 1))`; its first omitted term
/// is under `1e-25` at `ν ≤ 1/50`.
/// Reference: NIST Digital Library of Mathematical Functions, eq. 5.11.1 (Stirling's series), <https://dlmf.nist.gov/5.11.1>.
pub(crate) fn stirling_tail(nu: f64) -> f64 {
  let n2 = nu * nu;
  nu * [
    -691.0 / 360360.0,
    1.0 / 1188.0,
    -1.0 / 1680.0,
    1.0 / 1260.0,
    -1.0 / 360.0,
    1.0 / 12.0,
  ]
  .iter()
  .fold(0.0_f64, |acc, &c| acc.mul_add(n2, c))
}

#[cfg(test)]
mod tests;
