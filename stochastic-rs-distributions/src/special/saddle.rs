//! GIG normalisers `N(λ, χ, ψ) = ∫ w^{λ−1}e^{−(χ/w + ψw)/2} dw = 2(χ/ψ)^{λ/2}K_λ(√(χψ))` from `√(λ² + χψ) = 50` on as
//! `ln 2 + H_λ(w*)` plus Debye's remainder, `H_λ(w) = λ ln w − ½(χ/w + ψw)` at its maximiser, so a ratio of two meets as one divergence.
//! Reference: NIST Digital Library of Mathematical Functions, eq. 10.41.4 (Debye's expansion of `K_ν(νz)`, whose `−νη` with the power is `H_λ(w*)`), <https://dlmf.nist.gov/10.41.4>.

use super::bessel_i::debye_sum_at;
pub(crate) use super::bessel_k::debye_shape;
use super::split_pow2;
use super::twofold::two_prod;
use super::twofold::two_sum;

/// From this `√(ν² + x²)` on Debye's expansion holds to the last bit at any order `ν`; the GH, GIG and VG analytics take their
/// saddle forms there.
pub(crate) const LARGE_ORDER: f64 = 50.0;

/// The maximiser of `H_λ` from `√χ` and `√ψ`, `(λ + √(λ² + χψ))/ψ`, at `λ < 0` as `χ/(|λ| + √(λ² + χψ))`, which does not
/// cancel; neither squares a root, so it holds where `χ` or `ψ` would overflow.
pub(crate) fn saddle(lambda: f64, root_chi: f64, root_psi: f64) -> f64 {
  saddle_at(
    lambda,
    lambda.hypot(root_chi * root_psi),
    (root_chi, root_psi),
  )
}

/// [`saddle`] at a known `root = √(λ² + χψ)`.
fn saddle_at(lambda: f64, root: f64, (root_chi, root_psi): (f64, f64)) -> f64 {
  if lambda >= 0.0 {
    (lambda + root) / root_psi / root_psi
  } else {
    root_chi * (root_chi / (root - lambda))
  }
}

/// [`saddle_at`] as `(m, k)`, `w = 2ᵏm`, which may leave the doubles; where it does not, `m` holds `saddle_at`'s bits.
pub(crate) fn scaled_saddle(
  lambda: f64,
  root: f64,
  (root_chi, root_psi): (f64, f64),
) -> (f64, i32) {
  if lambda >= 0.0 {
    let (k, m) = split_pow2(root_psi);
    let (j, q) = split_pow2((lambda + root) / m / m);
    (q, j - 2 * k)
  } else {
    let (k, m) = split_pow2(root_chi);
    let (j, d) = split_pow2(root - lambda);
    let (i, q) = split_pow2(m * (m / d));
    (q, i + 2 * k - j)
  }
}

/// `ln` of [`saddle`] from its factors' logarithms, finite where the maximiser leaves the range of doubles.
pub(crate) fn ln_saddle(lambda: f64, root_chi: f64, root_psi: f64) -> f64 {
  let root = lambda.hypot(root_chi * root_psi);
  if lambda >= 0.0 {
    (lambda + root).ln() - 2.0 * root_psi.ln()
  } else {
    2.0 * root_chi.ln() - (root - lambda).ln()
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

/// `x/w − 1` in two parts for `x` and `w` in two parts, the difference formed exactly and the quotient's remainder by a
/// fused multiply-add.
pub(crate) fn offset((x, x_lo): (f64, f64), (w, w_lo): (f64, f64)) -> (f64, f64) {
  let (d, d_lo) = two_sum(x, -w);
  let rest = d_lo + (x_lo - w_lo);
  let e = (d + rest) / w;
  (e, ((-e).mul_add(w, d) + rest - e * w_lo) / w)
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

/// `H_λ(w(1 + e)) − H_λ(w)` at the maximiser `w`, `e` in two parts, in terms of one sign: `λ·ln1pmx(e) − (χ/2w)e²/(1 + e)` at
/// `λ ≥ 0`, else `λ·ℓ(e) − (ψw/2)e²/(1 + e)`, `ℓ(e) = ln(1 + e) − e/(1 + e)`, at the high part plus the slope times the low.
pub(crate) fn divergence(
  lambda: f64,
  (chi, psi): (f64, f64),
  w: f64,
  (e, e_lo): (f64, f64),
) -> f64 {
  let ratio = two_sum(1.0, e);
  let coefficient = if lambda >= 0.0 {
    0.5 * chi / w
  } else {
    0.5 * psi * w
  };
  // `D′(e)·e_lo`, `D′ = −e(λ + (χ/2w)(2 + e)/(1 + e))/(1 + e)`, or `e(λ − (ψw/2)(2 + e))/(1 + e)²` at `λ < 0`.
  let slope = if lambda >= 0.0 {
    -e * (lambda + coefficient * (2.0 + e) / ratio.0) / ratio.0
  } else {
    e * (lambda - coefficient * (2.0 + e)) / ratio.0 / ratio.0
  };
  let low = slope * e_lo;
  divergence_at(lambda, coefficient, e, ratio) + if low.is_finite() { low } else { 0.0 }
}

/// `divergence` at a one-part `e`, `coefficient` its `χ/2w` or `ψw/2`, `ratio` exactly `1 + e`, which keeps its digits as
/// `e → −1` where a rounded `1 + e` would not.
fn divergence_at(lambda: f64, coefficient: f64, e: f64, (ratio, ratio_lo): (f64, f64)) -> f64 {
  let s = e / (2.0 + e);
  if s.abs() <= 0.2 || (lambda >= 0.0 && s > 0.5) {
    let q = e * e / (1.0 + e);
    return if lambda >= 0.0 {
      lambda * ln_1p_mx(e) - coefficient * q
    } else {
      -lambda * ln_1p_mx(-e / (1.0 + e)) - coefficient * q
    };
  }
  let q = e * e / ratio;
  if s.abs() <= 0.5 {
    // `ln1pmx(e) = 2s³Σ − e²/(2 + e)` and `ℓ(e) = 2s³Σ + e²/((1 + e)(2 + e))`, where the two forms below would cancel.
    let s2 = s * s;
    let tail = 2.0
      * s
      * s2
      * WIDE
        .iter()
        .rev()
        .fold(0.0_f64, |acc, &c| acc.mul_add(s2, c));
    let lead = e * e / (2.0 + e);
    return if lambda >= 0.0 {
      lambda * (tail - lead) - coefficient * q
    } else {
      lambda * (lead / ratio + tail) - coefficient * q
    };
  }
  // A ratio past the normal range leaves the `1/x` or `x` term, or `|λ| ≥ 50` times the logarithm, past `e^{−10⁴}`.
  if !ratio.is_normal() {
    return f64::NEG_INFINITY;
  }
  let ln_ratio = ratio.ln() + ratio_lo / ratio;
  if lambda >= 0.0 {
    lambda * (ln_ratio - e) - coefficient * q
  } else {
    lambda * (ln_ratio - e / ratio) - coefficient * q
  }
}

/// `1/(2j + 3)` to `j = 26`: at `s² ≤ ¼` the next term of `Σ s^{2j}/(2j + 3)` is under `4e-18` of the sum.
const WIDE: [f64; 27] = {
  let mut c = [0.0; 27];
  let mut j = 0;
  while j < 27 {
    c[j] = 1.0 / (2 * j + 3) as f64;
    j += 1;
  }
  c
};

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

/// `N(λ + k, χ, ψ)/N(λ, χ, ψ) = w_kᵏe^{rest}`, `N(λ, χ, ψ) = 2(χ/ψ)^{λ/2}K_λ(√(χψ))`, both orders past `LARGE_ORDER`, as
/// `(w_k, rest)`: `H_λ(w_k) − H_λ(w)` a divergence, `w_k` the maximiser at `λ + k`, so `E Xᵏ` of the GIG rounds no `ln w_k`.
pub(crate) fn order_step(lambda: f64, k: f64, (rc, rp): (f64, f64)) -> (f64, f64) {
  let to = lambda + k;
  let omega = rc * rp;
  let (r, r_k) = (lambda.hypot(omega), to.hypot(omega));
  let wk = saddle_at(to, r_k, (rc, rp));
  let rest = rest_step(lambda.abs(), to.abs(), (omega, omega));
  // `w_k/w − 1` from the orders and roots alone, so a saddle past the range of doubles leaves it finite.
  let e = if lambda >= 0.0 {
    k * (1.0 + (lambda + to) / (r + r_k)) / (lambda + r)
  } else if to < 0.0 {
    k * (1.0 + (lambda.abs() + to.abs()) / (r + r_k)) / (to.abs() + r_k)
  } else {
    // `λ < 0 ≤ λ + k`: the divergence `λ(ln ρ − 1) − (λ_k + r_k)/2 + ψw + (λ − ψw/2)/ρ`, `ρ = w_k/w`, from `ψw_k = λ_k + r_k`.
    let psi_w = omega * (omega / (r - lambda));
    let ln_ratio = ln_saddle(to, rc, rp) - ln_saddle(lambda, rc, rp);
    let divergence = lambda * (ln_ratio - 1.0) - 0.5 * (to + r_k)
      + psi_w
      + (lambda - 0.5 * psi_w) * (-ln_ratio).exp();
    return (wk, divergence + rest);
  };
  // `χ/2w` at `λ ≥ 0` and `ψw/2` below are both `ω²/2(|λ| + r)`, which no saddle past the doubles moves.
  let coefficient = 0.5 * omega * (omega / (lambda.abs() + r));
  (
    wk,
    divergence_at(lambda, coefficient, e, two_sum(1.0, e)) + rest,
  )
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
