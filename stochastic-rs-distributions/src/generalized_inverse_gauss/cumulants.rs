//! The first four cumulants of GIG(λ, χ, ψ) as `sⁿk̂ₙ` for a scale `s`, none formed from cancelling raw moments: Debye's
//! expansion differentiated from `r = √(λ² + χψ) = 100`, the trapezoidal rule about the mode from 1.5, the Bessel ratios below.
//! Reference: NIST Digital Library of Mathematical Functions, eq. 10.41.4 (Debye's expansion, a function of `r` alone at a fixed order), <https://dlmf.nist.gov/10.41.4>.
//! Reference: Trefethen, L.N., Weideman, J.A.C. (2014), "The exponentially convergent trapezoidal rule", *SIAM Review* 56(3), 385-458, DOI 10.1137/130932132.

use crate::special::debye_moments;
use crate::special::k_half_ratio;
use crate::special::saddle;

/// From this `r` Debye's corrections hold double precision at either sign of `λ`; at a negative order they converge like
/// `(4/|λ|)ᵏ`, still `1e-12` short at `r = 50`.
const DEBYE_FROM: f64 = 100.0;

/// From this `r` the rule's step `STEP/√r` holds `3e-15`; below it the law is too wide in `ln x` for a short rule.
const QUADRATURE_FROM: f64 = 1.5;

const STEP: f64 = 0.2;

/// Nodes on either side of the mode; a law whose fourth-moment integrand runs past them has an inverse-gamma tail, which
/// carries its moments, so the Bessel ratios do not cancel there (`Σ|terms| ≤ 33|κ₄|`).
const SIDE: usize = 512;

/// `(s, [κ₁/s, κ₂/s², κ₃/s³, κ₄/s⁴])`.
pub(super) fn cumulants(lambda: f64, chi: f64, psi: f64) -> (f64, [f64; 4]) {
  let (root_chi, root_psi) = (chi.sqrt(), psi.sqrt());
  let omega = root_chi * root_psi;
  let r = lambda.hypot(omega);
  let w = saddle(lambda, root_chi, root_psi);
  if r >= DEBYE_FROM {
    return (w, debye(lambda, omega, r));
  }
  if r >= QUADRATURE_FROM
    && let Some(k) = trapezoid(lambda, omega, r)
  {
    return (w, k);
  }
  ratios(lambda, chi, psi, omega)
}

/// `(−2)ⁿ dⁿ ln N/dψⁿ` over `wⁿ`, `N = 2(χ/ψ)^{λ/2}K_λ(√(χψ))`: the order's part `w(ψ) = (λ + r)/ψ` differentiated in closed
/// form, every term positive, and `T = ½ ln(π/2r) + ln S` by `(χ/r²)ⁿθ(θ + 2)…(θ + 2n − 2)`, `θ = s d/ds`, `s = 1/r`.
fn debye(lambda: f64, omega: f64, r: f64) -> [f64; 4] {
  let nu = lambda.abs();
  let th = debye_moments(nu / r, 1.0 / r);
  let sum = 1.0 + th[0];
  let [g1, g2, g3, g4] = [th[1] / sum, th[2] / sum, th[3] / sum, th[4] / sum];
  let l1 = g1;
  let l2 = g2 - g1 * g1;
  let l3 = g3 - 3.0 * g1 * g2 + 2.0 * g1.powi(3);
  let l4 = g4 - 4.0 * g1 * g3 - 3.0 * g2 * g2 + 12.0 * g1 * g1 * g2 - 6.0 * g1.powi(4);
  // `q = χ/(rw)`: `(r − λ)/r` at `λ ≥ 0`, as `ω²/((r + λ)r)`, and `(r + |λ|)/r` below.
  let q = if lambda >= 0.0 {
    (omega / r) * (omega / (r + lambda))
  } else {
    1.0 + nu / r
  };
  let a = q / r;
  [
    1.0 + a * (0.5 + l1),
    1.0 / r + a * a * (1.0 + l2 + 2.0 * l1),
    (2.0 + q) / (r * r) + a.powi(3) * (4.0 + l3 + 6.0 * l2 + 8.0 * l1),
    (6.0 + 6.0 * q + 3.0 * q * q) / r.powi(3)
      + a.powi(4) * (24.0 + l4 + 12.0 * l3 + 44.0 * l2 + 48.0 * l1),
  ]
}

/// The central moments of `x/w` by the trapezoidal rule in `t = ln(x/w)`, where the density is `e^{φ(t)}`,
/// `φ = −|λ|·E(±t) − δ(cosh t − 1)`, `E(x) = eˣ − 1 − x`, `δ = ω²/(r + |λ|)`; `None` past `SIDE` nodes on a side.
fn trapezoid(lambda: f64, omega: f64, r: f64) -> Option<[f64; 4]> {
  let nu = lambda.abs();
  let delta = omega * (omega / (r + nu));
  let sign = if lambda >= 0.0 { 1.0 } else { -1.0 };
  let h = STEP / r.sqrt();
  let mut nodes = [(0.0_f64, 0.0_f64); 2 * SIDE];
  let mut len = 0;
  for side in [1.0, -1.0] {
    let mut i = if side > 0.0 { 0.0 } else { 1.0 };
    loop {
      if i >= SIDE as f64 {
        return None;
      }
      let t = side * i * h;
      let f = (-nu * exp_m1_mx(sign * t) - 2.0 * delta * (0.5 * t).sinh().powi(2)).exp();
      let z = t.exp_m1();
      nodes[len] = (f, z);
      len += 1;
      if f * (1.0 + z.abs()).powi(4) < 1e-22 {
        break;
      }
      i += 1.0;
    }
  }
  let nodes = &nodes[..len];
  let total = nodes.iter().map(|n| n.0).sum::<f64>();
  let d = nodes.iter().map(|&(f, z)| f * z).sum::<f64>() / total;
  let mut moments = [0.0; 3];
  for &(f, z) in nodes {
    let c = z - d;
    moments[0] += f * c * c;
    moments[1] += f * c * c * c;
  }
  let mu2 = moments[0] / total;
  for &(f, z) in nodes {
    let c2 = (z - d) * (z - d);
    moments[2] += f * c2 * (c2 - 3.0 * mu2);
  }
  Some([1.0 + d, mu2, moments[1] / total, moments[2] / total])
}

/// `eˣ − 1 − x`, its Taylor series to `x¹⁷/17!` below `|x| = ½`, where the next term is under `1e-20` of the sum.
fn exp_m1_mx(x: f64) -> f64 {
  /// `1/(k + 2)!`, `k = 0, …, 15`.
  const C: [f64; 16] = [
    1.0 / 2.0,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5040.0,
    1.0 / 40320.0,
    1.0 / 362880.0,
    1.0 / 3628800.0,
    1.0 / 39916800.0,
    1.0 / 479001600.0,
    1.0 / 6227020800.0,
    1.0 / 87178291200.0,
    1.0 / 1307674368000.0,
    1.0 / 20922789888000.0,
    1.0 / 355687428096000.0,
  ];
  if x.abs() >= 0.5 {
    return x.exp_m1() - x;
  }
  x * x * C.iter().rev().fold(0.0_f64, |acc, &c| acc.mul_add(x, c))
}

/// Scaled by the root mean square `s`: `ln(E Xᵏ/sᵏ) = Σ_{i<k} ln R_{λ+i} − (k/2)(ln R_λ + ln R_{λ+1})`, `R_ν = K_{ν+1}(ω)/K_ν(ω)`
/// as `ln g_ν − h` from `ν = −½` and `h − ln g_{−ν−1}` below, `g_ν = (ω/2)R_ν`, `h = ln(ω/2)`, the `h` parts summed apart.
fn ratios(lambda: f64, chi: f64, psi: f64, omega: f64) -> (f64, [f64; 4]) {
  let h = (0.5 * omega).ln();
  let parts = [0.0, 1.0, 2.0, 3.0].map(|i| {
    let nu = lambda + i;
    if nu >= -0.5 {
      (k_half_ratio(nu, omega), -1.0)
    } else {
      (1.0 / k_half_ratio(-nu - 1.0, omega), 1.0)
    }
  });
  let ln_b = |k: usize| {
    let (head, power) = (0.5 * k as f64, parts[0].1 + parts[1].1);
    let g =
      parts[..k].iter().map(|p| p.0.ln()).sum::<f64>() - head * (parts[0].0.ln() + parts[1].0.ln());
    g + (parts[..k].iter().map(|p| p.1).sum::<f64>() - head * power) * h
  };
  let mean = if parts[0].1 < 0.0 {
    2.0 * parts[0].0 / psi
  } else {
    0.5 * chi * parts[0].0
  };
  let (l1, l3, l4) = (ln_b(1), ln_b(3), ln_b(4));
  let (b1, k2) = (l1.exp(), 1.0 - (2.0 * l1).exp());
  // `b₄ − 4b₁b₃` with `b₄` factored out, so a heavy tail whose `b₄` and `b₁b₃` both overflow reads `+∞`, not `∞ − ∞`.
  let mu4 = l4.exp() * (1.0 - 4.0 * (l1 + l3 - l4).exp()) + 6.0 * b1 * b1 - 3.0 * b1.powi(4);
  (
    mean / b1,
    [
      b1,
      k2,
      l3.exp() - 3.0 * b1 + 2.0 * b1.powi(3),
      mu4 - 3.0 * k2 * k2,
    ],
  )
}
