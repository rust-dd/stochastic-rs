//! The first four cumulants of GIG(λ, χ, ψ) as `sⁿρⁿ⁻¹k̂ₙ` for scales `s` and `ρ`, none from cancelling raw moments: Debye's
//! expansion differentiated from `r = √(λ² + χψ) = 100`, the trapezoidal rule about the mode from 3, the Bessel ratios below.
//! Reference: NIST Digital Library of Mathematical Functions, eq. 10.41.4 (Debye's expansion, a function of `r` alone at a fixed order), <https://dlmf.nist.gov/10.41.4>.
//! Reference: Trefethen, L.N., Weideman, J.A.C. (2014), "The exponentially convergent trapezoidal rule", *SIAM Review* 56(3), 385-458, DOI 10.1137/130932132.

use std::f64::consts::LN_2;

use crate::special::debye_moments;
use crate::special::k_half_ratio;
use crate::special::k_half_ratio_plain;
use crate::special::ldexp;
use crate::special::ln_two_over;
use crate::special::quarter;
use crate::special::scaled_saddle;
use crate::special::split_pow2;

/// From this `r` Debye's corrections hold double precision at either sign of `λ`; at a negative order they converge like
/// `(4/|λ|)ᵏ`, still `1e-12` short at `r = 50`.
const DEBYE_FROM: f64 = 100.0;

/// From this `r` the rule's step `STEP/√r` holds `3e-15`; below it the Bessel ratios, their cancellation under 200, hold `3e-13`.
const QUADRATURE_FROM: f64 = 3.0;

const STEP: f64 = 0.2;

/// Nodes on either side of the mode; a law whose fourth-moment integrand runs past them has an inverse-gamma tail, which
/// carries its moments, so the Bessel ratios do not cancel there (`Σ|terms| ≤ 33|κ₄|`).
const SIDE: usize = 512;

/// `((m, e), ρ, [κ₁/s, κ₂/(s²ρ), κ₃/(s³ρ²), κ₄/(s⁴ρ³)])`, `s = 2ᵉm` split, `ρ = 1/r` where
/// Debye's cumulants fall like its powers, else 1; `(χ, ψ)` given only where both are exact.
pub(super) fn cumulants(
  lambda: f64,
  (root_chi, root_psi): (f64, f64),
  squares: Option<(f64, f64)>,
) -> Cumulants {
  let omega = root_chi * root_psi;
  let r = lambda.hypot(omega);
  let w = || scaled_saddle(lambda, r, (root_chi, root_psi));
  if r >= DEBYE_FROM {
    if !(r + lambda.abs()).is_finite() {
      let (l, r) = quarter(lambda, (root_chi, root_psi));
      let omega = (0.5 * root_chi) * (0.5 * root_psi);
      return (w(), 0.25 / r, (debye(l, omega, r, 0.25), 0));
    }
    return (w(), 1.0 / r, (debye(lambda, omega, r, 1.0), 0));
  }
  if r >= QUADRATURE_FROM
    && let Some(k) = trapezoid(lambda, omega, r)
  {
    return (w(), 1.0, (k, 0));
  }
  let (s, k) = ratios(lambda, (root_chi, root_psi), squares, omega);
  (s, 1.0, k)
}

/// `((m, e), ρ, ([k₁, k₂, k₃2⁻ᵗ, k₄2⁻²ᵗ], t))`, `κₙ = sⁿρⁿ⁻¹kₙ`, `s = 2ᵉm`.
pub(crate) type Cumulants = ((f64, i32), f64, ([f64; 4], i32));

/// `(−2)ⁿ dⁿ ln N/dψⁿ` over `wⁿ/rⁿ⁻¹`: `w = (λ + r)/ψ` differentiated in closed form, all terms
/// positive, `T = ½ ln(π/2r) + ln S` by `θ(θ + 2)…(θ + 2n − 2)` in `1/r`; quarters past MAX.
fn debye(lambda: f64, omega: f64, r: f64, unit: f64) -> [f64; 4] {
  let nu = lambda.abs();
  let th = debye_moments(nu / r, unit / r);
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
  let a = q * unit / r;
  [
    1.0 + a * (0.5 + l1),
    1.0 + q * a * (1.0 + l2 + 2.0 * l1),
    (2.0 + q) + q * q * a * (4.0 + l3 + 6.0 * l2 + 8.0 * l1),
    (6.0 + 6.0 * q + 3.0 * q * q) + q * q * q * a * (24.0 + l4 + 12.0 * l3 + 44.0 * l2 + 48.0 * l1),
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

/// `ln(E Xᵏ/sᵏ)` from the ratios `R_ν = K_{ν+1}(ω)/K_ν(ω)` as `ln g_ν − h`, `g_ν = (ω/2)R_ν`,
/// `h = ln(ω/2)` summed apart, from `ν = −½` (`h − ln g_{−ν−1}` below), `s` the root mean square.
fn ratios(
  lambda: f64,
  roots: (f64, f64),
  squares: Option<(f64, f64)>,
  omega: f64,
) -> ((f64, i32), ([f64; 4], i32)) {
  let d = ln_two_over(roots.0, roots.1, omega);
  if omega.is_normal()
    && let Some(parts) = plain_parts(lambda, omega, d)
  {
    let logs = parts.map(|p| p.0.ln());
    let (l1, l3, l4) = (
      ln_moment(&logs, &parts, d, 1),
      ln_moment(&logs, &parts, d, 3),
      ln_moment(&logs, &parts, d, 4),
    );
    let upper = parts[0].1 < 0.0;
    let (k, m) = side(upper, roots, squares);
    let mean = if upper {
      2.0 * parts[0].0 / m
    } else {
      0.5 * m * parts[0].0
    };
    let (b1, k2) = (l1.exp(), 1.0 - (2.0 * l1).exp());
    let quotient = mean / b1;
    if b1.is_normal() && quotient.is_normal() {
      let (j, s) = split_pow2(quotient);
      let scale = (s, if upper { j - k } else { j + k });
      return (scale, shape(b1, k2, (l1, l3, l4)));
    }
  }
  split_ratios(lambda, roots, squares, omega, d)
}

/// The four ratios as doubles where each is normal, the route of every ordinary law.
fn plain_parts(lambda: f64, omega: f64, d: f64) -> Option<[(f64, f64); 4]> {
  let mut parts = [(0.0, 0.0); 4];
  for (i, part) in parts.iter_mut().enumerate() {
    let nu = lambda + i as f64;
    *part = if nu >= -0.5 {
      (k_half_ratio_plain(nu, omega, d)?, -1.0)
    } else {
      let inverse = 1.0 / k_half_ratio_plain(-nu - 1.0, omega, d)?;
      if !inverse.is_normal() {
        return None;
      }
      (inverse, 1.0)
    };
  }
  Some(parts)
}

/// `ln b_k = ln(E Xᵏ/sᵏ)` from the parts' logarithms and powers of `ω/2`.
fn ln_moment(logs: &[f64; 4], parts: &[(f64, f64); 4], d: f64, k: usize) -> f64 {
  let (head, power) = (0.5 * k as f64, parts[0].1 + parts[1].1);
  let g = logs[..k].iter().sum::<f64>() - head * (logs[0] + logs[1]);
  g - (parts[..k].iter().map(|p| p.1).sum::<f64>() - head * power) * d
}

/// `ψ` above `ν = −½`, else `χ`, split, from the roots where a square is not exact.
fn side(upper: bool, (root_chi, root_psi): (f64, f64), squares: Option<(f64, f64)>) -> (i32, f64) {
  match squares {
    Some((chi, psi)) => split_pow2(if upper { psi } else { chi }),
    None => {
      let (k, m) = split_pow2(if upper { root_psi } else { root_chi });
      let (j, q) = split_pow2(m * m);
      (2 * k + j, q)
    }
  }
}

/// `[k₁, k₂, k₃, k₄]` from `b₁` and the logarithms of `b₃`, `b₄`, `s` the root mean square.
fn shape(b1: f64, k2: f64, (l1, l3, l4): (f64, f64, f64)) -> ([f64; 4], i32) {
  if l4 > EXCESS_FROM {
    return heavy(b1, k2, (l1, l3, l4));
  }
  // `b₄ − 4b₁b₃` with `b₄` factored out, so a heavy tail whose `b₄` and `b₁b₃` both overflow reads `+∞`, not `∞ − ∞`.
  let mu4 = l4.exp() * (1.0 - 4.0 * (l1 + l3 - l4).exp()) + 6.0 * b1 * b1 - 3.0 * b1.powi(4);
  (
    [
      b1,
      k2,
      l3.exp() - 3.0 * b1 + 2.0 * b1.powi(3),
      mu4 - 3.0 * k2 * k2,
    ],
    0,
  )
}

/// [`ratios`] with each ratio as `2ᵏm`, where `ω` or a ratio leaves the normal range.
#[cold]
#[inline(never)]
fn split_ratios(
  lambda: f64,
  roots: (f64, f64),
  squares: Option<(f64, f64)>,
  omega: f64,
  d: f64,
) -> ((f64, i32), ([f64; 4], i32)) {
  let parts = [0.0, 1.0, 2.0, 3.0].map(|i| {
    let nu = lambda + i;
    if nu >= -0.5 {
      (k_half_ratio(nu, omega, d), -1.0)
    } else {
      let (j, g) = k_half_ratio(-nu - 1.0, omega, d);
      let (i, q) = split_pow2(1.0 / g);
      ((i - j, q), 1.0)
    }
  });
  // `ln g` of the double where it is normal, so a law on this route keeps the plain logarithm's bits.
  let ln = |(j, g): (i32, f64)| {
    let v = ldexp(g, j);
    if v.is_normal() {
      v.ln()
    } else {
      g.ln() + f64::from(j) * LN_2
    }
  };
  let logs = parts.map(|p| ln(p.0));
  let plain = parts.map(|p| (0.0, p.1));
  let (l1, l3, l4) = (
    ln_moment(&logs, &plain, d, 1),
    ln_moment(&logs, &plain, d, 3),
    ln_moment(&logs, &plain, d, 4),
  );
  let mean_of = |((e, g), power): ((i32, f64), f64)| {
    let upper = power < 0.0;
    let (k, m) = side(upper, roots, squares);
    if upper {
      (2.0 * g / m, e - k)
    } else {
      (0.5 * m * g, e + k)
    }
  };
  let (mean, e0) = mean_of(parts[0]);
  let (b1, k2) = (l1.exp(), 1.0 - (2.0 * l1).exp());
  let (j, s) = split_pow2(mean / b1);
  ((s, e0 + j), shape(b1, k2, (l1, l3, l4)))
}

/// Past this `ln b₄` the ratios' `k₃` and `k₄` take their power of two apart.
const EXCESS_FROM: f64 = 600.0;

/// `([k₁, k₂, k₃2⁻ᵗ, k₄2⁻²ᵗ], t)`, `ln b₄ − 2t ln 2` near 600 and `b₃ ≤ √b₄`: a heavy tail's
/// `k₄` past MAX where its product with the GH's tilt is not.
#[cold]
#[inline(never)]
fn heavy(b1: f64, k2: f64, (l1, l3, l4): (f64, f64, f64)) -> ([f64; 4], i32) {
  let t = ((l4 - EXCESS_FROM) / (2.0 * LN_2)).ceil();
  let shift = t * LN_2;
  let ti = t as i32;
  let mu4 = (l4 - 2.0 * shift).exp() * (1.0 - 4.0 * (l1 + l3 - l4).exp())
    + ldexp(6.0 * b1 * b1 - 3.0 * b1.powi(4) - 3.0 * k2 * k2, -2 * ti);
  let k3 = (l3 - shift).exp() - ldexp(3.0 * b1 - 2.0 * b1.powi(3), -ti);
  ([b1, k2, k3, mu4], ti)
}
