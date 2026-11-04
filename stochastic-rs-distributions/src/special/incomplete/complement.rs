//! `1 − I_x(a, b)` read from `x` itself, so neither a rounded `1 − x` nor a cancelling `1 − I` enters it, and `ln(a·B(a, b))`
//! free of cancellation at a shape at most 1.
//! Reference: DiDonato, A.R., Morris, A.H. (1992), "Algorithm 708: Significant digit computation of the incomplete beta function ratios", ACM TOMS 18(3), 360-373, DOI 10.1145/131766.131776 (`bpser`'s series, `algdiv`'s Stirling difference).
//! Reference: NIST Digital Library of Mathematical Functions, eq. 8.17.7 (`B_x(a, b) = x^a/a·F(a, 1 − b; a + 1; x)`), <https://dlmf.nist.gov/8.17.7>; eq. 5.11.1 (Stirling's series), <https://dlmf.nist.gov/5.11.1>.

use super::recip_gamma_1p_m1_over;
use crate::special::expm1_over;
use crate::special::ln_1p_over;
use crate::special::ln_gamma;

/// From this argument `ln Γ` differences are Stirling's, whose seventh term is under `2e-18` of the digamma; unit shifts
/// lift a smaller one.
const STIRLING_FROM: f64 = 16.0;

/// `B_{2k}/(2k(2k − 1))`, `k = 1, …, 6`, Stirling's coefficients (DLMF 5.11.1).
const STIRLING: [f64; 6] = [
  1.0 / 12.0,
  -1.0 / 360.0,
  1.0 / 1260.0,
  -1.0 / 1680.0,
  1.0 / 1188.0,
  -691.0 / 360360.0,
];

/// `1 − I_x(p, q) = −expm1(E)`, `E = p ln x − ln(pB(p, q)) + ln(1 + pΣ)`, `Σ = Σ_{n≥1} (1 − q)_n xⁿ/(n!(p + n))`, at
/// `0 < p ≤ 1`, `0 < x ≤ ½`, `qx ≤ 2`; `E` is `p` times a sum of order one, so a subnormal tail rounds once.
pub(super) fn upper_series(p: f64, q: f64, x: f64) -> f64 {
  let (mut term, mut sum, mut n) = (1.0, 0.0, 1.0);
  while n <= 500.0 {
    term *= (n - q) / n * x;
    let w = term / (p + n);
    sum += w;
    if w.abs() <= 1e-17 * sum.abs() {
      break;
    }
    n += 1.0;
  }
  let (s, z, t) = shape_beta_parts(p, q);
  let xz = x * z;
  // `x` and `z` meet at `qx ≈ 1` with logarithms of either sign, which their product cancels exactly.
  let ln_xz = if xz >= 1e-290 {
    xz.ln()
  } else {
    x.ln() + z.ln()
  };
  let e = p.mul_add(ln_xz + s + ln_1p_over(p * sum) * sum, -t);
  -e.exp_m1()
}

/// `ln(p·B(p, q))`, free of the cancellation of `ln Γ` terms where a shape is at most 1.
pub(super) fn ln_shape_beta(p: f64, q: f64) -> f64 {
  if p <= 1.0 && p <= q {
    let (s, z, t) = shape_beta_parts(p, q);
    t - p * (z.ln() + s)
  } else if q <= 1.0 {
    let (s, z, t) = shape_beta_parts(q, p);
    p.ln() - q.ln() + t - q * (z.ln() + s)
  } else {
    p.ln() + ln_gamma(p) + ln_gamma(q) - ln_gamma(p + q)
  }
}

/// `−ln(pB(p, q)) = p(ln z + s) − t` at `0 < p ≤ 1`: `−ln Γ(1 + p)` from A&S 6.1.34, `ln Γ(z + p) − ln Γ(z)` at `z = q + n ≥ 16`
/// by Stirling's difference, the unit shifts `ln(1 + p/(q + j))` over `p` in `s`, and `t` the first one where `1/q` overflows.
fn shape_beta_parts(p: f64, q: f64) -> (f64, f64, f64) {
  let g_over = recip_gamma_1p_m1_over(p);
  let mut s = ln_1p_over(p * g_over) * g_over;
  let (mut z, mut t) = (q, 0.0);
  if q < STIRLING_FROM {
    let first = ln_1p_over(p / q) / q;
    if first.is_finite() {
      s -= first;
    } else {
      t = (p / q).ln_1p();
    }
    z = q + 1.0;
    while z < STIRLING_FROM {
      s -= ln_1p_over(p / z) / z;
      z += 1.0;
    }
  }
  // `[(z + p − ½)ln(1 + u) − p]/p` with `u = p/z`, then `Σ c_k((z + p)^{1−2k} − z^{1−2k})/p` through `expm1`.
  let (u, z2) = (p / z, (z * z).recip());
  let l1u = ln_1p_over(u);
  let (mut power, mut phi) = (z.recip(), 0.0);
  for (k, c) in STIRLING.iter().enumerate() {
    let m = (2 * k + 1) as f64;
    phi += c * m * power * expm1_over(-m * u * l1u);
    power *= z2;
  }
  s += ((z + p - 0.5) / z) * l1u - 1.0 - l1u / z * phi;
  (s, z, t)
}
