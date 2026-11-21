//! Temme's uniform expansion of `I_x(a, b)` at large `a`, `b` (DLMF 8.18.9): DiDonato and Morris's
//! coefficient recursion, the scaled `erfc` from Mills' ratio, a two-part exponent.
//! Reference: DiDonato, A.R., Morris, A.H. (1992), "Algorithm 708: Significant digit computation of the incomplete beta function ratios", ACM TOMS 18(3), 360-373, DOI 10.1145/131766.131776 (`basym`).
//! Reference: NIST Digital Library of Mathematical Functions, eq. 8.18.8–8.18.12, <https://dlmf.nist.gov/8.18.ii>.
//! Reference: Temme, N.M. (1992), "Asymptotic inversion of the incomplete beta function", *Journal of Computational and Applied Mathematics* 41(1-2), 145-157, DOI 10.1016/0377-0427(92)90244-R.

use std::f64::consts::TAU;

use crate::special::mills_ratio;

/// The expansion's terms, each pair a power of `1/√min(a, b)` further; past 100 the twentieth is
/// under `1e-23`.
const TERMS: usize = 20;

/// `2/√π`.
const E0: f64 = std::f64::consts::FRAC_2_SQRT_PI;

/// `2^{−3/2}`.
const E1: f64 = 0.353_553_390_593_273_8;

/// `I_x(a, b)` at `λ = a − (a + b)x ≥ 0`, shapes past 100, as `(value, ln value, ln(value/bt))`:
/// `e^{−f}` from the two-part `f = a·g(−λ/a) + b·g(λ/b)`, `e^{−Δ}`, `2/√π` and the series.
pub(super) fn expansion(a: f64, b: f64, (f, f_lo): (f64, f64), star: f64) -> (f64, f64, f64) {
  let (h, r0, r1, w0) = if a < b {
    let h = a / b;
    (
      h,
      (h + 1.0).recip(),
      (b - a) / b,
      (a * (h + 1.0)).sqrt().recip(),
    )
  } else {
    let h = b / a;
    (
      h,
      (h + 1.0).recip(),
      (b - a) / a,
      (b * (h + 1.0)).sqrt().recip(),
    )
  };
  let (mut a0, mut b0, mut c, mut d) = (
    [0.0; TERMS + 1],
    [0.0; TERMS + 1],
    [0.0; TERMS + 1],
    [0.0; TERMS + 1],
  );
  a0[0] = r1 * (2.0 / 3.0);
  c[0] = -0.5 * a0[0];
  d[0] = -c[0];
  let z = (f + f_lo).max(0.0).sqrt() * std::f64::consts::SQRT_2;
  // `ζ = z·w0 ≤ 0.03√2` in the band: the recursion carries `J_n = j_n·w0^n`, since `z^n` alone
  // overflows past a shape of about 1e100.
  let zeta = z * w0;
  let zeta2 = zeta * zeta;
  let w2 = w0 * w0;
  let (mut j0, mut j1) = (E1 * mills_ratio(z), E1 * w0);
  let mut sum = j0 + d[0] * j1;
  let (mut s, h2, mut hn, mut znm1, mut zn) = (1.0, h * h, 1.0, zeta * w0, zeta2 * w0);
  let mut n = 2;
  while n <= TERMS {
    hn *= h2;
    a0[n - 1] = r0 * 2.0 * (h * hn + 1.0) / (n as f64 + 2.0);
    s += hn;
    a0[n] = r1 * 2.0 * s / (n as f64 + 3.0);
    for i in n..=n + 1 {
      let r = -0.5 * (i as f64 + 1.0);
      b0[0] = r * a0[0];
      for m in 2..=i {
        let mut bsum = 0.0;
        for j in 1..m {
          bsum += (j as f64 * r - (m - j) as f64) * a0[j - 1] * b0[m - j - 1];
        }
        b0[m - 1] = r * a0[m - 1] + bsum / m as f64;
      }
      c[i - 1] = b0[i - 1] / (i as f64 + 1.0);
      let mut dsum = 0.0;
      for j in 1..i {
        dsum += d[i - j - 1] * c[j - 1];
      }
      d[i - 1] = -(dsum + c[i - 1]);
    }
    j0 = E1 * znm1 + (n as f64 - 1.0) * w2 * j0;
    j1 = E1 * zn + n as f64 * w2 * j1;
    znm1 *= zeta2;
    zn *= zeta2;
    let t0 = d[n - 1] * j0;
    let t1 = d[n] * j1;
    sum += t0 + t1;
    if t0.abs() + t1.abs() <= f64::EPSILON * sum {
      break;
    }
    n += 2;
  }
  let rest = E0 * (-f_lo - star).exp() * sum;
  let ln_rest = rest.ln();
  // `I/(x^a y^b/B)` by Stirling's `B`: `2/√π · sum · √(2π(a + b)/(ab))`.
  let ln_ratio = ln_rest + f_lo + star + 0.5 * (TAU * (a.recip() + b.recip())).ln();
  ((-f).exp() * rest, ln_rest - f, ln_ratio)
}
