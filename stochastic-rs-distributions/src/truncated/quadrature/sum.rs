//! Error-free sums and products, the running sum the mass walks carry their roundings in, and an
//! exponent's terms summed exactly in a frame scaled below the doubles' range.
//!
//! Reference: Shewchuk, J. R. (1997), "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates", *Discrete & Computational Geometry* 18(3), 305-363, DOI 10.1007/PL00009321 (the growing expansion).

pub(in crate::truncated) use crate::special::two_prod;
pub(in crate::truncated) use crate::special::two_sum;

use crate::special::ldexp;
use crate::special::split_pow2;

/// The terms' sum to a double-double as if formed in thrice the precision: two cascades of two-sums
/// that carry each rounding error to the next term, then the errors' sum added once (SumK, K = 3).
pub(in crate::truncated) fn accurate_sum<const N: usize>(mut p: [f64; N]) -> (f64, f64) {
  for _ in 0..2 {
    for i in 1..N {
      (p[i], p[i - 1]) = two_sum(p[i], p[i - 1]);
    }
  }
  two_sum(p[N - 1], p[..N - 1].iter().sum::<f64>())
}

/// A running sum carrying its additions' rounding errors apart: a panel far below the sum keeps its
/// digits, so the cdf next to 1 and `M₀` differ by the panels between, not by their roundings.
#[derive(Clone, Copy, Default)]
pub(super) struct Sum {
  pub(super) hi: f64,
  pub(super) lo: f64,
}

impl Sum {
  #[inline]
  pub(super) fn add(&mut self, x: f64) {
    let (hi, e) = two_sum(self.hi, x);
    self.hi = hi;
    self.lo += e;
  }

  #[inline]
  pub(super) fn value(self) -> f64 {
    self.hi + self.lo
  }
}

/// A nonoverlapping expansion: its components sum exactly to every double added, the zeros dropped.
struct Expansion {
  e: [f64; 72],
  n: usize,
}

impl Expansion {
  fn new() -> Self {
    Self { e: [0.0; 72], n: 0 }
  }

  fn add(&mut self, b: f64) {
    let (mut q, mut m) = (b, 0);
    for i in 0..self.n {
      let (s, h) = two_sum(q, self.e[i]);
      if h != 0.0 {
        self.e[m] = h;
        m += 1;
      }
      q = s;
    }
    if q != 0.0 {
      self.e[m] = q;
      m += 1;
    }
    self.n = m;
  }

  /// The sum, smallest components first, so the largest is rounded once.
  fn value(&self) -> f64 {
    self.e[..self.n].iter().sum()
  }
}

/// Terms `v·2^p`, each exact; a product of two doubles is two of them.
pub(in crate::truncated) struct Terms {
  parts: [(f64, i32); 32],
  n: usize,
}

impl Terms {
  pub(in crate::truncated) fn new() -> Self {
    Self {
      parts: [(0.0, 0); 32],
      n: 0,
    }
  }

  /// `v·2^p`; a non-finite `v` makes the sum NaN.
  pub(in crate::truncated) fn scaled(&mut self, v: f64, p: i32) {
    if v != 0.0 {
      self.parts[self.n] = (v, p);
      self.n += 1;
    }
  }

  /// `f·g·2^p` exactly, from the factors' mantissas.
  pub(in crate::truncated) fn product(&mut self, f: f64, g: f64, p: i32) {
    if f == 0.0 || g == 0.0 {
      return;
    }
    if !(f.is_finite() && g.is_finite()) {
      return self.scaled(f64::NAN, 0);
    }
    let ((ef, mf), (eg, mg)) = (split_pow2(f.abs()), split_pow2(g.abs()));
    let (h, l) = two_prod(mf, mg);
    let sign = f.signum() * g.signum();
    self.scaled(sign * h, ef + eg + p);
    self.scaled(sign * l, ef + eg + p);
  }

  /// The sum where `exp` reads it, else `±∞`: the parts scaled `2^{−k}` below the largest's
  /// `2^1000`, exact to `2^{k−1074}`; cancelling into range, it restores what the frame dropped.
  pub(in crate::truncated) fn total(&self) -> f64 {
    let parts = &self.parts[..self.n];
    if parts.iter().any(|(v, _)| !v.is_finite()) {
      return f64::NAN;
    }
    let top = parts
      .iter()
      .map(|&(v, p)| split_pow2(v.abs()).0 + p)
      .max()
      .unwrap_or(0);
    let k = (top - 1000).max(0);
    let mut frame = Expansion::new();
    for &(v, p) in parts {
      frame.add(ldexp(v, p - k));
    }
    let approx = frame.value();
    if approx.abs() > ldexp(4096.0, -k) {
      return approx.signum() * f64::INFINITY;
    }
    if k == 0 {
      return approx;
    }
    let mut sum = Expansion::new();
    for &c in &frame.e[..frame.n] {
      sum.add(ldexp(c, k));
    }
    for &(v, p) in parts {
      let u = ldexp(v, p);
      if u.is_finite() {
        sum.add(u - ldexp(ldexp(v, p - k), k));
      }
    }
    sum.value()
  }
}
