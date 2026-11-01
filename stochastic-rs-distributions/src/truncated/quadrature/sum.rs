//! Error-free sums and products, and the running sum the mass walks carry their roundings in.

/// `s + e = a + b` exactly.
pub(in crate::truncated) fn two_sum(a: f64, b: f64) -> (f64, f64) {
  let s = a + b;
  let z = s - a;
  (s, (a - (s - z)) + (b - z))
}

/// `p + e = a·b` exactly while `e` stays a normal double.
pub(in crate::truncated) fn two_prod(a: f64, b: f64) -> (f64, f64) {
  let p = a * b;
  (p, a.mul_add(b, -p))
}

/// A running sum whose additions' rounding errors are carried apart: a panel far smaller than the sum keeps its digits,
/// so the cdf next to 1 and `M₀` differ by the panels between them, not by the roundings of adding those.
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
