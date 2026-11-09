//! Error-free transformations: a sum or product of two doubles as an unevaluated sum of two.
//! Reference: Dekker, T.J. (1971), "A floating-point technique for extending the available precision", *Numerische Mathematik* 18(3), 224-242, DOI 10.1007/BF01397083.
//! Reference: Shewchuk, J.R. (1997), "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates", *Discrete & Computational Geometry* 18(3), 305-363, Theorem 7 (Knuth's two-sum).

/// `s + e = a + b` exactly; public for the t copulas' level gaps, outside the stability promise.
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
  let s = a + b;
  let z = s - a;
  (s, (a - (s - z)) + (b - z))
}

/// `p + e = a·b` exactly where the product neither overflows nor underflows, by a fused
/// multiply-add; public for the t copulas' conditional argument, outside the stability promise.
pub fn two_prod(a: f64, b: f64) -> (f64, f64) {
  let p = a * b;
  (p, a.mul_add(b, -p))
}

/// `(a + a_lo)(b + b_lo)` to `ε²` of the product, the low parts' product dropped.
pub(crate) fn mul_parts((a, a_lo): (f64, f64), (b, b_lo): (f64, f64)) -> (f64, f64) {
  let (p, e) = two_prod(a, b);
  two_sum(p, e + (a * b_lo + a_lo * b))
}

/// `(a + a_lo) + (b + b_lo)` to `ε²` of the larger part.
pub(crate) fn add_parts((a, a_lo): (f64, f64), (b, b_lo): (f64, f64)) -> (f64, f64) {
  let (s, e) = two_sum(a, b);
  two_sum(s, e + (a_lo + b_lo))
}

/// `(a + a_lo) − (b + b_lo)` to `ε²` of the larger part.
pub(crate) fn sub_parts(a: (f64, f64), (b, b_lo): (f64, f64)) -> (f64, f64) {
  add_parts(a, (-b, -b_lo))
}
