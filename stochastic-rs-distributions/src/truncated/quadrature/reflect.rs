//! A kernel read in the reflected coordinate `−x`, so a law's upper end is its lower one and the
//! cdf's relative digits follow it there.

use super::Kernel;
use super::Two;
use super::panels::Cut;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::truncated) struct Reflected<K>(pub(in crate::truncated) K);

/// `−v` without a `−0`.
fn neg(v: f64) -> f64 {
  0.0 - v
}

impl<K: Kernel> Kernel for Reflected<K> {
  const FROM_POINT: bool = K::FROM_POINT;

  const QUADRATIC: bool = K::QUADRATIC;

  fn ln_k(&self, x: f64) -> f64 {
    self.0.ln_k(neg(x))
  }

  fn anchor(&mut self, c: Cut, at_mode: bool) {
    self.0.anchor(c.neg(), at_mode);
  }

  fn ln_rel(&self, x: f64, y: f64, branch: Option<(f64, f64)>) -> f64 {
    self
      .0
      .ln_rel(neg(x), neg(y), branch.map(|(s, t)| (neg(s), neg(t))))
  }

  fn d1(&self, x: f64, y: f64) -> f64 {
    -self.0.d1(neg(x), neg(y))
  }

  fn d2(&self, x: f64, y: f64) -> f64 {
    self.0.d2(neg(x), neg(y))
  }

  fn mode(&self) -> Cut {
    self.0.mode().neg()
  }

  fn branch_points(&self) -> [Option<(f64, f64)>; 2] {
    let [first, second] = self.0.branch_points();
    [second, first].map(|point| point.map(|(s, e)| (neg(s), e)))
  }

  fn ln_rel_regular(&self, x: f64, y: f64, s: f64) -> f64 {
    self.0.ln_rel_regular(neg(x), neg(y), neg(s))
  }

  fn ln_rel_two(&self, x: Cut, y: f64) -> Two {
    self.0.ln_rel_two(x.neg(), neg(y))
  }

  fn sharpen(&mut self) {
    self.0.sharpen();
  }

  fn length_scale(&self) -> f64 {
    self.0.length_scale()
  }

  fn support(&self) -> (f64, f64) {
    let (lo, up) = self.0.support();
    (neg(up), neg(lo))
  }

  fn base_moments(&self) -> [f64; 4] {
    let [mean, sd, skewness, kurtosis] = self.0.base_moments();
    [neg(mean), sd, -skewness, kurtosis]
  }
}
