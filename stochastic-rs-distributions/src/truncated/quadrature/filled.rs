//! The engine and what its first calls cache: the walk from `b`, the moments and the entropy; cells
//! inside it would let `&self` calls write it, and its hot loops reload what they read.

use std::ops::Deref;
use std::sync::OnceLock;

use super::Kernel;
use super::Point;
use super::Truncation;
use super::Walked;
use super::panels::Cut;

#[derive(Clone, Debug)]
pub(in crate::truncated) struct Filled<K> {
  t: Truncation<K>,
  /// Boxed, as most laws never form it and every law's fill moves the cell.
  upper: OnceLock<Box<Walked>>,
  summed: OnceLock<[f64; 4]>,
  entropy: OnceLock<f64>,
}

impl<K> Deref for Filled<K> {
  type Target = Truncation<K>;

  fn deref(&self) -> &Truncation<K> {
    &self.t
  }
}

impl<K: Kernel> Filled<K> {
  /// `kernel` on `[lo, up]`, `lo < up`, filled.
  pub(in crate::truncated) fn new(kernel: K, lo: impl Into<Cut>, up: impl Into<Cut>) -> Self {
    let mut t = Self::placed(kernel, lo.into(), up.into());
    t.fill();
    t
  }

  /// The engine placed, before any sum, so a model that holds it fills it where it lies.
  pub(in crate::truncated) fn placed(kernel: K, lo: Cut, up: Cut) -> Self {
    Self {
      t: Truncation::placed(kernel, lo, up),
      upper: OnceLock::new(),
      summed: OnceLock::new(),
      entropy: OnceLock::new(),
    }
  }

  pub(in crate::truncated) fn fill(&mut self) {
    self.t.fill();
  }

  pub(in crate::truncated) fn quantile(&self, p: f64) -> Point {
    self.t.quantile(p, &self.upper)
  }

  pub(in crate::truncated) fn located_moments(
    &self,
    scale: f64,
    p: i32,
  ) -> (Cut, [f64; 2], [f64; 3]) {
    self.t.located_moments(scale, p, &self.summed)
  }

  pub(in crate::truncated) fn moments(&self, scale: f64, p: i32) -> [f64; 4] {
    self.t.moments(scale, p, &self.summed)
  }

  pub(in crate::truncated) fn entropy(&self) -> f64 {
    self.t.entropy(&self.entropy)
  }
}
