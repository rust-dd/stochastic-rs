//! The truncated normal's kernels: the plain one and its mgf's tilt, in standardised units.

use crate::truncated::quadrature::Cut;
use crate::truncated::quadrature::Kernel;

/// `e^{−z²/2}` in standardised units, about the reference point `c`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct NormalKernel {
  pub(super) c: f64,
}

impl Kernel for NormalKernel {
  const FROM_POINT: bool = true;

  const QUADRATIC: bool = true;

  fn ln_k(&self, z: f64) -> f64 {
    -0.5 * z * z
  }

  fn anchor(&mut self, c: Cut, _at_mode: bool) {
    self.c = c.x;
  }

  fn ln_rel(&self, _z: f64, y: f64, _branch: Option<(f64, f64)>) -> f64 {
    -0.5 * y * (y + 2.0 * self.c)
  }

  fn d1(&self, z: f64, _y: f64) -> f64 {
    -z
  }

  fn d2(&self, _z: f64, _y: f64) -> f64 {
    -1.0
  }

  fn mode(&self) -> Cut {
    Cut::at(0.0)
  }

  fn branch_points(&self) -> [Option<(f64, f64)>; 2] {
    [None, None]
  }

  fn length_scale(&self) -> f64 {
    1.0
  }

  fn support(&self) -> (f64, f64) {
    (f64::NEG_INFINITY, f64::INFINITY)
  }

  fn base_moments(&self) -> [f64; 4] {
    [0.0, 1.0, 0.0, 0.0]
  }
}

/// `e^{−y²/2 + s·y}` beside a bound at `y = 0`, `s` the tilted mode's exact offset from it: the mgf's tilt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct TiltedKernel {
  pub(super) s: f64,
  pub(super) c: f64,
}

impl Kernel for TiltedKernel {
  /// About the mode, so a coordinate far from 0 still reads finite.
  fn ln_k(&self, z: f64) -> f64 {
    let y = z - self.s;
    -0.5 * y * y
  }

  fn anchor(&mut self, c: Cut, _at_mode: bool) {
    self.c = c.x;
  }

  /// `−y(y + 2c)/2 + s·y`, grouped so `s·y` never cancels against `c·y` at `c = s`.
  fn ln_rel(&self, _z: f64, y: f64, _branch: Option<(f64, f64)>) -> f64 {
    y * ((self.s - self.c) - 0.5 * y)
  }

  fn d1(&self, _z: f64, y: f64) -> f64 {
    (self.s - self.c) - y
  }

  fn d2(&self, _z: f64, _y: f64) -> f64 {
    -1.0
  }

  fn mode(&self) -> Cut {
    Cut::at(self.s)
  }

  fn branch_points(&self) -> [Option<(f64, f64)>; 2] {
    [None, None]
  }

  fn length_scale(&self) -> f64 {
    1.0
  }

  fn support(&self) -> (f64, f64) {
    (f64::NEG_INFINITY, f64::INFINITY)
  }

  fn base_moments(&self) -> [f64; 4] {
    [self.s, 1.0, 0.0, 0.0]
  }
}
