//! The truncated normal's kernels: the plain one, its mgf's tilt in standardised units, and the tilt in the law's
//! own offset where `σt` overflows.

use crate::truncated::quadrature::Cut;
use crate::truncated::quadrature::Kernel;

/// `e^{−z²/2}` in standardised units, about the reference point `c`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct NormalKernel {
  pub(super) c: f64,
}

impl Kernel for NormalKernel {
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

/// `e^{−z²/2 + s·z}`, whose integral over the interval's own `[α, β]` is the mgf at `s = σt`: the interval never moves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct TiltedKernel {
  pub(super) s: f64,
  pub(super) c: f64,
}

impl Kernel for TiltedKernel {
  fn ln_k(&self, z: f64) -> f64 {
    z * (self.s - 0.5 * z)
  }

  fn anchor(&mut self, c: Cut, _at_mode: bool) {
    self.c = c.x;
  }

  /// `−y(y + 2c)/2 + s·y`, grouped so `s·y` never cancels against `c·y` at `c = s`.
  fn ln_rel(&self, _z: f64, y: f64, _branch: Option<(f64, f64)>) -> f64 {
    y * ((self.s - self.c) - 0.5 * y)
  }

  fn d1(&self, z: f64, _y: f64) -> f64 {
    self.s - z
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

/// `e^{tv − (v/σ)²/2}` in the law's own offset `v = x − μ`, for a `t` whose `σt` overflows: no product of the two
/// is formed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct OffsetTiltedKernel {
  pub(super) t: f64,
  pub(super) sigma: f64,
  pub(super) c: f64,
}

impl Kernel for OffsetTiltedKernel {
  fn ln_k(&self, v: f64) -> f64 {
    let z = v / self.sigma;
    self.t * v - 0.5 * z * z
  }

  fn anchor(&mut self, c: Cut, _at_mode: bool) {
    self.c = c.x;
  }

  fn ln_rel(&self, _v: f64, y: f64, _branch: Option<(f64, f64)>) -> f64 {
    let zy = y / self.sigma;
    self.t * y - 0.5 * zy * (zy + 2.0 * (self.c / self.sigma))
  }

  fn d1(&self, v: f64, _y: f64) -> f64 {
    self.t - v / self.sigma / self.sigma
  }

  fn d2(&self, _v: f64, _y: f64) -> f64 {
    let inv = 1.0 / self.sigma;
    -inv * inv
  }

  fn mode(&self) -> Cut {
    Cut::at(self.t * self.sigma * self.sigma)
  }

  fn branch_points(&self) -> [Option<(f64, f64)>; 2] {
    [None, None]
  }

  fn length_scale(&self) -> f64 {
    1.0 / self.t.abs()
  }

  fn support(&self) -> (f64, f64) {
    (f64::NEG_INFINITY, f64::INFINITY)
  }

  fn base_moments(&self) -> [f64; 4] {
    [self.t * self.sigma * self.sigma, self.sigma, 0.0, 0.0]
  }
}
