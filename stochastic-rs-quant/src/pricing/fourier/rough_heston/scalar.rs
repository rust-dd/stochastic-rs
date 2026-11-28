//! The scalar a Riccati solve runs in: a complex number for the chf, a Taylor jet for cumulants.

use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;

use num_complex::Complex64;

/// What the three solves ask of their scalar; `re` and `norm` read a jet's constant term.
pub(super) trait RiccatiScalar:
  Copy
  + Add<Output = Self>
  + Sub<Output = Self>
  + Mul<Output = Self>
  + Div<Output = Self>
  + Neg<Output = Self>
  + Mul<f64, Output = Self>
{
  const ZERO: Self;

  fn real(x: f64) -> Self;

  /// The principal square root.
  fn sqrt(self) -> Self;

  fn re(self) -> f64;

  fn norm(self) -> f64;
}

impl RiccatiScalar for Complex64 {
  const ZERO: Self = Complex64::ZERO;

  fn real(x: f64) -> Self {
    Complex64::new(x, 0.0)
  }

  fn sqrt(self) -> Self {
    Complex64::sqrt(self)
  }

  fn re(self) -> f64 {
    self.re
  }

  fn norm(self) -> f64 {
    Complex64::norm(self)
  }
}

/// `Σ cₙsⁿ` truncated after `s⁴`: one solve at `z = s` gives `ln E[e^{sX}]` to the fourth cumulant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Jet4(pub(super) [f64; 5]);

impl Jet4 {
  pub(super) const VARIABLE: Self = Self([0.0, 1.0, 0.0, 0.0, 0.0]);
}

impl Add for Jet4 {
  type Output = Self;

  fn add(self, other: Self) -> Self {
    Self(std::array::from_fn(|n| self.0[n] + other.0[n]))
  }
}

impl Sub for Jet4 {
  type Output = Self;

  fn sub(self, other: Self) -> Self {
    Self(std::array::from_fn(|n| self.0[n] - other.0[n]))
  }
}

impl Neg for Jet4 {
  type Output = Self;

  fn neg(self) -> Self {
    Self(self.0.map(|c| -c))
  }
}

impl Mul<f64> for Jet4 {
  type Output = Self;

  fn mul(self, scale: f64) -> Self {
    Self(self.0.map(|c| c * scale))
  }
}

impl Mul for Jet4 {
  type Output = Self;

  fn mul(self, other: Self) -> Self {
    Self(std::array::from_fn(|n| {
      (0..=n).map(|i| self.0[i] * other.0[n - i]).sum()
    }))
  }
}

impl Div for Jet4 {
  type Output = Self;

  fn div(self, other: Self) -> Self {
    let mut quotient = [0.0; 5];
    for n in 0..5 {
      let known = (0..n).map(|i| quotient[i] * other.0[n - i]).sum::<f64>();
      quotient[n] = (self.0[n] - known) / other.0[0];
    }
    Self(quotient)
  }
}

impl RiccatiScalar for Jet4 {
  const ZERO: Self = Self([0.0; 5]);

  fn real(x: f64) -> Self {
    Self([x, 0.0, 0.0, 0.0, 0.0])
  }

  /// NaN unless the constant term is positive: the series has no real root at a branch point.
  fn sqrt(self) -> Self {
    let mut root = [0.0; 5];
    root[0] = self.0[0].sqrt();
    for n in 1..5 {
      let known = (1..n).map(|i| root[i] * root[n - i]).sum::<f64>();
      root[n] = (self.0[n] - known) / (2.0 * root[0]);
    }
    Self(root)
  }

  fn re(self) -> f64 {
    self.0[0]
  }

  fn norm(self) -> f64 {
    self.0[0].abs()
  }
}
