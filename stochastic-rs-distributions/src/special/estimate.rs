//! A value with an estimate of its absolute error, so that a law built from cancelling logarithms answers NaN where no
//! digit survives instead of a confident wrong number.

use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Sub;

/// `value` with an absolute error about `err`, propagated to first order through `+ − × ÷` and powers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Estimate {
  value: f64,
  err: f64,
}

impl Estimate {
  /// `value`, whose logarithm sums `terms`: relative error `ε·Σ|terms|`, NaN where that reaches 1 even if `value`
  /// under- or overflowed; where a term is not finite, IEEE arithmetic decides alone.
  pub(crate) fn of(value: f64, terms: &[f64]) -> Self {
    if !terms.iter().all(|t| t.is_finite()) {
      return Self { value, err: 0.0 };
    }
    let rel = f64::EPSILON * terms.iter().map(|t| t.abs()).sum::<f64>();
    if rel >= 1.0 {
      return Self {
        value: f64::NAN,
        err: f64::NAN,
      };
    }
    Self {
      value,
      err: value.abs() * rel,
    }
  }

  pub(crate) fn from_log_sum(sum: f64, terms: &[f64]) -> Self {
    Self::of(sum.exp(), terms)
  }

  /// The value, or NaN where a finite value's error reaches its size.
  pub(crate) fn or_nan(self) -> f64 {
    if self.value.is_finite() && self.err > 0.0 && self.err >= self.value.abs() {
      f64::NAN
    } else {
      self.value
    }
  }

  pub(crate) fn powi(self, n: i32) -> Self {
    Self {
      value: self.value.powi(n),
      err: f64::from(n.unsigned_abs()) * self.value.abs().powi(n - 1) * self.err,
    }
  }

  pub(crate) fn powf(self, p: f64) -> Self {
    Self {
      value: self.value.powf(p),
      err: p.abs() * self.value.abs().powf(p - 1.0) * self.err,
    }
  }
}

impl Add for Estimate {
  type Output = Self;

  fn add(self, b: Self) -> Self {
    Self {
      value: self.value + b.value,
      err: self.err + b.err,
    }
  }
}

impl Sub for Estimate {
  type Output = Self;

  fn sub(self, b: Self) -> Self {
    Self {
      value: self.value - b.value,
      err: self.err + b.err,
    }
  }
}

impl Sub<f64> for Estimate {
  type Output = Self;

  fn sub(self, c: f64) -> Self {
    Self {
      value: self.value - c,
      err: self.err,
    }
  }
}

impl Mul for Estimate {
  type Output = Self;

  fn mul(self, b: Self) -> Self {
    Self {
      value: self.value * b.value,
      err: self.value.abs() * b.err + b.value.abs() * self.err,
    }
  }
}

impl Mul<Estimate> for f64 {
  type Output = Estimate;

  fn mul(self, b: Estimate) -> Estimate {
    Estimate {
      value: self * b.value,
      err: self.abs() * b.err,
    }
  }
}

impl Div for Estimate {
  type Output = Self;

  fn div(self, b: Self) -> Self {
    let value = self.value / b.value;
    Self {
      value,
      err: (self.err + value.abs() * b.err) / b.value.abs(),
    }
  }
}
