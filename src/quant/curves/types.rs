//! Core types for yield curve construction.
//!
//! Reference: ISDA 2006 Definitions; Brigo & Mercurio, "Interest Rate Models — Theory and Practice" (2006)

use std::fmt::Display;

use crate::traits::FloatExt;

/// Compounding convention for interest rate calculations.
///
/// $$
/// D(t) = \begin{cases}
/// e^{-r t} & \text{continuous} \\
/// \frac{1}{1 + r\,t} & \text{simple} \\
/// \frac{1}{(1 + r/n)^{n t}} & \text{periodic with frequency } n
/// \end{cases}
/// $$
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Compounding {
  /// Continuous compounding: $D = e^{-r t}$.
  #[default]
  Continuous,
  /// Simple (money-market) compounding: $D = 1/(1 + r t)$.
  Simple,
  /// Compounding with a fixed frequency per year (1 = annual, 2 = semi-annual, 4 = quarterly).
  Periodic(u32),
}

impl Display for Compounding {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Continuous => write!(f, "Continuous"),
      Self::Simple => write!(f, "Simple"),
      Self::Periodic(n) => write!(f, "Periodic({n})"),
    }
  }
}

impl Compounding {
  /// Convert a zero rate to a discount factor.
  pub fn discount_factor<T: FloatExt>(&self, rate: T, tau: T) -> T {
    match self {
      Self::Continuous => (-rate * tau).exp(),
      Self::Simple => T::one() / (T::one() + rate * tau),
      Self::Periodic(n) => {
        let n_t = T::from_f64_fast(*n as f64);
        (T::one() + rate / n_t).powf(-n_t * tau)
      }
    }
  }

  /// Convert a discount factor to a zero rate.
  pub fn zero_rate<T: FloatExt>(&self, df: T, tau: T) -> T {
    if tau <= T::zero() {
      return T::zero();
    }
    match self {
      Self::Continuous => -df.ln() / tau,
      Self::Simple => (T::one() / df - T::one()) / tau,
      Self::Periodic(n) => {
        let n_t = T::from_f64_fast(*n as f64);
        n_t * (df.powf(-T::one() / (n_t * tau)) - T::one())
      }
    }
  }
}

/// Interpolation method for the yield curve.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterpolationMethod {
  /// Linear interpolation on zero rates.
  #[default]
  LinearOnZeroRates,
  /// Log-linear interpolation on discount factors (piecewise constant forward rates).
  LogLinearOnDiscountFactors,
  /// Natural cubic spline on zero rates.
  CubicSplineOnZeroRates,
  /// Monotone convex on forward rates (Hagan & West, 2006).
  MonotoneConvex,
}

impl Display for InterpolationMethod {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::LinearOnZeroRates => write!(f, "Linear on zero rates"),
      Self::LogLinearOnDiscountFactors => write!(f, "Log-linear on discount factors"),
      Self::CubicSplineOnZeroRates => write!(f, "Cubic spline on zero rates"),
      Self::MonotoneConvex => write!(f, "Monotone convex"),
    }
  }
}

/// Rate instrument type used in bootstrapping.
#[derive(Debug, Clone)]
pub enum Instrument<T: FloatExt> {
  /// Cash deposit: `(maturity_in_years, rate)`.
  Deposit { maturity: T, rate: T },
  /// Forward Rate Agreement: `(start, end, rate)`.
  Fra { start: T, end: T, rate: T },
  /// Interest rate future: `(start, end, price)`.
  /// Implied rate = 100 - price; convexity adjustment applied during bootstrapping.
  Future {
    start: T,
    end: T,
    price: T,
    /// Rate volatility for convexity adjustment.
    sigma: T,
  },
  /// Par swap: `(maturity, par_rate, payment_frequency)`.
  Swap {
    maturity: T,
    rate: T,
    frequency: u32,
  },
}

impl<T: FloatExt> Instrument<T> {
  /// The maturity (or end date) of the instrument.
  pub fn maturity(&self) -> T {
    match self {
      Self::Deposit { maturity, .. } => *maturity,
      Self::Fra { end, .. } | Self::Future { end, .. } => *end,
      Self::Swap { maturity, .. } => *maturity,
    }
  }
}

/// A single calibrated point on the curve: `(time, discount_factor)`.
#[derive(Debug, Clone, Copy)]
pub struct CurvePoint<T: FloatExt> {
  pub time: T,
  pub discount_factor: T,
}
