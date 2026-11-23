//! Core types for yield curve construction.
//!
//! Reference: ISDA 2006 Definitions
//! Reference: Brigo & Mercurio (2006), 2nd ed., DOI 10.1007/978-3-540-34604-3

use std::fmt::Display;

use crate::calendar::Frequency;
use crate::traits::RealExt;

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
#[non_exhaustive]
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
  pub fn discount_factor<T: RealExt>(&self, rate: T, tau: T) -> T {
    match self {
      Self::Continuous => (-rate * tau).exp(),
      Self::Simple => T::one() / (T::one() + rate * tau),
      Self::Periodic(n) => {
        let n_t = T::from_f64_fast(*n as f64);
        (T::one() + rate / n_t).powf(-n_t * tau)
      }
    }
  }

  /// Convert a discount factor to a zero rate; NaN at a non-positive `tau`.
  pub fn zero_rate<T: RealExt>(&self, df: T, tau: T) -> T {
    if tau <= T::zero() {
      return T::nan();
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
#[non_exhaustive]
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

/// A market quote the curve bootstrapper turns into a discount factor.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum BootstrapInstrument<T: RealExt> {
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
    frequency: Frequency,
  },
  /// Par swap on explicit payment times in years, the last fixed one its maturity. `float_times`
  /// is read only against an exogenous discount curve, where the floating leg does not telescope.
  SwapWithSchedule {
    rate: T,
    fixed_times: Vec<T>,
    float_times: Vec<T>,
  },
}

impl<T: RealExt> BootstrapInstrument<T> {
  /// The maturity (or end date) of the instrument; NaN for an empty schedule.
  pub fn maturity(&self) -> T {
    match self {
      Self::Deposit { maturity, .. } => *maturity,
      Self::Fra { end, .. } | Self::Future { end, .. } => *end,
      Self::Swap { maturity, .. } => *maturity,
      Self::SwapWithSchedule { fixed_times, .. } => {
        fixed_times.last().copied().unwrap_or_else(T::nan)
      }
    }
  }
}

/// A single calibrated point on the curve: `(time, discount_factor)`.
#[derive(Debug, Clone, Copy)]
pub struct CurvePoint<T: RealExt> {
  pub time: T,
  pub discount_factor: T,
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn continuous_compounding_round_trip() {
    let cc = Compounding::Continuous;
    let r: f64 = 0.05;
    let tau: f64 = 2.0;
    let df = cc.discount_factor(r, tau);
    assert!((df - (-r * tau).exp()).abs() < 1e-12);
    assert!((cc.zero_rate(df, tau) - r).abs() < 1e-12);
  }

  #[test]
  fn simple_compounding_round_trip() {
    let sc = Compounding::Simple;
    let r: f64 = 0.05;
    let tau: f64 = 0.5;
    let df = sc.discount_factor(r, tau);
    assert!((df - 1.0 / (1.0 + r * tau)).abs() < 1e-12);
    assert!((sc.zero_rate(df, tau) - r).abs() < 1e-12);
  }

  #[test]
  fn periodic_compounding_round_trip() {
    let pc = Compounding::Periodic(2);
    let r: f64 = 0.04;
    let tau: f64 = 1.0;
    let df = pc.discount_factor(r, tau);
    assert!((pc.zero_rate(df, tau) - r).abs() < 1e-10);
  }

  #[test]
  fn instrument_maturity() {
    let dep = BootstrapInstrument::<f64>::Deposit {
      maturity: 0.5,
      rate: 0.03,
    };
    assert_eq!(dep.maturity(), 0.5);
    let fra = BootstrapInstrument::<f64>::Fra {
      start: 0.5,
      end: 1.0,
      rate: 0.04,
    };
    assert_eq!(fra.maturity(), 1.0);
  }
}
