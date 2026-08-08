//! Terminal payoff definitions for the Heston Malliavin estimator.

use crate::OptionType;

/// A payoff determined only by the terminal underlying price.
pub trait TerminalPayoff {
  /// Returns the undiscounted payoff at maturity.
  fn value(&self, terminal_spot: f64) -> f64;
}

impl<F> TerminalPayoff for F
where
  F: Fn(f64) -> f64,
{
  fn value(&self, terminal_spot: f64) -> f64 {
    self(terminal_spot)
  }
}

/// One signed vanilla option leg.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VanillaLeg {
  /// Call or put direction.
  pub kind: OptionType,
  /// Strike price.
  pub strike: f64,
  /// Signed quantity of the option.
  pub quantity: f64,
}

impl VanillaLeg {
  /// Creates a signed vanilla leg.
  pub fn new(kind: OptionType, strike: f64, quantity: f64) -> Self {
    Self {
      kind,
      strike,
      quantity,
    }
  }

  /// Creates one long call.
  pub fn call(strike: f64) -> Self {
    Self::new(OptionType::Call, strike, 1.0)
  }

  /// Creates one long put.
  pub fn put(strike: f64) -> Self {
    Self::new(OptionType::Put, strike, 1.0)
  }
}

impl TerminalPayoff for VanillaLeg {
  fn value(&self, terminal_spot: f64) -> f64 {
    let intrinsic = match self.kind {
      OptionType::Call => (terminal_spot - self.strike).max(0.0),
      OptionType::Put => (self.strike - terminal_spot).max(0.0),
    };
    self.quantity * intrinsic
  }
}

/// A linear portfolio of European call and put legs.
#[derive(Debug, Clone, PartialEq)]
pub struct VanillaPortfolio {
  legs: Vec<VanillaLeg>,
}

impl VanillaPortfolio {
  /// Creates a portfolio from signed vanilla legs.
  pub fn new(legs: Vec<VanillaLeg>) -> Self {
    Self { legs }
  }

  /// Creates one long call.
  pub fn call(strike: f64) -> Self {
    Self::new(vec![VanillaLeg::call(strike)])
  }

  /// Creates one long put.
  pub fn put(strike: f64) -> Self {
    Self::new(vec![VanillaLeg::put(strike)])
  }

  /// Creates a vertical as a long option at `long_strike` and a short option
  /// at `short_strike`.
  pub fn vertical(kind: OptionType, long_strike: f64, short_strike: f64) -> Self {
    Self::new(vec![
      VanillaLeg::new(kind, long_strike, 1.0),
      VanillaLeg::new(kind, short_strike, -1.0),
    ])
  }

  /// Returns the signed legs.
  pub fn legs(&self) -> &[VanillaLeg] {
    &self.legs
  }
}

impl TerminalPayoff for VanillaPortfolio {
  fn value(&self, terminal_spot: f64) -> f64 {
    self.legs.iter().map(|leg| leg.value(terminal_spot)).sum()
  }
}
