//! FX forward pricing via covered interest rate parity.
//!
//! $$
//! F = S \cdot e^{(r_d - r_f)\,\tau}
//! $$
//!
//! where $S$ is the spot rate, $r_d$ the domestic (quote-currency) rate,
//! $r_f$ the foreign (base-currency) rate, and $\tau$ the time to maturity
//! in years.
//!
//! Reference: Hull, *Options, Futures, and Other Derivatives*, 11th ed.,
//! Chapter 5 — Determination of Forward and Futures Prices.

use crate::traits::FloatExt;

use super::quoting::CurrencyPair;

/// FX forward pricer using covered interest parity (CIP).
#[derive(Debug, Clone, Copy)]
pub struct FxForward<T: FloatExt> {
  /// The currency pair (base / quote).
  pub pair: CurrencyPair,
  /// Spot exchange rate (units of quote per unit of base).
  pub spot: T,
  /// Continuously-compounded domestic (quote-currency) risk-free rate.
  pub domestic_rate: T,
  /// Continuously-compounded foreign (base-currency) risk-free rate.
  pub foreign_rate: T,
  /// Time to maturity in years.
  pub maturity: T,
}

impl<T: FloatExt> std::fmt::Display for FxForward<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "FxForward({}, spot={:.6}, T={:.4})",
      self.pair,
      self.spot.to_f64().unwrap(),
      self.maturity.to_f64().unwrap()
    )
  }
}

impl<T: FloatExt> FxForward<T> {
  pub fn new(pair: CurrencyPair, spot: T, domestic_rate: T, foreign_rate: T, maturity: T) -> Self {
    Self {
      pair,
      spot,
      domestic_rate,
      foreign_rate,
      maturity,
    }
  }

  /// Forward exchange rate under continuous compounding.
  ///
  /// $F = S \cdot e^{(r_d - r_f)\,\tau}$
  pub fn forward_rate(&self) -> T {
    self.spot * ((self.domestic_rate - self.foreign_rate) * self.maturity).exp()
  }

  /// Forward exchange rate under simple (linear) compounding.
  ///
  /// $F = S \cdot \frac{1 + r_d \tau}{1 + r_f \tau}$
  pub fn forward_rate_simple(&self) -> T {
    let one = T::one();
    self.spot * (one + self.domestic_rate * self.maturity) / (one + self.foreign_rate * self.maturity)
  }

  /// Forward points: $F - S$.
  pub fn forward_points(&self) -> T {
    self.forward_rate() - self.spot
  }

  /// Forward points under simple compounding.
  pub fn forward_points_simple(&self) -> T {
    self.forward_rate_simple() - self.spot
  }

  /// Forward premium (or discount) as a fraction of spot: $(F - S) / S$.
  pub fn premium(&self) -> T {
    self.forward_points() / self.spot
  }

  /// Annualised forward premium: $\frac{F - S}{S \cdot \tau}$.
  pub fn annualised_premium(&self) -> T {
    self.forward_points() / (self.spot * self.maturity)
  }

  /// Implied domestic rate given a forward rate and foreign rate.
  ///
  /// $r_d = r_f + \frac{1}{\tau}\ln\!\left(\frac{F}{S}\right)$
  pub fn implied_domestic_rate(spot: T, forward: T, foreign_rate: T, maturity: T) -> T {
    foreign_rate + (forward / spot).ln() / maturity
  }

  /// Implied foreign rate given a forward rate and domestic rate.
  ///
  /// $r_f = r_d - \frac{1}{\tau}\ln\!\left(\frac{F}{S}\right)$
  pub fn implied_foreign_rate(spot: T, forward: T, domestic_rate: T, maturity: T) -> T {
    domestic_rate - (forward / spot).ln() / maturity
  }
}
