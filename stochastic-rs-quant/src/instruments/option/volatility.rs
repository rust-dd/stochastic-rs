//! Volatility models plugged into caplet, cap, floor, and swaption pricing.
//!
//! $$
//! \sigma_{\mathrm{Black}}(F,K,\tau),\qquad
//! \sigma_{\mathrm{Bachelier}}(F,K,\tau),\qquad
//! \sigma_{\mathrm{Sabr}}(F,K,\tau)
//! $$
//!
//! ## Why a separate trait from [`ModelPricer`](crate::traits::ModelPricer)?
//!
//! [`ModelPricer`] is the canonical pricing trait for *equity* options:
//! `price_call(s, k, r, q, tau)` consumes spot, strike, rates, and dividend
//! yield and returns an option price. [`VolatilityModel`] sits one layer
//! lower for *interest-rate* options: given a forward $F$ (e.g. forward
//! Libor / SOFR rate or swap rate) and strike $K$, it returns the implied
//! volatility (Black-76 lognormal or Bachelier normal) — the actual
//! caplet / cap / floor / swaption pricer then consumes that volatility
//! together with its own day-count and discount machinery.
//!
//! These are intentionally parallel traits: the rates pipeline uses forwards
//! and tenor structure, the equity pipeline uses spots and dividend yields.
//! Unifying them would require collapsing fundamentally different inputs
//! into a single signature.
//!
//! Reference: Hagan, Kumar, Lesniewski, Woodward, "Managing Smile Risk",
//! Wilmott Magazine (2002).

use super::types::VolatilityQuoteKind;
use crate::pricing::sabr::hagan_implied_vol;
use crate::traits::FloatExt;

/// Pluggable volatility model used by caplet, cap, floor, and swaption pricers.
///
/// Implementations return the volatility appropriate for their quote family:
/// a lognormal (Black-76) vol or a normal (Bachelier) vol.
pub trait VolatilityModel<T: FloatExt>: Send + Sync {
  /// Implied volatility at forward $F$, strike $K$, and expiry $\tau$.
  fn implied_volatility(&self, forward: T, strike: T, tau: T) -> T;

  /// Quote family of the returned volatility.
  fn quote_kind(&self) -> VolatilityQuoteKind;
}

/// Constant lognormal (Black-76) volatility.
#[derive(Debug, Clone, Copy)]
pub struct BlackVolatility<T: FloatExt> {
  /// Constant lognormal volatility.
  pub sigma: T,
}

impl<T: FloatExt> BlackVolatility<T> {
  /// Build a constant Black-76 volatility surface.
  pub fn new(sigma: T) -> Self {
    Self { sigma }
  }
}

impl<T: FloatExt> VolatilityModel<T> for BlackVolatility<T> {
  fn implied_volatility(&self, _forward: T, _strike: T, _tau: T) -> T {
    self.sigma
  }

  fn quote_kind(&self) -> VolatilityQuoteKind {
    VolatilityQuoteKind::Lognormal
  }
}

/// Constant normal (Bachelier) volatility.
#[derive(Debug, Clone, Copy)]
pub struct BachelierVolatility<T: FloatExt> {
  /// Constant normal volatility.
  pub sigma: T,
}

impl<T: FloatExt> BachelierVolatility<T> {
  /// Build a constant Bachelier volatility surface.
  pub fn new(sigma: T) -> Self {
    Self { sigma }
  }
}

impl<T: FloatExt> VolatilityModel<T> for BachelierVolatility<T> {
  fn implied_volatility(&self, _forward: T, _strike: T, _tau: T) -> T {
    self.sigma
  }

  fn quote_kind(&self) -> VolatilityQuoteKind {
    VolatilityQuoteKind::Normal
  }
}

/// Sabr implied-volatility surface using the Hagan (2002) general-$\beta$
/// lognormal expansion.
#[derive(Debug, Clone, Copy)]
pub struct SabrVolatility<T: FloatExt> {
  /// Sabr level parameter $\alpha$.
  pub alpha: T,
  /// Sabr Cev exponent $\beta$ ($\beta=0$ normal, $\beta=1$ lognormal).
  pub beta: T,
  /// Sabr volatility of volatility $\nu$.
  pub nu: T,
  /// Sabr correlation $\rho$.
  pub rho: T,
}

impl<T: FloatExt> SabrVolatility<T> {
  /// Build a Sabr implied-volatility surface.
  pub fn new(alpha: T, beta: T, nu: T, rho: T) -> Self {
    Self {
      alpha,
      beta,
      nu,
      rho,
    }
  }
}

impl<T: FloatExt> VolatilityModel<T> for SabrVolatility<T> {
  fn implied_volatility(&self, forward: T, strike: T, tau: T) -> T {
    let sigma = hagan_implied_vol(
      strike.to_f64().unwrap_or(0.0),
      forward.to_f64().unwrap_or(0.0),
      tau.to_f64().unwrap_or(0.0),
      self.alpha.to_f64().unwrap_or(0.0),
      self.beta.to_f64().unwrap_or(0.0),
      self.nu.to_f64().unwrap_or(0.0),
      self.rho.to_f64().unwrap_or(0.0),
    );
    T::from_f64_fast(sigma)
  }

  fn quote_kind(&self) -> VolatilityQuoteKind {
    VolatilityQuoteKind::Lognormal
  }
}

/// Shifted Sabr for negative or near-zero rates.
///
/// The displacement $s$ is added to both forward and strike so that the
/// underlying Hagan expansion sees a positive shifted forward
/// $F_s = F + s$ and shifted strike $K_s = K + s$. Under this convention, a
/// call on $F$ with strike $K$ equals a call on $F_s$ with strike $K_s$, so the
/// pricing formulae stay identical after the substitution.
///
/// Reference: Oblój, "Fine-tune your smile: Correction to Hagan et al.",
/// Wilmott Magazine (2008).
#[derive(Debug, Clone, Copy)]
pub struct ShiftedSabrVolatility<T: FloatExt> {
  /// Sabr level parameter $\alpha$ in the shifted coordinates.
  pub alpha: T,
  /// Sabr Cev exponent $\beta$.
  pub beta: T,
  /// Sabr volatility of volatility $\nu$.
  pub nu: T,
  /// Sabr correlation $\rho$.
  pub rho: T,
  /// Displacement $s$ so that $F_s = F + s$ and $K_s = K + s$.
  pub shift: T,
}

impl<T: FloatExt> ShiftedSabrVolatility<T> {
  /// Build a shifted Sabr volatility surface with displacement `shift`.
  pub fn new(alpha: T, beta: T, nu: T, rho: T, shift: T) -> Self {
    Self {
      alpha,
      beta,
      nu,
      rho,
      shift,
    }
  }
}

impl<T: FloatExt> VolatilityModel<T> for ShiftedSabrVolatility<T> {
  fn implied_volatility(&self, forward: T, strike: T, tau: T) -> T {
    let sigma = hagan_implied_vol(
      (strike + self.shift).to_f64().unwrap_or(0.0),
      (forward + self.shift).to_f64().unwrap_or(0.0),
      tau.to_f64().unwrap_or(0.0),
      self.alpha.to_f64().unwrap_or(0.0),
      self.beta.to_f64().unwrap_or(0.0),
      self.nu.to_f64().unwrap_or(0.0),
      self.rho.to_f64().unwrap_or(0.0),
    );
    T::from_f64_fast(sigma)
  }

  fn quote_kind(&self) -> VolatilityQuoteKind {
    VolatilityQuoteKind::Lognormal
  }
}
