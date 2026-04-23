//! Volatility models plugged into caplet, cap, floor, and swaption pricing.
//!
//! $$
//! \sigma_{\mathrm{Black}}(F,K,\tau),\qquad
//! \sigma_{\mathrm{Bachelier}}(F,K,\tau),\qquad
//! \sigma_{\mathrm{SABR}}(F,K,\tau)
//! $$
//!
//! Reference: Hagan, Kumar, Lesniewski, Woodward, "Managing Smile Risk",
//! Wilmott Magazine (2002).

use super::types::VolatilityQuoteKind;
use crate::quant::pricing::sabr::hagan_implied_vol;
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

/// SABR implied-volatility surface using the Hagan (2002) general-$\beta$
/// lognormal expansion.
#[derive(Debug, Clone, Copy)]
pub struct SabrVolatility<T: FloatExt> {
  /// SABR level parameter $\alpha$.
  pub alpha: T,
  /// SABR CEV exponent $\beta$ ($\beta=0$ normal, $\beta=1$ lognormal).
  pub beta: T,
  /// SABR volatility of volatility $\nu$.
  pub nu: T,
  /// SABR correlation $\rho$.
  pub rho: T,
}

impl<T: FloatExt> SabrVolatility<T> {
  /// Build a SABR implied-volatility surface.
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
