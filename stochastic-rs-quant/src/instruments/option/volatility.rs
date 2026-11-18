//! Volatility models for caplets, caps, floors and swaptions: a forward and a strike in, a
//! Black-76 or Bachelier vol out, where `ModelPricer` takes a spot, rates and a dividend yield.
//!
//! $$
//! \sigma_{\mathrm{Black}}(F,K,\tau),\qquad
//! \sigma_{\mathrm{Bachelier}}(F,K,\tau),\qquad
//! \sigma_{\mathrm{Sabr}}(F,K,\tau)
//! $$
//!
//! Reference: Hagan, Kumar, Lesniewski & Woodward (2002), "Managing Smile Risk", Wilmott Magazine

use super::types::VolatilityQuoteKind;
use crate::pricing::sabr::SabrParams;
use crate::pricing::sabr::hagan_implied_vol;
use crate::traits::RealExt;

/// Volatility model of the caplet, cap, floor and swaption pricers; it returns a lognormal
/// (Black-76) or a normal (Bachelier) vol, according to its quote family.
pub trait VolatilityModel<T: RealExt>: Send + Sync {
  /// Implied volatility at forward $F$, strike $K$, and expiry $\tau$.
  fn implied_volatility(&self, forward: T, strike: T, tau: T) -> T;

  /// Quote family of the returned volatility.
  fn quote_kind(&self) -> VolatilityQuoteKind;
}

/// Constant lognormal (Black-76) volatility.
#[derive(Debug, Clone, Copy)]
pub struct BlackVolatility<T: RealExt> {
  /// Constant lognormal volatility.
  pub sigma: T,
}

impl<T: RealExt> BlackVolatility<T> {
  /// Build a constant Black-76 volatility surface.
  pub fn new(sigma: T) -> Self {
    Self { sigma }
  }
}

impl<T: RealExt> VolatilityModel<T> for BlackVolatility<T> {
  fn implied_volatility(&self, _forward: T, _strike: T, _tau: T) -> T {
    self.sigma
  }

  fn quote_kind(&self) -> VolatilityQuoteKind {
    VolatilityQuoteKind::Lognormal
  }
}

/// Constant normal (Bachelier) volatility.
#[derive(Debug, Clone, Copy)]
pub struct BachelierVolatility<T: RealExt> {
  /// Constant normal volatility.
  pub sigma: T,
}

impl<T: RealExt> BachelierVolatility<T> {
  /// Build a constant Bachelier volatility surface.
  pub fn new(sigma: T) -> Self {
    Self { sigma }
  }
}

impl<T: RealExt> VolatilityModel<T> for BachelierVolatility<T> {
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
pub struct SabrVolatility<T: RealExt> {
  /// Sabr level parameter $\alpha$.
  pub alpha: T,
  /// Sabr Cev exponent $\beta$ ($\beta=0$ normal, $\beta=1$ lognormal).
  pub beta: T,
  /// Sabr correlation $\rho$.
  pub rho: T,
  /// Sabr volatility of volatility $\nu$.
  pub nu: T,
}

impl<T: RealExt> SabrVolatility<T> {
  /// Build a Sabr implied-volatility surface.
  pub fn new(alpha: T, beta: T, rho: T, nu: T) -> Self {
    Self {
      alpha,
      beta,
      rho,
      nu,
    }
  }
}

impl<T: RealExt> VolatilityModel<T> for SabrVolatility<T> {
  fn implied_volatility(&self, forward: T, strike: T, tau: T) -> T {
    let sigma = hagan_implied_vol(
      strike.to_f64().unwrap_or(f64::NAN),
      forward.to_f64().unwrap_or(f64::NAN),
      tau.to_f64().unwrap_or(f64::NAN),
      &sabr_params(self.alpha, self.beta, self.rho, self.nu),
    );
    T::from_f64_fast(sigma)
  }

  fn quote_kind(&self) -> VolatilityQuoteKind {
    VolatilityQuoteKind::Lognormal
  }
}

/// Shifted SABR for negative or near-zero rates: the displacement is added to the forward and
/// the strike, so a call on $F$ at $K$ is a call on $F + s$ at $K + s$.
///
/// Reference: Oblój (2008), "Fine-tune your smile: Correction to Hagan et al.", Wilmott Magazine
#[derive(Debug, Clone, Copy)]
pub struct ShiftedSabrVolatility<T: RealExt> {
  /// Sabr level parameter $\alpha$ in the shifted coordinates.
  pub alpha: T,
  /// Sabr Cev exponent $\beta$.
  pub beta: T,
  /// Sabr correlation $\rho$.
  pub rho: T,
  /// Sabr volatility of volatility $\nu$.
  pub nu: T,
  /// Displacement $s$ so that $F_s = F + s$ and $K_s = K + s$.
  pub shift: T,
}

impl<T: RealExt> ShiftedSabrVolatility<T> {
  /// Build a shifted Sabr volatility surface with displacement `shift`.
  pub fn new(alpha: T, beta: T, rho: T, nu: T, shift: T) -> Self {
    Self {
      alpha,
      beta,
      rho,
      nu,
      shift,
    }
  }
}

impl<T: RealExt> VolatilityModel<T> for ShiftedSabrVolatility<T> {
  fn implied_volatility(&self, forward: T, strike: T, tau: T) -> T {
    let sigma = hagan_implied_vol(
      (strike + self.shift).to_f64().unwrap_or(f64::NAN),
      (forward + self.shift).to_f64().unwrap_or(f64::NAN),
      tau.to_f64().unwrap_or(f64::NAN),
      &sabr_params(self.alpha, self.beta, self.rho, self.nu),
    );
    T::from_f64_fast(sigma)
  }

  fn quote_kind(&self) -> VolatilityQuoteKind {
    VolatilityQuoteKind::Lognormal
  }
}

fn sabr_params<T: RealExt>(alpha: T, beta: T, rho: T, nu: T) -> SabrParams {
  SabrParams {
    alpha: alpha.to_f64().unwrap_or(f64::NAN),
    beta: beta.to_f64().unwrap_or(f64::NAN),
    rho: rho.to_f64().unwrap_or(f64::NAN),
    nu: nu.to_f64().unwrap_or(f64::NAN),
  }
}
