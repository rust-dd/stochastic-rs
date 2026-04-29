//! Instrument / PricingEngine decouple — the architectural split of a
//! payoff (`Instrument`) from the model+market+method that values it
//! (`PricingEngine`).
//!
//! ```ignore
//! let opt = EuropeanOption::new_tau(100.0, OptionType::Call, 1.0);
//! let bs  = AnalyticBSEngine::new(spot, vol, rate, div);
//! let r   = bs.calculate(&opt);
//! println!("NPV = {}, Δ = {}", r.npv(), r.greeks().unwrap().delta);
//!
//! // Swap engines without touching the instrument:
//! let heston = AnalyticHestonEngine::new(spot, params, rate, div);
//! let r2 = heston.calculate(&opt);
//! ```

use super::pricing::Greeks;

/// A financial contract.
///
/// Implementors describe **what** is being priced — payoff, strike, maturity,
/// barriers, currency of settlement. They do **not** carry a model, market
/// data, or numerical method. Pair with a [`PricingEngine`] to obtain a
/// price.
pub trait Instrument {
  /// Short, human-readable kind for diagnostics. Default `"Instrument"`.
  fn instrument_kind(&self) -> &'static str {
    "Instrument"
  }
}

/// A `(model, market, method)` triple that prices an [`Instrument`].
///
/// Engines own the dynamic state (spot, vol, rates, calibrated parameters,
/// MC seed, …) and hold reactive market data through
/// [`crate::market::Handle`] so price updates automatically follow market
/// updates. The `Instrument` is borrowed read-only.
///
/// `PricingEngine<I>` is generic in the instrument type to support
/// engine families that price several payoffs from the same trait
/// implementation (e.g. an `AnalyticBSEngine` impl for both
/// `EuropeanOption` and `DigitalOption`).
pub trait PricingEngine<I: ?Sized> {
  /// Output container — must report at least an NPV.
  type Result: PricingResult;

  /// Compute the result for `instrument` using current market state.
  fn calculate(&self, instrument: &I) -> Self::Result;
}

/// Output of a pricing engine.
///
/// `npv()` is the only required field. Engines that compute Greeks or
/// MC error estimates expose them via [`greeks`](Self::greeks) and
/// [`error_estimate`](Self::error_estimate); analytical engines that
/// don't compute MC error return `None`.
pub trait PricingResult {
  /// Net present value.
  fn npv(&self) -> f64;

  /// Aggregate Greeks if the engine produced them.
  fn greeks(&self) -> Option<Greeks> {
    None
  }

  /// Standard error of the NPV estimate. `Some` for Monte Carlo engines,
  /// `None` for closed-form engines.
  fn error_estimate(&self) -> Option<f64> {
    None
  }
}

/// Convenience extension: call `instrument.npv(&engine)` directly.
///
/// Blanket-implemented for every [`Instrument`]. Equivalent to
/// `engine.calculate(&instrument).npv()` but reads naturally at call sites
/// where the user thinks "price this instrument with this engine".
pub trait InstrumentExt: Instrument {
  /// Net present value via `engine`.
  fn npv<E: PricingEngine<Self>>(&self, engine: &E) -> f64
  where
    Self: Sized,
  {
    engine.calculate(self).npv()
  }

  /// Full pricing result via `engine` — NPV + Greeks + error estimate.
  fn price<E: PricingEngine<Self>>(&self, engine: &E) -> E::Result
  where
    Self: Sized,
  {
    engine.calculate(self)
  }
}

impl<I: Instrument + ?Sized> InstrumentExt for I {}

/// Bundle holding NPV plus optional Greeks and error estimate.
///
/// The default [`PricingResult`] payload — most engines populate this and
/// return it. Closed-form engines fill `npv` and `greeks`; Monte Carlo
/// engines populate `error_estimate` as well.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StandardResult {
  pub npv: f64,
  pub greeks: Option<Greeks>,
  pub error_estimate: Option<f64>,
}

impl StandardResult {
  /// Result with just an NPV; Greeks and error estimate are `None`.
  pub const fn npv_only(npv: f64) -> Self {
    Self {
      npv,
      greeks: None,
      error_estimate: None,
    }
  }

  /// Result with NPV and Greeks.
  pub const fn with_greeks(npv: f64, greeks: Greeks) -> Self {
    Self {
      npv,
      greeks: Some(greeks),
      error_estimate: None,
    }
  }

  /// MC-style result: NPV, Greeks, and standard error.
  pub const fn mc(npv: f64, greeks: Option<Greeks>, error_estimate: f64) -> Self {
    Self {
      npv,
      greeks,
      error_estimate: Some(error_estimate),
    }
  }
}

impl PricingResult for StandardResult {
  fn npv(&self) -> f64 {
    self.npv
  }
  fn greeks(&self) -> Option<Greeks> {
    self.greeks
  }
  fn error_estimate(&self) -> Option<f64> {
    self.error_estimate
  }
}
