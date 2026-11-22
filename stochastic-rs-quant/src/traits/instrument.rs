//! A payoff (`Instrument`) apart from the model, market and method that value it (`PricingEngine`):
//! two instruments and two engines, a cross-engine check rather than a third pricing layer.
//!
//! ```
//! use stochastic_rs_quant::OptionType;
//! use stochastic_rs_quant::calendar::Maturity;
//! use stochastic_rs_quant::instruments::equity::EuropeanOption;
//! use stochastic_rs_quant::pricing::engines::{AnalyticBsEngine, AnalyticHestonEngine};
//! use stochastic_rs_quant::pricing::heston::HestonParams;
//! use stochastic_rs_quant::traits::{PricingEngine, PricingResult};
//!
//! let opt = EuropeanOption::new(100.0, OptionType::Call, Maturity::Tau(1.0));
//! let bs  = AnalyticBsEngine::with_constants(100.0, 0.20, 0.05, 0.0);
//! let r   = bs.calculate(&opt);
//! if let Some(g) = r.greeks() {
//!     println!("NPV = {}, Δ = {}", r.npv(), g.delta);
//! }
//!
//! // Swap engines without touching the instrument:
//! let params = HestonParams { v0: 0.04, kappa: 1.5, theta: 0.04, sigma: 0.3, rho: -0.6 };
//! let heston = AnalyticHestonEngine::with_constants(100.0, 0.05, 0.0, params);
//! let r2 = heston.calculate(&opt);
//! assert!(r2.npv() > 0.0);
//! ```

use super::pricing::Greeks;

/// A financial contract, what is priced (payoff, strike, maturity, barriers, settlement currency),
/// with no model, market data or method; a [`PricingEngine`] prices it.
pub trait Instrument {
  /// Short, human-readable kind for diagnostics. Default `"Instrument"`.
  fn instrument_kind(&self) -> &'static str {
    "Instrument"
  }
}

/// A `(model, market, method)` triple pricing a borrowed [`Instrument`] off market handles; generic
/// in `I` so one engine prices several payoffs (`AnalyticBsEngine`: European and digital).
pub trait PricingEngine<I: ?Sized> {
  /// Output container — must report at least an NPV.
  type Result: PricingResult;

  /// Compute the result for `instrument` using current market state.
  fn calculate(&self, instrument: &I) -> Self::Result;
}

/// A pricing engine's output: `npv()` always, [`greeks`](Self::greeks) and
/// [`error_estimate`](Self::error_estimate) when the engine computes them, else `None`.
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

/// `instrument.npv(&engine)`, that is `engine.calculate(&instrument).npv()`, for every
/// [`Instrument`].
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

/// The default [`PricingResult`]: closed-form engines fill `npv` and `greeks`, Monte Carlo engines
/// `error_estimate` as well.
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
