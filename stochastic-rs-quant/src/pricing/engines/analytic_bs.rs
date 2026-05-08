//! Closed-form Black-Scholes engine for [`EuropeanOption`] /
//! [`DigitalOption`].
//!
//! Wraps [`BSMPricer`] / [`CashOrNothingPricer`] / [`AssetOrNothingPricer`]
//! behind reactive market handles so the engine re-prices automatically
//! after a market update.

use std::sync::Arc;

use crate::OptionType;
use crate::instruments::equity::DigitalKind;
use crate::instruments::equity::DigitalOption;
use crate::instruments::equity::EuropeanOption;
use crate::market::Handle;
use crate::market::Quote;
use crate::market::SimpleQuote;
use crate::pricing::AssetOrNothingPricer;
use crate::pricing::BSMCoc;
use crate::pricing::BSMPricer;
use crate::pricing::CashOrNothingPricer;
use crate::traits::GreeksExt;
use crate::traits::PricerExt;
use crate::traits::PricingEngine;
use crate::traits::StandardResult;

/// Analytic Black-Scholes engine.
///
/// Holds [`Handle`]s to spot, volatility, risk-free rate, and dividend
/// yield quotes — relinking any handle takes effect on the next
/// [`calculate`](Self::calculate) call.
#[derive(Clone)]
pub struct AnalyticBSEngine {
  pub spot: Handle<SimpleQuote<f64>>,
  pub volatility: Handle<SimpleQuote<f64>>,
  pub risk_free: Handle<SimpleQuote<f64>>,
  pub dividend_yield: Handle<SimpleQuote<f64>>,
  pub coc: BSMCoc,
}

impl AnalyticBSEngine {
  /// Build from explicit handles. Defaults to `BSMCoc::Merton1973` (equity
  /// with continuous dividend yield).
  pub fn new(
    spot: Handle<SimpleQuote<f64>>,
    volatility: Handle<SimpleQuote<f64>>,
    risk_free: Handle<SimpleQuote<f64>>,
    dividend_yield: Handle<SimpleQuote<f64>>,
  ) -> Self {
    Self {
      spot,
      volatility,
      risk_free,
      dividend_yield,
      coc: BSMCoc::Merton1973,
    }
  }

  /// Convenience: wrap scalar values in fresh `SimpleQuote`s and `Handle`s.
  /// Useful in tests and one-shot pricing.
  pub fn with_constants(s: f64, sigma: f64, r: f64, q: f64) -> Self {
    Self::new(
      Handle::new(Arc::new(SimpleQuote::new(s))),
      Handle::new(Arc::new(SimpleQuote::new(sigma))),
      Handle::new(Arc::new(SimpleQuote::new(r))),
      Handle::new(Arc::new(SimpleQuote::new(q))),
    )
  }

  /// Override the cost-of-carry convention.
  pub fn with_coc(mut self, coc: BSMCoc) -> Self {
    self.coc = coc;
    self
  }

  fn read_quote(handle: &Handle<SimpleQuote<f64>>, default: f64) -> f64 {
    handle.current().map(|q| q.value()).unwrap_or(default)
  }

  fn build_pricer(&self, strike: f64, opt_type: OptionType, opt: &EuropeanOption) -> BSMPricer {
    BSMPricer {
      s: Self::read_quote(&self.spot, 0.0),
      v: Self::read_quote(&self.volatility, 0.0),
      k: strike,
      r: Self::read_quote(&self.risk_free, 0.0),
      r_d: None,
      r_f: None,
      q: Some(Self::read_quote(&self.dividend_yield, 0.0)),
      tau: opt.tau,
      eval: opt.eval,
      expiration: opt.expiry,
      option_type: opt_type,
      b: self.coc,
    }
  }
}

impl PricingEngine<EuropeanOption> for AnalyticBSEngine {
  type Result = StandardResult;

  fn calculate(&self, opt: &EuropeanOption) -> StandardResult {
    let pricer = self.build_pricer(opt.strike, opt.option_type, opt);
    let npv = pricer.calculate_price();
    let greeks = pricer.greeks();
    StandardResult::with_greeks(npv, greeks)
  }
}

impl PricingEngine<DigitalOption> for AnalyticBSEngine {
  type Result = StandardResult;

  fn calculate(&self, opt: &DigitalOption) -> StandardResult {
    let s = Self::read_quote(&self.spot, 0.0);
    let sigma = Self::read_quote(&self.volatility, 0.0);
    let r = Self::read_quote(&self.risk_free, 0.0);
    let q = Self::read_quote(&self.dividend_yield, 0.0);
    let b = r - q;
    let t = opt.tau.unwrap_or(f64::NAN);
    let npv = match opt.kind {
      DigitalKind::CashOrNothing { cash } => CashOrNothingPricer {
        s,
        k: opt.strike,
        cash,
        r,
        b,
        sigma,
        t,
        option_type: opt.option_type,
      }
      .price(),
      DigitalKind::AssetOrNothing => AssetOrNothingPricer {
        s,
        k: opt.strike,
        r,
        b,
        sigma,
        t,
        option_type: opt.option_type,
      }
      .price(),
    };
    StandardResult::npv_only(npv)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::InstrumentExt;
  use crate::traits::PricingResult;

  #[test]
  fn european_call_atm_round_trip() {
    let opt = EuropeanOption::new_tau(100.0, OptionType::Call, 1.0);
    let engine = AnalyticBSEngine::with_constants(100.0, 0.20, 0.05, 0.0);
    let r = engine.calculate(&opt);
    assert!(r.npv() > 0.0);
    let g = r.greeks().unwrap();
    assert!(g.delta > 0.0 && g.delta < 1.0);
    assert!(g.gamma > 0.0);
    assert!(g.vega > 0.0);
  }

  #[test]
  fn european_put_call_parity_via_engine() {
    let call = EuropeanOption::new_tau(100.0, OptionType::Call, 1.0);
    let put = EuropeanOption::new_tau(100.0, OptionType::Put, 1.0);
    let engine = AnalyticBSEngine::with_constants(100.0, 0.20, 0.05, 0.02);
    let c = engine.calculate(&call).npv();
    let p = engine.calculate(&put).npv();
    let parity = 100.0 * (-0.02_f64).exp() - 100.0 * (-0.05_f64).exp();
    assert!((c - p - parity).abs() < 1e-8);
  }

  #[test]
  fn instrument_ext_npv_shortcut() {
    let opt = EuropeanOption::new_tau(110.0, OptionType::Call, 0.5);
    let engine = AnalyticBSEngine::with_constants(100.0, 0.25, 0.04, 0.0);
    let direct = engine.calculate(&opt).npv();
    let via_ext = opt.npv(&engine);
    assert!((direct - via_ext).abs() < 1e-15);
  }

  #[test]
  fn relinking_volatility_changes_npv() {
    let opt = EuropeanOption::new_tau(100.0, OptionType::Call, 1.0);
    let vol_quote = Arc::new(SimpleQuote::new(0.20));
    let vol_handle = Handle::new(vol_quote.clone());
    let engine = AnalyticBSEngine::new(
      Handle::new(Arc::new(SimpleQuote::new(100.0))),
      vol_handle,
      Handle::new(Arc::new(SimpleQuote::new(0.05))),
      Handle::new(Arc::new(SimpleQuote::new(0.0))),
    );
    let v_lo = engine.calculate(&opt).npv();
    vol_quote.set_value(0.30);
    let v_hi = engine.calculate(&opt).npv();
    assert!(
      v_hi > v_lo,
      "higher vol should raise call price (lo={v_lo}, hi={v_hi})"
    );
  }

  #[test]
  fn digital_cash_or_nothing() {
    let opt = DigitalOption::cash_or_nothing(100.0, OptionType::Call, 1.0, 1.0);
    let engine = AnalyticBSEngine::with_constants(100.0, 0.20, 0.05, 0.0);
    let r = engine.calculate(&opt);
    assert!(r.npv() > 0.0 && r.npv() < 1.0);
  }
}
