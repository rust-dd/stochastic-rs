//! Closed-form Black-Scholes engine for [`EuropeanOption`] /
//! [`DigitalOption`].
//!
//! Wraps [`BsmPricer`] / [`CashOrNothingPricer`] / [`AssetOrNothingPricer`]
//! behind reactive market handles so the engine re-prices automatically
//! after a market update.

use std::sync::Arc;

use crate::instruments::equity::DigitalKind;
use crate::instruments::equity::DigitalOption;
use crate::instruments::equity::EuropeanOption;
use crate::market::Handle;
use crate::market::Quote;
use crate::market::SimpleQuote;
use crate::pricing::AssetOrNothingPricer;
use crate::pricing::BsmCoc;
use crate::pricing::BsmPricer;
use crate::pricing::CashOrNothingPricer;
use crate::traits::ModelPricer;
use crate::traits::PricingEngine;
use crate::traits::StandardResult;
use crate::traits::TimeExt;

/// Analytic Black-Scholes engine.
///
/// Holds [`Handle`]s to spot, volatility, risk-free rate, and dividend
/// yield quotes — relinking any handle takes effect on the next
/// [`calculate`](Self::calculate) call.
#[derive(Clone)]
pub struct AnalyticBsEngine {
  pub s: Handle<SimpleQuote<f64>>,
  pub volatility: Handle<SimpleQuote<f64>>,
  pub r: Handle<SimpleQuote<f64>>,
  pub dividend_yield: Handle<SimpleQuote<f64>>,
  pub coc: BsmCoc,
}

impl AnalyticBsEngine {
  /// Build from explicit handles. Defaults to `BsmCoc::Merton1973` (equity
  /// with continuous dividend yield).
  pub fn new(
    s: Handle<SimpleQuote<f64>>,
    volatility: Handle<SimpleQuote<f64>>,
    r: Handle<SimpleQuote<f64>>,
    dividend_yield: Handle<SimpleQuote<f64>>,
  ) -> Self {
    Self {
      s,
      volatility,
      r,
      dividend_yield,
      coc: BsmCoc::Merton1973,
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

  /// Overrides the cost-of-carry convention; [`BsmCoc::GarmanKohlhagen1983`] takes the engine's
  /// rate and dividend-yield handles as the domestic and foreign rates.
  pub fn with_coc(mut self, coc: BsmCoc) -> Self {
    self.coc = coc;
    self
  }

  /// Current value of a market handle; NaN when it is unlinked, since an unset handle is missing
  /// data and reading it as `0.0` would price at a market the caller never supplied.
  fn read_quote(handle: &Handle<SimpleQuote<f64>>) -> f64 {
    handle.current().map(|q| q.value()).unwrap_or(f64::NAN)
  }

  /// The model the handles describe and the `(s, k, r, q, tau)` query `opt` asks for; `tau` is
  /// the instrument's own, through [`TimeExt`].
  fn model_and_query(&self, opt: &EuropeanOption) -> (BsmPricer, (f64, f64, f64, f64, f64)) {
    let model = BsmPricer::new(Self::read_quote(&self.volatility), self.coc);
    let query = (
      Self::read_quote(&self.s),
      opt.strike,
      Self::read_quote(&self.r),
      Self::read_quote(&self.dividend_yield),
      opt.tau(),
    );
    (model, query)
  }

  /// The `(s, k, r, q, tau)` query of a [`DigitalOption`], as in
  /// [`model_and_query`](Self::model_and_query).
  fn digital_query(&self, opt: &DigitalOption) -> (f64, f64, f64, f64, f64) {
    (
      Self::read_quote(&self.s),
      opt.strike,
      Self::read_quote(&self.r),
      Self::read_quote(&self.dividend_yield),
      opt.tau(),
    )
  }
}

impl PricingEngine<EuropeanOption> for AnalyticBsEngine {
  type Result = StandardResult;

  /// The NPV and every Greek are NaN when any market handle is unlinked or the maturity is NaN,
  /// case 2 of the [failure convention](crate::traits::ModelPricer#how-pricing-fails).
  fn calculate(&self, opt: &EuropeanOption) -> StandardResult {
    let (model, (s, k, r, q, tau)) = self.model_and_query(opt);
    let ot = opt.option_type;
    let npv = model.price_option(s, k, r, q, tau, ot);
    StandardResult::with_greeks(npv, model.greeks(s, k, r, q, tau, ot))
  }
}

impl PricingEngine<DigitalOption> for AnalyticBsEngine {
  type Result = StandardResult;

  /// NPV only, at `b = r − q` whatever [`coc`](Self::coc) says; NaN when the maturity is NaN or
  /// any market handle is unlinked.
  fn calculate(&self, opt: &DigitalOption) -> StandardResult {
    let (s, k, r, q, tau) = self.digital_query(opt);
    let sigma = Self::read_quote(&self.volatility);
    let ot = opt.option_type;
    let npv = match opt.kind {
      DigitalKind::CashOrNothing { cash } => {
        CashOrNothingPricer::new(cash, sigma).price_option(s, k, r, q, tau, ot)
      }
      DigitalKind::AssetOrNothing => {
        AssetOrNothingPricer::new(sigma).price_option(s, k, r, q, tau, ot)
      }
    };
    StandardResult::npv_only(npv)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::OptionType;
  use crate::calendar::Maturity;
  use crate::traits::InstrumentExt;
  use crate::traits::PricingResult;

  #[test]
  fn european_call_atm_round_trip() {
    let opt = EuropeanOption::new(100.0, OptionType::Call, Maturity::Tau(1.0));
    let engine = AnalyticBsEngine::with_constants(100.0, 0.20, 0.05, 0.0);
    let r = engine.calculate(&opt);
    assert!(r.npv() > 0.0);
    let g = r.greeks().unwrap();
    assert!(g.delta > 0.0 && g.delta < 1.0);
    assert!(g.gamma > 0.0);
    assert!(g.vega > 0.0);
  }

  #[test]
  fn european_put_call_parity_via_engine() {
    let call = EuropeanOption::new(100.0, OptionType::Call, Maturity::Tau(1.0));
    let put = EuropeanOption::new(100.0, OptionType::Put, Maturity::Tau(1.0));
    let engine = AnalyticBsEngine::with_constants(100.0, 0.20, 0.05, 0.02);
    let c = engine.calculate(&call).npv();
    let p = engine.calculate(&put).npv();
    let parity = 100.0 * (-0.02_f64).exp() - 100.0 * (-0.05_f64).exp();
    assert!((c - p - parity).abs() < 1e-8);
  }

  #[test]
  fn instrument_ext_npv_shortcut() {
    let opt = EuropeanOption::new(110.0, OptionType::Call, Maturity::Tau(0.5));
    let engine = AnalyticBsEngine::with_constants(100.0, 0.25, 0.04, 0.0);
    let direct = engine.calculate(&opt).npv();
    let via_ext = opt.npv(&engine);
    assert!((direct - via_ext).abs() < 1e-15);
  }

  #[test]
  fn relinking_volatility_changes_npv() {
    let opt = EuropeanOption::new(100.0, OptionType::Call, Maturity::Tau(1.0));
    let vol_quote = Arc::new(SimpleQuote::new(0.20));
    let vol_handle = Handle::new(vol_quote.clone());
    let engine = AnalyticBsEngine::new(
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

  /// The engine's Greeks are exactly [`BsmPricer::greeks`] at the handles'
  /// current values and the instrument's own τ — all nine members, not just
  /// the three whose sign `european_call_atm_round_trip` checks. Guards the
  /// engine's wiring (quote reads, τ resolution, option type) rather than
  /// the Greek formulas, which `bsm/tests.rs` pins.
  #[test]
  fn engine_greeks_match_the_model_aggregate() {
    let opt = EuropeanOption::new(105.0, OptionType::Put, Maturity::Tau(0.75));
    let engine = AnalyticBsEngine::with_constants(100.0, 0.25, 0.05, 0.02);
    let got = engine.calculate(&opt).greeks().unwrap();
    let want = BsmPricer::new(0.25, BsmCoc::Merton1973).greeks(
      100.0,
      105.0,
      0.05,
      0.02,
      0.75,
      OptionType::Put,
    );
    assert_eq!(got.as_array(), want.as_array());
  }

  /// 2026-01-02 → 2027-01-02 is 365 days, so Act/365F gives τ = 1.0 exactly.
  #[test]
  fn engine_prices_a_dated_option_like_its_tau_equivalent() {
    let eval = chrono::NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();
    let expiry = chrono::NaiveDate::from_ymd_opt(2027, 1, 2).unwrap();
    let dated = EuropeanOption::new(100.0, OptionType::Call, Maturity::Dates { eval, expiry });
    let by_tau = EuropeanOption::new(100.0, OptionType::Call, Maturity::Tau(1.0));
    let engine = AnalyticBsEngine::with_constants(100.0, 0.20, 0.05, 0.0);
    let a = engine.calculate(&dated).npv();
    let b = engine.calculate(&by_tau).npv();
    assert!((a - b).abs() < 1e-12, "dated={a}, tau=1.0 gives {b}");
  }

  /// An unlinked spot reports NaN, as a NaN maturity does, instead of pricing at `s = 0`.
  #[test]
  fn an_unlinked_handle_reports_like_a_nan_maturity() {
    let put = EuropeanOption::new(100.0, OptionType::Put, Maturity::Tau(1.0));
    let no_spot = AnalyticBsEngine::new(
      Handle::empty(),
      Handle::new(Arc::new(SimpleQuote::new(0.20))),
      Handle::new(Arc::new(SimpleQuote::new(0.05))),
      Handle::new(Arc::new(SimpleQuote::new(0.0))),
    );
    let npv = no_spot.calculate(&put).npv();
    assert!(npv.is_nan(), "unlinked spot must not price, got {npv}");

    let no_tau = EuropeanOption::new(100.0, OptionType::Put, Maturity::Tau(f64::NAN));
    let full = AnalyticBsEngine::with_constants(100.0, 0.20, 0.05, 0.0);
    assert!(
      full.calculate(&no_tau).npv().is_nan(),
      "a NaN maturity must not price either"
    );
  }

  /// Each of the four handles independently, on both instrument types the
  /// engine serves, and through the Greeks as well as the NPV. Naming them
  /// one at a time is what stops the guard passing because some *other*
  /// input happened to be missing.
  #[test]
  fn every_unlinked_handle_poisons_npv_and_greeks() {
    let call = EuropeanOption::new(100.0, OptionType::Call, Maturity::Tau(1.0));
    let digital = DigitalOption::cash_or_nothing(100.0, OptionType::Call, 1.0, Maturity::Tau(1.0));
    let linked = |v: f64| Handle::new(Arc::new(SimpleQuote::new(v)));
    let quotes = [100.0, 0.20, 0.05, 0.0];

    for missing in 0..4 {
      let mut h = [
        linked(quotes[0]),
        linked(quotes[1]),
        linked(quotes[2]),
        linked(quotes[3]),
      ];
      h[missing] = Handle::empty();
      let [s, vol, r, q] = h;
      let engine = AnalyticBsEngine::new(s, vol, r, q);

      let res = engine.calculate(&call);
      assert!(
        res.npv().is_nan(),
        "handle {missing}: european npv {}",
        res.npv()
      );
      assert!(
        res.greeks().unwrap().as_array().iter().all(|g| g.is_nan()),
        "handle {missing}: greeks must be all-NaN"
      );
      let d = PricingEngine::<DigitalOption>::calculate(&engine, &digital).npv();
      assert!(d.is_nan(), "handle {missing}: digital npv {d}");
    }
  }

  #[test]
  fn an_expired_option_npvs_its_payoff_and_has_no_greek() {
    let engine = AnalyticBsEngine::with_constants(110.0, 0.20, 0.05, 0.01);
    let res = engine.calculate(&EuropeanOption::new(
      100.0,
      OptionType::Call,
      Maturity::Tau(0.0),
    ));
    assert_eq!(res.npv(), 10.0);
    let greeks = res.greeks().unwrap().as_array();
    assert!(greeks.iter().all(|g| g.is_nan()), "{greeks:?}");
    let put = EuropeanOption::new(100.0, OptionType::Put, Maturity::Tau(0.0));
    assert_eq!(engine.calculate(&put).npv(), 0.0);
    let digital =
      |opt: &DigitalOption| PricingEngine::<DigitalOption>::calculate(&engine, opt).npv();
    let cash = DigitalOption::cash_or_nothing(100.0, OptionType::Call, 10.0, Maturity::Tau(0.0));
    assert_eq!(digital(&cash), 10.0);
    assert_eq!(
      digital(&DigitalOption::asset_or_nothing(
        100.0,
        OptionType::Put,
        Maturity::Tau(0.0)
      )),
      0.0
    );
    let at_strike =
      DigitalOption::cash_or_nothing(110.0, OptionType::Call, 10.0, Maturity::Tau(0.0));
    assert_eq!(digital(&at_strike), 0.0);
  }

  #[test]
  fn digital_cash_or_nothing() {
    let opt = DigitalOption::cash_or_nothing(100.0, OptionType::Call, 1.0, Maturity::Tau(1.0));
    let engine = AnalyticBsEngine::with_constants(100.0, 0.20, 0.05, 0.0);
    let r = engine.calculate(&opt);
    assert!(r.npv() > 0.0 && r.npv() < 1.0);
  }

  /// The digital arm is exactly its model at the handles' current values and
  /// the instrument's own τ — the same guard
  /// `engine_greeks_match_the_model_aggregate` gives the European arm, and
  /// what makes "one file, one convention" checkable rather than asserted.
  /// Both kinds and both option types, since the engine's job here is the
  /// wiring (quote reads, τ, strike, payout, call/put dispatch) and not the
  /// formulas, which `digital_tests.rs` pins.
  #[test]
  fn digital_npv_matches_the_model_at_the_same_query() {
    let engine = AnalyticBsEngine::with_constants(100.0, 0.25, 0.05, 0.02);
    let query = (100.0, 105.0, 0.05, 0.02, 0.75);
    let (s, k, r, q, tau) = query;

    for ot in [OptionType::Call, OptionType::Put] {
      let cash = DigitalOption::cash_or_nothing(k, ot, 10.0, Maturity::Tau(tau));
      let got = PricingEngine::<DigitalOption>::calculate(&engine, &cash).npv();
      let want = CashOrNothingPricer::new(10.0, 0.25).price_option(s, k, r, q, tau, ot);
      assert_eq!(got, want, "cash-or-nothing {ot:?}");

      let asset = DigitalOption::asset_or_nothing(k, ot, Maturity::Tau(tau));
      let got = PricingEngine::<DigitalOption>::calculate(&engine, &asset).npv();
      let want = AssetOrNothingPricer::new(0.25).price_option(s, k, r, q, tau, ot);
      assert_eq!(got, want, "asset-or-nothing {ot:?}");
    }
  }

  /// A dated digital prices bit-identically to the τ its dates resolve to.
  #[test]
  fn a_date_built_digital_prices_like_the_tau_the_dates_resolve_to() {
    let dates = Maturity::Dates {
      eval: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
      expiry: chrono::NaiveDate::from_ymd_opt(2024, 7, 1).unwrap(),
    };
    let tau = 182.0 / 365.0;
    let engine = AnalyticBsEngine::with_constants(100.0, 0.25, 0.05, 0.02);

    for ot in [OptionType::Call, OptionType::Put] {
      let dated = DigitalOption::cash_or_nothing(105.0, ot, 10.0, dates);
      let by_tau = DigitalOption::cash_or_nothing(105.0, ot, 10.0, Maturity::Tau(tau));
      let got = PricingEngine::<DigitalOption>::calculate(&engine, &dated).npv();
      assert!(got.is_finite(), "dated cash-or-nothing {ot:?} priced {got}");
      assert_eq!(
        got,
        PricingEngine::<DigitalOption>::calculate(&engine, &by_tau).npv(),
        "dated cash-or-nothing {ot:?} must match its explicit-tau twin"
      );

      let dated = DigitalOption::asset_or_nothing(105.0, ot, dates);
      let by_tau = DigitalOption::asset_or_nothing(105.0, ot, Maturity::Tau(tau));
      let got = PricingEngine::<DigitalOption>::calculate(&engine, &dated).npv();
      assert!(
        got.is_finite(),
        "dated asset-or-nothing {ot:?} priced {got}"
      );
      assert_eq!(
        got,
        PricingEngine::<DigitalOption>::calculate(&engine, &by_tau).npv(),
        "dated asset-or-nothing {ot:?} must match its explicit-tau twin"
      );
    }
  }

  #[test]
  fn a_digital_with_a_nan_maturity_does_not_price() {
    let engine = AnalyticBsEngine::with_constants(100.0, 0.25, 0.05, 0.02);
    let opt =
      DigitalOption::cash_or_nothing(105.0, OptionType::Call, 10.0, Maturity::Tau(f64::NAN));
    let npv = PricingEngine::<DigitalOption>::calculate(&engine, &opt).npv();
    assert!(npv.is_nan(), "a NaN maturity must not price, got {npv}");
  }
}
