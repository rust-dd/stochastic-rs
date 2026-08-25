//! Closed-form Black-Scholes engine for [`EuropeanOption`] /
//! [`DigitalOption`].
//!
//! Wraps [`BSMPricer`] / [`CashOrNothingPricer`] / [`AssetOrNothingPricer`]
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
use crate::pricing::BSMCoc;
use crate::pricing::BSMPricer;
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
pub struct AnalyticBSEngine {
  pub s: Handle<SimpleQuote<f64>>,
  pub volatility: Handle<SimpleQuote<f64>>,
  pub r: Handle<SimpleQuote<f64>>,
  pub dividend_yield: Handle<SimpleQuote<f64>>,
  pub coc: BSMCoc,
}

impl AnalyticBSEngine {
  /// Build from explicit handles. Defaults to `BSMCoc::Merton1973` (equity
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
  ///
  /// # Behaviour change in 3.0
  ///
  /// [`BSMCoc::GarmanKohlhagen1983`] used to **panic** here: the old
  /// pricer resolved its carry from a separate `r_d` field that this
  /// engine never set, so the lookup hit an `expect` on `None`. The
  /// engine now passes its own rate and dividend-yield handles as the
  /// domestic and foreign rates — the standard Garman-Kohlhagen
  /// embedding — and returns a price instead. Code that relied on the
  /// panic to detect an unsupported convention will now get a number.
  pub fn with_coc(mut self, coc: BSMCoc) -> Self {
    self.coc = coc;
    self
  }

  /// Current value of a market handle, or [`f64::NAN`] when the handle is
  /// unlinked.
  ///
  /// An unset handle is *missing data*, not a zero market. Reading it as
  /// `0.0` priced the option at a spot, vol or rate the caller never
  /// supplied and returned a finite, confident, fictitious NPV; `NaN`
  /// propagates instead, so `npv().is_finite()` is a real check. This is
  /// the same answer the crate gives for a missing maturity
  /// ([`TimeExt::tau_or_from_dates`]) and for a missing curve quote
  /// (`market::rate_helper`, which drops the helper rather than substitute
  /// a rate), so every input this engine reads now fails the same way.
  fn read_quote(handle: &Handle<SimpleQuote<f64>>) -> f64 {
    handle.current().map(|q| q.value()).unwrap_or(f64::NAN)
  }

  /// The [`BSMPricer`] model the current handles describe, paired with the
  /// `(s, k, r, q, tau)` query point `opt` asks for. The instrument owns
  /// the maturity, so τ is resolved through its own
  /// [`TimeExt`](crate::traits::TimeExt) rather than by the model.
  fn model_and_query(&self, opt: &EuropeanOption) -> (BSMPricer, (f64, f64, f64, f64, f64)) {
    let model = BSMPricer::new(Self::read_quote(&self.volatility), self.coc);
    let query = (
      Self::read_quote(&self.s),
      opt.strike,
      Self::read_quote(&self.r),
      Self::read_quote(&self.dividend_yield),
      opt.tau_or_from_dates(),
    );
    (model, query)
  }

  /// The `(s, k, r, q, tau)` query point a [`DigitalOption`] asks for.
  ///
  /// The digital counterpart of [`model_and_query`](Self::model_and_query),
  /// minus the model: which of the two digital pricers to build depends on
  /// `opt.kind`, so [`calculate`](PricingEngine::calculate) picks that and
  /// this supplies the query both arms share.
  ///
  /// τ is read straight off `opt.tau` rather than through
  /// [`TimeExt`](crate::traits::TimeExt), because [`DigitalOption`] does not
  /// implement it — the `(eval, expiry)` pair it carries is inert. That is
  /// the one input the two arms of this engine still resolve differently.
  fn digital_query(&self, opt: &DigitalOption) -> (f64, f64, f64, f64, f64) {
    (
      Self::read_quote(&self.s),
      opt.strike,
      Self::read_quote(&self.r),
      Self::read_quote(&self.dividend_yield),
      opt.tau.unwrap_or(f64::NAN),
    )
  }
}

impl PricingEngine<EuropeanOption> for AnalyticBSEngine {
  type Result = StandardResult;

  /// The NPV and every Greek are [`f64::NAN`] when any market handle is
  /// unlinked or the instrument's maturity is unset — case 2 of the crate's
  /// [failure convention](crate::traits::ModelPricer#how-pricing-fails). An
  /// unset handle is missing data, not a zero market.
  fn calculate(&self, opt: &EuropeanOption) -> StandardResult {
    let (model, (s, k, r, q, tau)) = self.model_and_query(opt);
    let ot = opt.option_type;
    let npv = model.price_option(s, k, r, q, tau, ot);
    StandardResult::with_greeks(npv, model.greeks(s, k, r, q, tau, ot))
  }
}

impl PricingEngine<DigitalOption> for AnalyticBSEngine {
  type Result = StandardResult;

  /// Built the same way as the [`EuropeanOption`] arm above: one model from
  /// the market handles and the instrument's own contract terms, one
  /// `(s, k, r, q, tau)` query, one
  /// [`price_option`](ModelPricer::price_option) call. The digital pricers
  /// hold model state only, so the instance costs two `f64`s and carries
  /// nothing from the query — where this arm used to build a fresh
  /// eight-field pricer per call and read the price back off the struct.
  ///
  /// The digital pricers fix the cost of carry at $b = r - q$, so this arm
  /// ignores [`coc`](Self::coc); only the European arm honours it.
  ///
  /// NPV only — the closed-form digital Greeks are not wired through this
  /// engine, so [`PricingResult::greeks`](crate::traits::PricingResult) stays
  /// at [`Greeks::nan`](crate::traits::Greeks::nan).
  ///
  /// The NPV is [`f64::NAN`] when `opt.tau` is unset, following the same
  /// missing-data convention as [`crate::traits::TimeExt::tau_or_from_dates`],
  /// **and** when any of the `s`, `volatility`, `r` or `dividend_yield`
  /// handles is unlinked. The two used to disagree: a missing maturity
  /// poisoned the result while a missing spot read as `0.0` and priced the
  /// digital at a zero spot, so one unpopulated input reported itself and the
  /// other did not.
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

  /// The engine's Greeks are exactly [`BSMPricer::greeks`] at the handles'
  /// current values and the instrument's own τ — all nine members, not just
  /// the three whose sign `european_call_atm_round_trip` checks. Guards the
  /// engine's wiring (quote reads, τ resolution, option type) rather than
  /// the Greek formulas, which `bsm/tests.rs` pins.
  #[test]
  fn engine_greeks_match_the_model_aggregate() {
    let opt = EuropeanOption::new_tau(105.0, OptionType::Put, 0.75);
    let engine = AnalyticBSEngine::with_constants(100.0, 0.25, 0.05, 0.02);
    let got = engine.calculate(&opt).greeks().unwrap();
    let want = BSMPricer::new(0.25, BSMCoc::Merton1973).greeks(
      100.0,
      105.0,
      0.05,
      0.02,
      0.75,
      OptionType::Put,
    );
    assert_eq!(got.as_array(), want.as_array());
  }

  /// A dated instrument resolves its own τ through `TimeExt` now that the
  /// model holds no dates. 2026-01-02 → 2027-01-02 is 365 days in a
  /// non-leap year, so Act/365F gives τ = 1.0 exactly.
  #[test]
  fn engine_prices_a_dated_option_like_its_tau_equivalent() {
    let eval = chrono::NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();
    let expiry = chrono::NaiveDate::from_ymd_opt(2027, 1, 2).unwrap();
    let dated = EuropeanOption::new_dates(100.0, OptionType::Call, eval, expiry);
    let by_tau = EuropeanOption::new_tau(100.0, OptionType::Call, 1.0);
    let engine = AnalyticBSEngine::with_constants(100.0, 0.20, 0.05, 0.0);
    let a = engine.calculate(&dated).npv();
    let b = engine.calculate(&by_tau).npv();
    assert!((a - b).abs() < 1e-12, "dated={a}, tau=1.0 gives {b}");
  }

  /// One engine, one instrument, two ways to leave an input unpopulated —
  /// and they now report the same way. Before the fix an unlinked spot read
  /// as `0.0` and this priced a put at `K·e^{-rτ} = 95.12`: a finite,
  /// well-scaled NPV sourced from a spot the caller never supplied, which no
  /// `is_finite()` check downstream could tell from a real one. The unset
  /// maturity next to it already gave `NaN`.
  #[test]
  fn an_unlinked_handle_reports_like_an_unset_maturity() {
    let put = EuropeanOption::new_tau(100.0, OptionType::Put, 1.0);
    let no_spot = AnalyticBSEngine::new(
      Handle::empty(),
      Handle::new(Arc::new(SimpleQuote::new(0.20))),
      Handle::new(Arc::new(SimpleQuote::new(0.05))),
      Handle::new(Arc::new(SimpleQuote::new(0.0))),
    );
    let npv = no_spot.calculate(&put).npv();
    assert!(npv.is_nan(), "unlinked spot must not price, got {npv}");

    let no_tau = EuropeanOption { tau: None, ..put };
    let full = AnalyticBSEngine::with_constants(100.0, 0.20, 0.05, 0.0);
    assert!(
      full.calculate(&no_tau).npv().is_nan(),
      "unset maturity must not price either"
    );
  }

  /// Each of the four handles independently, on both instrument types the
  /// engine serves, and through the Greeks as well as the NPV. Naming them
  /// one at a time is what stops the guard passing because some *other*
  /// input happened to be missing.
  #[test]
  fn every_unlinked_handle_poisons_npv_and_greeks() {
    let call = EuropeanOption::new_tau(100.0, OptionType::Call, 1.0);
    let digital = DigitalOption::cash_or_nothing(100.0, OptionType::Call, 1.0, 1.0);
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
      let engine = AnalyticBSEngine::new(s, vol, r, q);

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
  fn digital_cash_or_nothing() {
    let opt = DigitalOption::cash_or_nothing(100.0, OptionType::Call, 1.0, 1.0);
    let engine = AnalyticBSEngine::with_constants(100.0, 0.20, 0.05, 0.0);
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
    let engine = AnalyticBSEngine::with_constants(100.0, 0.25, 0.05, 0.02);
    let query = (100.0, 105.0, 0.05, 0.02, 0.75);
    let (s, k, r, q, tau) = query;

    for ot in [OptionType::Call, OptionType::Put] {
      let cash = DigitalOption::cash_or_nothing(k, ot, 10.0, tau);
      let got = PricingEngine::<DigitalOption>::calculate(&engine, &cash).npv();
      let want = CashOrNothingPricer::new(10.0, 0.25).price_option(s, k, r, q, tau, ot);
      assert_eq!(got, want, "cash-or-nothing {ot:?}");

      let asset = DigitalOption::asset_or_nothing(k, ot, tau);
      let got = PricingEngine::<DigitalOption>::calculate(&engine, &asset).npv();
      let want = AssetOrNothingPricer::new(0.25).price_option(s, k, r, q, tau, ot);
      assert_eq!(got, want, "asset-or-nothing {ot:?}");
    }
  }
}
