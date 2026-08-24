//! Compile-time inventory of every pricer/engine struct in this crate.
//!
//! A pricer added without a family assignment fails to compile here, which is
//! the point: prose counts drift, `static_assertions`-style trait bounds do
//! not. Each list below is one of the families the A2 design defines.
//!
//! Derived from
//! `grep -rnE "pub struct [A-Za-z0-9_]*(Pricer|Engine)\b" --include='*.rs' stochastic-rs-quant/src`
//! — 71 structs, of which 21 live in `python/` (`Py`-prefixed PyO3 wrappers
//! that hold the wrapped type as an `inner` field and carry no trait of
//! their own — e.g. `PyBSMPricer { inner: BSMPricer }`). 50 structs remain.
//! The `\b` matters: it drops the 9 `*PricerBuilder` helpers and
//! `PortfolioEngineConfig`, which the bare `.*(?:Pricer|Engine)` regex an
//! earlier revision of this header cited would have swept in, making that
//! revision's stated command produce 81 rather than the 71 it claimed.
//!
//! 19 of the 50 implement one of [`PricerExt`], [`ModelPricer`], or
//! [`PricingEngine`]: **12** carry `ModelPricer` (the trait this registry
//! exists to guard), **5** the older `PricerExt`, **2** `PricingEngine`.
//! Every one of those three numbers is the length of the matching macro
//! invocation below and nothing else — re-derive rather than
//! arithmetic-adjust them, with (substituting the macro name):
//!
//! ```text
//! awk '/^assert_model_pricer!\(/,/^\);/' stochastic-rs-quant/tests/pricer_registry.rs | grep -c '^  [A-Z]'
//! ```
//!
//! The names are deliberately *not* repeated in this prose: the macro
//! invocation is the list, and a second copy of it here is a copy that can
//! drift.
//!
//! Per the A2 design (`docs/superpowers/specs/2026-08-23-a2-quant-consistency-design.md`,
//! decision D1) and its Task 5, 9 of the 10 `PricerExt` structs that
//! existed after Task 5a migrate to `ModelPricer`; `KirkSpreadPricer` is
//! explicitly excluded and instead joins the multi-asset no-trait family
//! once `PricerExt` is retired (Task 6). Task 3 gave `ModelPricer` to the 4
//! digital-option structs, which were `ORPHAN` before it ran; Task 5a added
//! `BSMPricer`; Task 5b is migrating the remaining nine one at a time, so
//! the two counts above move in lockstep until `assert_pricer_ext!` holds
//! `KirkSpreadPricer` alone. Each trait gets its own compile-checked list
//! below so a claim about *which* trait a struct carries can drift no more
//! than the claim that it carries one at all.

use stochastic_rs_quant::bonds::Cir;
use stochastic_rs_quant::bonds::HullWhite;
use stochastic_rs_quant::bonds::Vasicek;
use stochastic_rs_quant::instruments::EuropeanOption;
use stochastic_rs_quant::pricing::AnalyticBSEngine;
use stochastic_rs_quant::pricing::AnalyticHestonEngine;
use stochastic_rs_quant::pricing::AssetOrNothingPricer;
use stochastic_rs_quant::pricing::BSMPricer;
use stochastic_rs_quant::pricing::BjerksundStensland2002Pricer;
use stochastic_rs_quant::pricing::CashOrNothingPricer;
use stochastic_rs_quant::pricing::GapPricer;
use stochastic_rs_quant::pricing::HestonPricer;
use stochastic_rs_quant::pricing::HestonSlvPricer;
use stochastic_rs_quant::pricing::KirkSpreadPricer;
use stochastic_rs_quant::pricing::RBergomiPricer;
use stochastic_rs_quant::pricing::SuperSharePricer;
use stochastic_rs_quant::pricing::asian::AsianPricer;
use stochastic_rs_quant::pricing::finite_difference::FiniteDifferencePricer;
use stochastic_rs_quant::pricing::heston_stoch_corr::HestonStochCorrPricer;
use stochastic_rs_quant::pricing::malliavin_gbm::GbmMalliavinPricer;
use stochastic_rs_quant::pricing::merton_jump::Merton1976Pricer;
use stochastic_rs_quant::pricing::sabr::SabrPricer;
use stochastic_rs_quant::pricing::snell_envelope::SnellEnvelopePricer;
use stochastic_rs_quant::traits::ModelPricer;
use stochastic_rs_quant::traits::PricerExt;
use stochastic_rs_quant::traits::PricingEngine;
use stochastic_rs_quant::traits::ShortRatePricer;

/// Asserts `$t` implements [`ModelPricer`] at compile time.
macro_rules! assert_model_pricer {
  ($($t:ty),* $(,)?) => { $(const _: fn() = || {
    fn assert_impl<T: ModelPricer + ?Sized>() {}
    assert_impl::<$t>();
  };)* };
}

/// Asserts `$t` implements the legacy [`PricerExt`] surface at compile time.
macro_rules! assert_pricer_ext {
  ($($t:ty),* $(,)?) => { $(const _: fn() = || {
    fn assert_impl<T: PricerExt + ?Sized>() {}
    assert_impl::<$t>();
  };)* };
}

/// Asserts `$t` implements [`PricingEngine<I>`] at compile time for the
/// paired instrument type `$i`.
macro_rules! assert_pricing_engine {
  ($(($t:ty, $i:ty)),* $(,)?) => { $(const _: fn() = || {
    fn assert_impl<T: PricingEngine<I>, I>() {}
    assert_impl::<$t, $i>();
  };)* };
}

/// Asserts `$t` implements the short-rate bond family's [`ShortRatePricer`]
/// at compile time.
macro_rules! assert_short_rate_pricer {
  ($($t:ty),* $(,)?) => { $(const _: fn() = || {
    fn assert_impl<T: ShortRatePricer + ?Sized>() {}
    assert_impl::<$t>();
  };)* };
}

// Structs on the decoupled `price_call(s, k, r, q, tau)` surface used by
// calibration and vol-surface construction: the 4 digital options (Task 3),
// the 2 original members, `BSMPricer` (Task 5a), and Task 5b's arrivals.
// Reaches 16 when 5b has migrated all nine of the `assert_pricer_ext!`
// members below except `KirkSpreadPricer`.
//
// Several members override `price_put` rather than taking the trait's
// vanilla put-call-parity default. `BSMPricer` and `AsianPricer` because
// their cost-of-carry factor is `exp((b - r) * tau)`, which equals the
// default's `exp(-q * tau)` only when `b = r - q` — false for
// `BSMCoc::Bsm1973` at `q != 0` and for `Black1976` / `Asay1982`, and only
// on a measure-zero line for the Asian pricer's averaged-underlying carry.
// `BjerksundStensland2002Pricer` and `SnellEnvelopePricer` because European
// put-call parity does not hold for an American option at all: its put
// carries an early-exercise premium the call does not.
assert_model_pricer!(
  AsianPricer,
  AssetOrNothingPricer,
  BjerksundStensland2002Pricer,
  BSMPricer,
  CashOrNothingPricer,
  GapPricer,
  GbmMalliavinPricer,
  HestonSlvPricer,
  RBergomiPricer,
  SabrPricer,
  SnellEnvelopePricer,
  SuperSharePricer,
);

// Single-underlying options on the legacy bundled-market-data `PricerExt`
// surface (`calculate_call_put` / `calculate_price`, no `(s, k, r, q, tau)`
// query point) but not yet `ModelPricer`.
//
// `KirkSpreadPricer` sits here rather than in `no_trait_by_design`'s
// multi-asset family only because it still has `PricerExt` *today* — it
// prices a two-asset spread, so `ModelPricer`'s single strike fits it no
// better than it fits `MargrabePricer`, and Task 5 Step 1 of the A2 plan
// names it explicitly: "`KirkSpreadPricer` takes two forwards and is
// excluded by the design". It is not migrating to `ModelPricer`; once
// `PricerExt` is retired (Task 6) it becomes trait-less like its multi-asset
// siblings. Filed here by what it implements now, not by its final family.
assert_pricer_ext!(
  FiniteDifferencePricer,
  HestonPricer,
  HestonStochCorrPricer,
  KirkSpreadPricer,
  Merton1976Pricer,
);

// QuantLib-style decoupled engines (`Instrument` + `PricingEngine<I>` +
// `PricingResult`, see `traits::instrument`) rather than `ModelPricer`. This
// is a considered second pricing architecture, not a gap: `Instrument`
// describes the payoff, the engine owns model/market/method and reacts to
// market-data updates, and one engine can serve several instrument types
// (`AnalyticBSEngine` also prices `DigitalOption`, checked here only
// against `EuropeanOption`).
assert_pricing_engine!(
  (AnalyticBSEngine, EuropeanOption),
  (AnalyticHestonEngine, EuropeanOption),
);

// The short-rate bond family (Task 2). `Cir`, `HullWhite`, `Vasicek` are
// named for their model, not `*Pricer`/`*Engine`, so this file's header
// derivation command never sees them —
// they sit outside the 71/21/50/19 header counts entirely, not folded into
// any of them. Registered here anyway because `ShortRatePricer` is a real
// family with exactly these three implementors and D6's guard is worth
// having for it too.
assert_short_rate_pricer!(Cir, HullWhite, Vasicek);

mod no_trait_by_design {
  //! Families that deliberately carry no `ModelPricer` implementation, with
  //! the reason. Listing them here is what makes the omission deliberate
  //! rather than an oversight — see the A2 design's D1. Some of these do
  //! carry `PricerExt` or `PricingEngine` instead (see the lists above);
  //! this module is specifically about `ModelPricer`'s
  //! `price_call(s, k, r, q, tau)` shape not fitting.
  //!
  //! - Multi-asset (`ArithmeticBasketLevyPricer`, `GeometricBasketPricer`,
  //!   `McBasketPricer`, `MargrabePricer`, `McSpreadPricer`,
  //!   `StulzRainbowPricer`, `McRainbowPricer`): `MargrabePricer` is an
  //!   exchange option with no strike; the basket and spread pricers bundle
  //!   N legs and weights into the struct; the rainbow pricers price the
  //!   best/worst of N assets. A shared signature would need an
  //!   `Option<f64>` strike and a variable-length underlying list.
  //!   `KirkSpreadPricer` belongs conceptually to this family too (D1 names
  //!   it explicitly) but is listed under `assert_pricer_ext!` above instead,
  //!   because it still carries `PricerExt` until Task 6 retires it.
  //! - Path-dependent (`BarrierPricer`, `DoubleBarrierPricer`,
  //!   `MCBarrierPricer`, `FixedLookbackPricer`, `FloatingLookbackPricer`,
  //!   `CliquetPricer`, `McCliquetPricer`, `AutocallablePricer`,
  //!   `BermudanLsmPricer`, `SimpleChooserPricer`, `ComplexChooserPricer`,
  //!   `ForwardStartPricer`, `CompoundPricer`, `VarianceSwapPricer`,
  //!   `VolatilitySwapPricer`): each bundles its own contract parameters
  //!   (barrier level, lookback window, cliquet reset schedule, autocall
  //!   trigger/coupon schedule, Bermudan exercise dates, chooser decision
  //!   date, ...) into the struct and prices off that, not off a single
  //!   `(s, k, r, q, tau)` query point.
  //! - Fourier / characteristic-function engines (`CarrMadanPricer`,
  //!   `GilPelaezPricer`, `LewisPricer`, `FrftCarrMadanPricer`, `CosEngine`,
  //!   `CosPricer`, `CgmysvPricer`): the first five take
  //!   `model: &impl FourierModelExt` per call, and `CosPricer` builds a
  //!   `RegimeSwitchingModel` internally from raw inputs and prices through
  //!   it — all six route through the Fourier/characteristic-function
  //!   machinery. Engines parameterised *by* a model, not models: the
  //!   blanket `impl<T: FourierModelExt> ModelPricer for T`
  //!   (`pricing/fourier/mod.rs`) already gives the models themselves
  //!   (`HestonFourier`, `CgmysvModel`, `RegimeSwitchingModel`, ...) their
  //!   `ModelPricer`. `CgmysvPricer` is the true outlier in this group: it
  //!   takes no model parameter and touches no characteristic function at
  //!   all — it duplicates `CgmysvParams` as struct fields and is a
  //!   self-contained Monte Carlo pricer returning `McResult` (price ±
  //!   standard error), so `ModelPricer`'s plain-`f64` return doesn't fit it
  //!   either. It is listed here as `CgmysvModel`'s Monte Carlo counterpart,
  //!   not because it shares the other six's call shape.
  //! - Non-option engines (`CashflowPricer`, `PortfolioEngine`): not
  //!   single-underlying option pricers at all.
  //!   `CashflowPricer::leg_npv`/`cashflow_npv` discount a bond/loan
  //!   cashflow schedule against a curve stack; `PortfolioEngine::optimize`/
  //!   `score_momentum`/`build_momentum` allocate weights across N assets.
  //!   Neither has a single option struck at `K` on `S` to put through
  //!   `price_call(s, k, r, q, tau)`. The design spec's own count implies
  //!   them (31 orphans post-Task-3, "29 of them in `pricing/`" — these two
  //!   are the other 2) without naming them; this bullet is the missing name.
}
