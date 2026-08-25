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
//! The `\b` matters: the bare `.*(?:Pricer|Engine)` regex an earlier
//! revision of this header cited reports 72 against the same tree, one
//! more, and that one is `PortfolioEngineConfig`. The gap used to be ten —
//! the other nine were `*PricerBuilder` helpers, of which Task 5 deleted
//! eight and Task 6 the last (`KirkSpreadPricerBuilder`).
//!
//! 18 of the 50 implement either [`ModelPricer`] or [`PricingEngine`]:
//! **16** carry `ModelPricer` (the trait this registry exists to guard) and
//! **2** `PricingEngine`.
//!
//! The `assert_model_pricer!` list below is **18** long, not 16, and the gap
//! is not an error in either number: `LevyModel` and `CrrModel` implement
//! `ModelPricer` without being named `*Pricer` or `*Engine`, so the header
//! derivation command never sees them and they are not among the 50 — the
//! same situation as the short-rate trio further down. They are registered
//! anyway, because a list that guards a trait is worth more than a list that
//! matches a regex. Every count here is the length of the matching macro
//! invocation and nothing else — re-derive rather than arithmetic-adjust:
//!
//! ```text
//! awk '/^assert_model_pricer!\(/,/^\);/' stochastic-rs-quant/tests/pricer_registry.rs | grep -c '^  [A-Z]'
//! awk '/^assert_pricing_engine!\(/,/^\);/' stochastic-rs-quant/tests/pricer_registry.rs | grep -c '^  ('
//! awk '/^assert_vanilla_european_call!\(/,/^\);/' stochastic-rs-quant/tests/pricer_registry.rs | grep -c '^  [A-Z]'
//! awk '/^assert_not_vanilla_european_call!\(/,/^\);/' stochastic-rs-quant/tests/pricer_registry.rs | grep -c '^  [A-Z]'
//! ```
//!
//! Four commands, not one with the macro name substituted: an earlier
//! revision of this header said one would do, but `assert_pricing_engine!`
//! takes `(pricer, instrument)` pairs, so its entries open with a paren
//! and `grep -c '^  [A-Z]'` reports 0 for it.
//!
//! The last two commands partition the first: every `ModelPricer` in this
//! file is classified exactly once as a European vanilla call pricer or as
//! one of the payoffs a Black inversion cannot describe, so **11 + 7 = 18**
//! is an invariant an added pricer breaks until someone decides which side
//! it falls on. That decision used to be made by silence — [`ModelSurface`]
//! blanket-covered every `ModelPricer`, so a digital acquired a vol surface
//! by existing.
//!
//! The names are deliberately *not* repeated in this prose: the macro
//! invocation is the list, and a second copy of it here is a copy that can
//! drift.
//!
//! Per the A2 design (`docs/superpowers/specs/2026-08-23-a2-quant-consistency-design.md`,
//! decision D1), `PricerExt` is retired and this file no longer has a list
//! for it. Task 3 gave `ModelPricer` to the 4 digital-option structs, which
//! were `ORPHAN` before it ran — it did not take their bundled
//! `s`/`k`/`r`/`b`/`tau`/`option_type` fields away, so for a while each of
//! the four answered the same question two ways; the follow-up removed the
//! fields, so all 16 entries below now hold model (and contract) state
//! only. Task 5a added `BSMPricer`; Task 5b migrated
//! nine more, taking `PricerExt` from 10 implementors down to
//! `KirkSpreadPricer` alone; Task 6 took the trait off that one and deleted
//! it. `KirkSpreadPricer` did not migrate to `ModelPricer` — it prices a
//! two-asset spread, so a single strike fits it no better than it fits
//! `MargrabePricer` — and is now filed under `no_trait_by_design` with its
//! multi-asset siblings. Each surviving trait gets its own compile-checked
//! list below so a claim about *which* trait a struct carries can drift no
//! more than the claim that it carries one at all.

use std::marker::PhantomData;

use stochastic_rs_quant::bonds::Cir;
use stochastic_rs_quant::bonds::HullWhite;
use stochastic_rs_quant::bonds::Vasicek;
use stochastic_rs_quant::calibration::levy::LevyModel;
use stochastic_rs_quant::instruments::EuropeanOption;
use stochastic_rs_quant::lattice::equity::CrrModel;
use stochastic_rs_quant::pricing::AnalyticBSEngine;
use stochastic_rs_quant::pricing::AnalyticHestonEngine;
use stochastic_rs_quant::pricing::AssetOrNothingPricer;
use stochastic_rs_quant::pricing::BSMPricer;
use stochastic_rs_quant::pricing::BjerksundStensland2002Pricer;
use stochastic_rs_quant::pricing::CashOrNothingPricer;
use stochastic_rs_quant::pricing::GapPricer;
use stochastic_rs_quant::pricing::HestonPricer;
use stochastic_rs_quant::pricing::HestonSlvPricer;
use stochastic_rs_quant::pricing::HestonStochCorrPricer;
use stochastic_rs_quant::pricing::RBergomiPricer;
use stochastic_rs_quant::pricing::SuperSharePricer;
use stochastic_rs_quant::pricing::asian::AsianPricer;
use stochastic_rs_quant::pricing::finite_difference::FiniteDifferencePricer;
use stochastic_rs_quant::pricing::malliavin_gbm::GbmMalliavinPricer;
use stochastic_rs_quant::pricing::merton_jump::Merton1976Pricer;
use stochastic_rs_quant::pricing::sabr::SabrPricer;
use stochastic_rs_quant::pricing::snell_envelope::SnellEnvelopePricer;
use stochastic_rs_quant::traits::ModelPricer;
use stochastic_rs_quant::traits::PricingEngine;
use stochastic_rs_quant::traits::ShortRatePricer;
use stochastic_rs_quant::traits::VanillaEuropeanCall;

/// Asserts `$t` implements [`ModelPricer`] at compile time.
macro_rules! assert_model_pricer {
  ($($t:ty),* $(,)?) => { $(const _: fn() = || {
    fn assert_impl<T: ModelPricer + ?Sized>() {}
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

/// Asserts `$t` implements [`VanillaEuropeanCall`] at compile time — that
/// its `price_call` is a European vanilla call, which is what earns it a
/// `ModelSurface::vol_surface`.
macro_rules! assert_vanilla_european_call {
  ($($t:ty),* $(,)?) => { $(const _: fn() = || {
    fn assert_impl<T: VanillaEuropeanCall + ?Sized>() {}
    assert_impl::<$t>();
  };)* };
}

/// Probe carrying a `bool` that reports whether `T` implements
/// [`VanillaEuropeanCall`]: the inherent `impl` below applies only when the
/// bound holds and shadows the trait's `false` when it does, so the constant
/// is `true` exactly for implementors.
///
/// Rust has no `where T: !Trait`, and a *missing* impl is precisely what
/// needs pinning here — an added blanket impl, or a marker pasted onto a
/// digital, is the regression this file exists to catch, and both are
/// invisible to a positive-only inventory.
struct VanillaProbe<T: ?Sized>(PhantomData<T>);

trait MaybeVanillaEuropeanCall {
  const IS_VANILLA_EUROPEAN_CALL: bool = false;
}

impl<T: ?Sized> MaybeVanillaEuropeanCall for VanillaProbe<T> {}

impl<T: VanillaEuropeanCall + ?Sized> VanillaProbe<T> {
  const IS_VANILLA_EUROPEAN_CALL: bool = true;
}

/// Asserts `$t` does **not** implement [`VanillaEuropeanCall`], so
/// `$t::vol_surface(..)` does not compile.
///
/// The positive control below the invocation is not decoration: if the
/// shadowing ever stopped resolving, every negative assertion here would
/// pass vacuously and this whole list would become a comment. The control
/// fails to compile in that case.
macro_rules! assert_not_vanilla_european_call {
  ($($t:ty),* $(,)?) => { $(
    const _: () = assert!(!<VanillaProbe<$t>>::IS_VANILLA_EUROPEAN_CALL);
  )* };
}

// Structs on the decoupled `price_call(s, k, r, q, tau)` surface used by
// calibration and vol-surface construction: the 4 digital options (Task 3),
// the 2 original members, `BSMPricer` (Task 5a), and Task 5b's 9 arrivals.
// 5b is complete, so this list has reached the 16 the A2 design predicted,
// plus `LevyModel` and `CrrModel` — real `ModelPricer` implementors that the
// header's `pub struct *(Pricer|Engine)` derivation cannot see, so they were
// missing here while the prose count was correct about the 50.
//
// Every 5b arrival overrides `price_put` rather than taking the trait's
// vanilla put-call-parity default, for one of three reasons:
//
//  - cost of carry: `BSMPricer`, `AsianPricer`, `Merton1976Pricer` carry at
//    `exp((b - r) * tau)`, which equals the default's `exp(-q * tau)` only
//    when `b = r - q` — false for `BSMCoc::Bsm1973` at `q != 0` and for
//    `Black1976` / `Asay1982`, and true only on a measure-zero line for the
//    Asian pricer's averaged-underlying carry;
//  - American exercise: `BjerksundStensland2002Pricer`,
//    `SnellEnvelopePricer` and `FiniteDifferencePricer` price a put that
//    carries an early-exercise premium the call does not, so European
//    parity is not an approximation but the wrong model;
//  - exactness: `SabrPricer`, `HestonPricer`, `HestonStochCorrPricer` and
//    `GbmMalliavinPricer` all have carry `b = r - q`, where the default is
//    mathematically right — but it recomposes the put from the call and so
//    can land an ulp away from the closed form, drop a `max(0)` floor
//    (`HestonStochCorrPricer`, `GbmMalliavinPricer`), or run a second
//    independent Monte Carlo (`GbmMalliavinPricer`).
assert_model_pricer!(
  AsianPricer,
  AssetOrNothingPricer,
  BjerksundStensland2002Pricer,
  BSMPricer,
  CashOrNothingPricer,
  CrrModel<f64>,
  FiniteDifferencePricer,
  GapPricer,
  GbmMalliavinPricer,
  HestonPricer,
  HestonSlvPricer,
  HestonStochCorrPricer,
  LevyModel,
  Merton1976Pricer,
  RBergomiPricer,
  SabrPricer,
  SnellEnvelopePricer,
  SuperSharePricer,
);

// The subset of the list above whose `price_call` is a European vanilla
// call, and so may be inverted through the Black formula into an implied
// vol surface. `ModelSurface` is blanket-implemented over this trait, not
// over `ModelPricer`, so membership here is what makes `vol_surface`
// callable at all.
//
// Absent from the list because a blanket impl in `pricing/fourier/mod.rs`
// covers them: every `FourierModelExt` model. Gil-Pelaez prices
// `S e^{-q tau} P_1 - K e^{-r tau} P_2` — the vanilla European call — for
// all of them, so they are covered by the same one-line reasoning rather
// than one entry each.
//
// `FiniteDifferencePricer` is the one member whose answer is per-instance:
// it holds its exercise style in a field, so it carries the trait and
// reports `NaN` from `vanilla_call_forward` at `OptionStyle::American`.
// `BSMPricer` and `Merton1976Pricer` are the two whose *forward* is
// per-instance, at `b(r, q)` rather than the default `r - q`.
assert_vanilla_european_call!(
  BSMPricer,
  CrrModel<f64>,
  FiniteDifferencePricer,
  GbmMalliavinPricer,
  HestonPricer,
  HestonSlvPricer,
  HestonStochCorrPricer,
  LevyModel,
  Merton1976Pricer,
  RBergomiPricer,
  SabrPricer,
);

// The complement: `ModelPricer`s whose payoff a Black inversion cannot
// describe, listed so the omission is a decision rather than an oversight.
// Each one used to get `vol_surface` from the `ModelPricer` blanket and
// returned a finite, plausible, meaningless surface for it.
//
//  - the four digitals (`AssetOrNothingPricer`, `CashOrNothingPricer`,
//    `GapPricer`, `SuperSharePricer`): the payoff is not `(S_T - K)^+`, so
//    no Black volatility reproduces it. Nothing about the output says so —
//    an asset-or-nothing call is worth `F N(d_1)`, inside the no-arbitrage
//    band `(max(F - K, 0), F)` at every strike, so every point inverts.
//  - `AsianPricer`: a vanilla call, but on the geometric *average* rather
//    than on `S_T`, and carried at `b = (r - q - sigma_A^2 / 6) / 2`. The
//    inversion would return an averaged-underlying volatility labelled as
//    the spot's, near `sigma / sqrt(3)` and so especially easy to mistake
//    for a real one.
//  - `BjerksundStensland2002Pricer`, `SnellEnvelopePricer`: American
//    exercise, and unconditionally so — unlike `FiniteDifferencePricer`
//    there is no European setting to fall back to. An American call at
//    `q = 0` *is* the European one, which is what makes the general case
//    dangerous rather than obviously wrong: at `q = 0.06` both invert to
//    within 0.008 of the model's own volatility, close enough to pass any
//    eyeball check and wrong at every point.
assert_not_vanilla_european_call!(
  AsianPricer,
  AssetOrNothingPricer,
  BjerksundStensland2002Pricer,
  CashOrNothingPricer,
  GapPricer,
  SnellEnvelopePricer,
  SuperSharePricer,
);

// Positive control for the probe above — see
// `assert_not_vanilla_european_call!`. Deliberately *not* a member of the
// list it guards.
const _: () = assert!(<VanillaProbe<HestonPricer>>::IS_VANILLA_EUROPEAN_CALL);

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
  //! carry `PricingEngine` instead (see the list above); this module is
  //! specifically about `ModelPricer`'s `price_call(s, k, r, q, tau)` shape
  //! not fitting.
  //!
  //! - Multi-asset (`ArithmeticBasketLevyPricer`, `GeometricBasketPricer`,
  //!   `McBasketPricer`, `MargrabePricer`, `KirkSpreadPricer`,
  //!   `McSpreadPricer`, `StulzRainbowPricer`, `McRainbowPricer`):
  //!   `MargrabePricer` is an exchange option with no strike;
  //!   `KirkSpreadPricer` has one but strikes it against a *spread* of two
  //!   forwards; the basket and spread pricers bundle N legs and weights
  //!   into the struct; the rainbow pricers price the best/worst of N
  //!   assets. A shared signature would need an `Option<f64>` strike and a
  //!   variable-length underlying list. `KirkSpreadPricer` is the only one
  //!   of the eight that follows D3's model/query split today
  //!   (`KirkSpreadPricer::call_put(f1, f2, x, r, tau)` against a struct
  //!   holding `v1`/`v2`/`corr`), because Task 6 had to reshape it when it
  //!   took `PricerExt` away; the other seven still bundle their query.
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
