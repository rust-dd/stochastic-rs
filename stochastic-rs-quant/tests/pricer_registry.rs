//! Inventory of every pricer and engine in this crate, checked two ways.
//!
//! **At compile time**, each list below is a macro invocation that asserts,
//! entry by entry, that the named type implements the trait the list is named
//! for. A pricer that *loses* its trait stops compiling here.
//!
//! **At run time**, `registry_matches_crate_source` re-derives the same
//! inventory from `src/**/*.rs` with `syn` (see `tests/pricer_inventory/`) and
//! diffs it against these lists. A pricer *added* without a registry entry
//! fails there.
//!
//! The second half is not belt and braces. A hand-curated list can be wrong
//! about a name it contains; it cannot be wrong about a name it has never
//! heard of. During this file's own review a real `ReviewProbeOrphanPricer`
//! was added to `pricing/asian.rs`, the registry was left untouched, and every
//! assertion here passed — unaware the struct existed.
//!
//! ## What counts as a pricer
//!
//! Two signals, unioned, because either alone has a hole this file has already
//! fallen into:
//!
//!  - **carries a pricing trait** — one of [`ModelPricer`], [`PricingEngine`],
//!    [`ShortRatePricer`], [`VanillaEuropeanCall`];
//!  - **is named like one** — a `pub struct` whose identifier ends in `Pricer`
//!    or `Engine`.
//!
//! Name shape alone is blind to [`LevyModel`], [`CrrModel`], [`Cir`],
//! [`HullWhite`] and [`Vasicek`]: all five implement a pricing trait under a
//! model's name. An earlier revision of this header derived its inventory from
//! `pub struct *(Pricer|Engine)` and so counted none of them, while its opening
//! line claimed to cover every pricer/engine struct in the crate. Trait
//! membership alone is blind to everything in `NO_TRAIT_BY_DESIGN` — which is
//! exactly where an orphan hides, an orphan being a struct that implements
//! nothing.
//!
//! Every name in the union has to be either in a trait list or in
//! `NO_TRAIT_BY_DESIGN` with a reason. The audit fails on one that is in
//! neither, and equally on a `NO_TRAIT_BY_DESIGN` entry the source has since
//! dropped or has since given a trait to.
//!
//! ## Counts
//!
//! This prose states none. Every figure the file used to carry is the length of
//! a list the audit now checks, so it is re-derived rather than remembered:
//!
//! ```text
//! cd stochastic-rs-quant/tests
//! awk '/^assert_model_pricer!\(/,/^\);/'              pricer_registry.rs | grep -c '^  [A-Z]'
//! awk '/^assert_vanilla_european_call!\(/,/^\);/'     pricer_registry.rs | grep -c '^  [A-Z]'
//! awk '/^assert_not_vanilla_european_call!\(/,/^\);/' pricer_registry.rs | grep -c '^  [A-Z]'
//! awk '/^assert_pricing_engine!\(/,/^\);/'            pricer_registry.rs | grep -c '^  ('
//! awk '/^const NO_TRAIT_BY_DESIGN/,/^\];/'            pricer_registry.rs | grep -c '^  ('
//! ```
//!
//! Five commands, not one with the name substituted: `assert_pricing_engine!`
//! takes `(pricer, instrument)` pairs and `NO_TRAIT_BY_DESIGN` holds
//! `(struct, reason)` pairs, so both open with a paren and `grep -c '^  [A-Z]'`
//! reports 0 for them. An earlier revision of this header offered one command
//! with the macro name substituted; the command it printed returns 0 for
//! `assert_pricing_engine!`.
//!
//! ## What this does not catch
//!
//! A struct that neither carries a pricing trait nor is named `*Pricer` /
//! `*Engine` — a `FooModel` with an inherent `price()` and nothing else — is in
//! neither signal and passes unseen. So does a pricer-shaped struct declared
//! `pub(crate)` rather than `pub`, and anything under `src/python/`, which the
//! scan skips by path. The first is the real residual: closing it needs a
//! signal the compiler exposes, and Rust cannot enumerate a trait's
//! implementors at run time.
//!
//! Per the A2 design (`docs/superpowers/specs/2026-08-23-a2-quant-consistency-design.md`,
//! decision D1), `PricerExt` is retired and this file no longer has a list for
//! it. Each surviving trait gets its own compile-checked list, so a claim about
//! *which* trait a struct carries can drift no more than the claim that it
//! carries one at all.

use std::marker::PhantomData;

use stochastic_rs_quant::bonds::Cir;
use stochastic_rs_quant::bonds::HullWhite;
use stochastic_rs_quant::bonds::Vasicek;
use stochastic_rs_quant::calibration::levy::LevyModel;
use stochastic_rs_quant::instruments::DigitalOption;
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

mod pricer_inventory;

/// Asserts `$t` implements [`ModelPricer`] at compile time, and publishes the
/// list to the runtime audit so the compile-time list and the audited list are
/// one list rather than two that can drift.
macro_rules! assert_model_pricer {
  ($($t:ty),* $(,)?) => {
    $(const _: fn() = || {
      fn assert_impl<T: ModelPricer + ?Sized>() {}
      assert_impl::<$t>();
    };)*
    const REGISTERED_MODEL_PRICER: &[&str] = &[$(stringify!($t)),*];
  };
}

/// Asserts `$t` implements [`PricingEngine<I>`] at compile time for the paired
/// instrument type `$i`, and publishes the pairs to the audit.
macro_rules! assert_pricing_engine {
  ($(($t:ty, $i:ty)),* $(,)?) => {
    $(const _: fn() = || {
      fn assert_impl<T: PricingEngine<I>, I>() {}
      assert_impl::<$t, $i>();
    };)*
    const REGISTERED_PRICING_ENGINE: &[(&str, &str)] =
      &[$((stringify!($t), stringify!($i))),*];
  };
}

/// Asserts `$t` implements the short-rate bond family's [`ShortRatePricer`] at
/// compile time, and publishes the list to the audit.
macro_rules! assert_short_rate_pricer {
  ($($t:ty),* $(,)?) => {
    $(const _: fn() = || {
      fn assert_impl<T: ShortRatePricer + ?Sized>() {}
      assert_impl::<$t>();
    };)*
    const REGISTERED_SHORT_RATE_PRICER: &[&str] = &[$(stringify!($t)),*];
  };
}

/// Asserts `$t` implements [`VanillaEuropeanCall`] at compile time — that its
/// `price_call` is a European vanilla call, which is what earns it a
/// `ModelSurface::vol_surface` — and publishes the list to the audit.
macro_rules! assert_vanilla_european_call {
  ($($t:ty),* $(,)?) => {
    $(const _: fn() = || {
      fn assert_impl<T: VanillaEuropeanCall + ?Sized>() {}
      assert_impl::<$t>();
    };)*
    const REGISTERED_VANILLA_EUROPEAN_CALL: &[&str] = &[$(stringify!($t)),*];
  };
}

/// Probe carrying a `bool` that reports whether `T` implements
/// [`VanillaEuropeanCall`]: the inherent `impl` below applies only when the
/// bound holds and shadows the trait's `false` when it does, so the constant is
/// `true` exactly for implementors.
///
/// Rust has no `where T: !Trait`, and a *missing* impl is precisely what needs
/// pinning here — an added blanket impl, or a marker pasted onto a digital, is
/// the regression this file exists to catch, and both are invisible to a
/// positive-only inventory.
struct VanillaProbe<T: ?Sized>(PhantomData<T>);

trait MaybeVanillaEuropeanCall {
  const IS_VANILLA_EUROPEAN_CALL: bool = false;
}

impl<T: ?Sized> MaybeVanillaEuropeanCall for VanillaProbe<T> {}

impl<T: VanillaEuropeanCall + ?Sized> VanillaProbe<T> {
  const IS_VANILLA_EUROPEAN_CALL: bool = true;
}

/// Asserts `$t` does **not** implement [`VanillaEuropeanCall`], so
/// `$t::vol_surface(..)` does not compile, and publishes the list to the audit,
/// which checks that this list and `assert_vanilla_european_call!` tile
/// `assert_model_pricer!` exactly.
///
/// The positive control below the invocation is not decoration: if the
/// shadowing ever stopped resolving, every negative assertion here would pass
/// vacuously and this whole list would become a comment. The control fails to
/// compile in that case.
macro_rules! assert_not_vanilla_european_call {
  ($($t:ty),* $(,)?) => {
    $(const _: () = assert!(!<VanillaProbe<$t>>::IS_VANILLA_EUROPEAN_CALL);)*
    const REGISTERED_NOT_VANILLA_EUROPEAN_CALL: &[&str] = &[$(stringify!($t)),*];
  };
}

// Structs on the decoupled `price_call(s, k, r, q, tau)` surface used by
// calibration and vol-surface construction: the 4 digital options (Task 3), the
// 2 original members, `BSMPricer` (Task 5a), Task 5b's 9 arrivals, and
// `LevyModel` and `CrrModel` — real implementors that the retired
// `pub struct *(Pricer|Engine)` derivation could not see.
//
// Every 5b arrival overrides `price_put` rather than taking the trait's vanilla
// put-call-parity default, for one of three reasons:
//
//  - cost of carry: `BSMPricer`, `AsianPricer`, `Merton1976Pricer` carry at
//    `exp((b - r) * tau)`, which equals the default's `exp(-q * tau)` only when
//    `b = r - q` — false for `BSMCoc::Bsm1973` at `q != 0` and for `Black1976` /
//    `Asay1982`, and true only on a measure-zero line for the Asian pricer's
//    averaged-underlying carry;
//  - American exercise: `BjerksundStensland2002Pricer`, `SnellEnvelopePricer`
//    and `FiniteDifferencePricer` price a put that carries an early-exercise
//    premium the call does not, so European parity is not an approximation but
//    the wrong model;
//  - exactness: `SabrPricer`, `HestonPricer`, `HestonStochCorrPricer` and
//    `GbmMalliavinPricer` all have carry `b = r - q`, where the default is
//    mathematically right — but it recomposes the put from the call and so can
//    land an ulp away from the closed form, drop a `max(0)` floor
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

// The subset of the list above whose `price_call` is a European vanilla call,
// and so may be inverted through the Black formula into an implied vol surface.
// `ModelSurface` is blanket-implemented over this trait, not over
// `ModelPricer`, so membership here is what makes `vol_surface` callable at
// all.
//
// Absent from the list because a blanket impl in `pricing/fourier/mod.rs`
// covers them: every `FourierModelExt` model. Gil-Pelaez prices
// `S e^{-q tau} P_1 - K e^{-r tau} P_2` — the vanilla European call — for all
// of them, so they are covered by the same one-line reasoning rather than one
// entry each. `BLANKET_IMPLS` below pins that blanket, so a *second* one
// arriving is a test failure rather than a silent widening.
//
// `FiniteDifferencePricer` is the one member whose answer is per-instance: it
// holds its exercise style in a field, so it carries the trait and reports
// `NaN` from `vanilla_call_forward` at `OptionStyle::American`. `BSMPricer` and
// `Merton1976Pricer` are the two whose *forward* is per-instance, at `b(r, q)`
// rather than the default `r - q`.
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
// describe, listed so the omission is a decision rather than an oversight. Each
// one used to get `vol_surface` from the `ModelPricer` blanket and returned a
// finite, plausible, meaningless surface for it.
//
//  - the four digitals (`AssetOrNothingPricer`, `CashOrNothingPricer`,
//    `GapPricer`, `SuperSharePricer`): the payoff is not `(S_T - K)^+`, so no
//    Black volatility reproduces it. Nothing about the output says so — an
//    asset-or-nothing call is worth `F N(d_1)`, inside the no-arbitrage band
//    `(max(F - K, 0), F)` at every strike, so every point inverts.
//  - `AsianPricer`: a vanilla call, but on the geometric *average* rather than
//    on `S_T`, and carried at `b = (r - q - sigma_A^2 / 6) / 2`. The inversion
//    would return an averaged-underlying volatility labelled as the spot's,
//    near `sigma / sqrt(3)` and so especially easy to mistake for a real one.
//  - `BjerksundStensland2002Pricer`, `SnellEnvelopePricer`: American exercise,
//    and unconditionally so — unlike `FiniteDifferencePricer` there is no
//    European setting to fall back to. An American call at `q = 0` *is* the
//    European one, which is what makes the general case dangerous rather than
//    obviously wrong: at `q = 0.06` both invert to within 0.008 of the model's
//    own volatility, close enough to pass any eyeball check and wrong at every
//    point.
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
// `assert_not_vanilla_european_call!`. Deliberately *not* a member of the list
// it guards.
const _: () = assert!(<VanillaProbe<HestonPricer>>::IS_VANILLA_EUROPEAN_CALL);

// QuantLib-style decoupled engines (`Instrument` + `PricingEngine<I>` +
// `PricingResult`, see `traits::instrument`) rather than `ModelPricer`. This is
// a considered second pricing architecture, not a gap: `Instrument` describes
// the payoff, the engine owns model/market/method and reacts to market-data
// updates, and one engine can serve several instrument types.
//
// One entry per `(engine, instrument)` impl, not per engine: the audit diffs
// pairs, so `AnalyticBSEngine`'s second instrument is checked rather than
// mentioned in a comment, which is how it was handled before.
assert_pricing_engine!(
  (AnalyticBSEngine, DigitalOption),
  (AnalyticBSEngine, EuropeanOption),
  (AnalyticHestonEngine, EuropeanOption),
);

// The short-rate bond family (Task 2). `Cir`, `HullWhite`, `Vasicek` are named
// for their model, not `*Pricer`/`*Engine`, so only the trait signal sees them.
assert_short_rate_pricer!(Cir, HullWhite, Vasicek);

/// Blanket impls of a tracked trait, as `(trait, selecting bound)`. Pinned
/// because a blanket hands a pricing trait to types no list can enumerate: the
/// audit fails if one is added, removed, or re-bound.
const BLANKET_IMPLS: &[(&str, &str)] = &[
  ("ModelPricer", "FourierModelExt"),
  ("VanillaEuropeanCall", "FourierModelExt"),
];

const MULTI_ASSET: &str = "multi-asset: no single (s, k) query point";
const PATH_DEPENDENT: &str = "path-dependent: contract parameters live in the struct";
const FOURIER_ENGINE: &str = "Fourier engine: parameterised by a model, not a model";
const NON_OPTION: &str = "not a single-underlying option pricer";

// Families that deliberately carry no `ModelPricer` implementation, with the
// reason. Listing them is what makes the omission deliberate rather than an
// oversight — see the A2 design's D1. Some carry `PricingEngine` instead (see
// the list above); this list is specifically about `ModelPricer`'s
// `price_call(s, k, r, q, tau)` shape not fitting.
//
//  - `MULTI_ASSET`: `MargrabePricer` is an exchange option with no strike;
//    `KirkSpreadPricer` has one but strikes it against a *spread* of two
//    forwards; the basket and spread pricers bundle N legs and weights into the
//    struct; the rainbow pricers price the best/worst of N assets. A shared
//    signature would need an `Option<f64>` strike and a variable-length
//    underlying list. `KirkSpreadPricer` is the only one of the eight that
//    follows D3's model/query split today
//    (`KirkSpreadPricer::call_put(f1, f2, x, r, tau)` against a struct holding
//    `v1`/`v2`/`corr`), because Task 6 had to reshape it when it took
//    `PricerExt` away; the other seven still bundle their query.
//  - `PATH_DEPENDENT`: each bundles its own contract parameters (barrier level,
//    lookback window, cliquet reset schedule, autocall trigger/coupon schedule,
//    Bermudan exercise dates, chooser decision date, ...) into the struct and
//    prices off that, not off a single `(s, k, r, q, tau)` query point.
//  - `FOURIER_ENGINE`: `CarrMadanPricer`, `GilPelaezPricer`, `LewisPricer`,
//    `FrftCarrMadanPricer` and `CosEngine` take `model: &impl FourierModelExt`
//    per call; `CosPricer` builds a `RegimeSwitchingModel` internally from raw
//    inputs and prices through it. The blanket
//    `impl<T: FourierModelExt> ModelPricer for T` already gives the models
//    themselves (`HestonFourier`, `CgmysvModel`, `RegimeSwitchingModel`, ...)
//    their `ModelPricer`. `CgmysvPricer` is the outlier in this group: it takes
//    no model parameter and touches no characteristic function at all — it
//    duplicates `CgmysvParams` as struct fields and is a self-contained Monte
//    Carlo pricer returning `McResult` (price ± standard error), so
//    `ModelPricer`'s plain-`f64` return does not fit it either. It is listed
//    here as `CgmysvModel`'s Monte Carlo counterpart, not because it shares the
//    other six's call shape.
//  - `NON_OPTION`: `CashflowPricer::leg_npv`/`cashflow_npv` discount a
//    bond/loan cashflow schedule against a curve stack;
//    `PortfolioEngine::optimize`/`score_momentum`/`build_momentum` allocate
//    weights across N assets. Neither has a single option struck at `K` on `S`
//    to put through `price_call(s, k, r, q, tau)`.
const NO_TRAIT_BY_DESIGN: &[(&str, &str)] = &[
  ("ArithmeticBasketLevyPricer", MULTI_ASSET),
  ("AutocallablePricer", PATH_DEPENDENT),
  ("BarrierPricer", PATH_DEPENDENT),
  ("BermudanLsmPricer", PATH_DEPENDENT),
  ("CarrMadanPricer", FOURIER_ENGINE),
  ("CashflowPricer", NON_OPTION),
  ("CgmysvPricer", FOURIER_ENGINE),
  ("CliquetPricer", PATH_DEPENDENT),
  ("ComplexChooserPricer", PATH_DEPENDENT),
  ("CompoundPricer", PATH_DEPENDENT),
  ("CosEngine", FOURIER_ENGINE),
  ("CosPricer", FOURIER_ENGINE),
  ("DoubleBarrierPricer", PATH_DEPENDENT),
  ("FixedLookbackPricer", PATH_DEPENDENT),
  ("FloatingLookbackPricer", PATH_DEPENDENT),
  ("ForwardStartPricer", PATH_DEPENDENT),
  ("FrftCarrMadanPricer", FOURIER_ENGINE),
  ("GeometricBasketPricer", MULTI_ASSET),
  ("GilPelaezPricer", FOURIER_ENGINE),
  ("KirkSpreadPricer", MULTI_ASSET),
  ("LewisPricer", FOURIER_ENGINE),
  ("MCBarrierPricer", PATH_DEPENDENT),
  ("MargrabePricer", MULTI_ASSET),
  ("McBasketPricer", MULTI_ASSET),
  ("McCliquetPricer", PATH_DEPENDENT),
  ("McRainbowPricer", MULTI_ASSET),
  ("McSpreadPricer", MULTI_ASSET),
  ("PortfolioEngine", NON_OPTION),
  ("SimpleChooserPricer", PATH_DEPENDENT),
  ("StulzRainbowPricer", MULTI_ASSET),
  ("VarianceSwapPricer", PATH_DEPENDENT),
  ("VolatilitySwapPricer", PATH_DEPENDENT),
];

/// Re-derives the inventory from `src/**/*.rs` and diffs it against the lists
/// above. This is the half a hand-curated file cannot do for itself: it fails
/// on a pricer the file has never heard of.
#[test]
fn registry_matches_crate_source() {
  let registered = pricer_inventory::Registered {
    model_pricer: REGISTERED_MODEL_PRICER,
    vanilla_european_call: REGISTERED_VANILLA_EUROPEAN_CALL,
    not_vanilla_european_call: REGISTERED_NOT_VANILLA_EUROPEAN_CALL,
    short_rate_pricer: REGISTERED_SHORT_RATE_PRICER,
    pricing_engine: REGISTERED_PRICING_ENGINE,
    no_trait_by_design: NO_TRAIT_BY_DESIGN,
    blanket_impls: BLANKET_IMPLS,
  };
  let inventory = pricer_inventory::scan_crate_source();
  let problems = pricer_inventory::audit(&inventory, &registered);
  assert!(
    problems.is_empty(),
    "the source and this registry disagree. Register the type in the list for \
     the trait it carries, or add it to `NO_TRAIT_BY_DESIGN` with the reason it \
     carries none.\n\n{}",
    problems.join("\n")
  );
}
