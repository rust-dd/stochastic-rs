//! Every pricer and engine, checked at compile time (each list asserts its trait) and at run time
//! against `src` re-derived by `syn` (`pricer_inventory`), so an unlisted pricer fails; the counts:
//!
//! ```text
//! cd stochastic-rs-quant/tests
//! awk '/^assert_model_pricer!\(/,/^\);/'              pricer_registry.rs | grep -c '^  [A-Z]'
//! awk '/^assert_vanilla_european_call!\(/,/^\);/'     pricer_registry.rs | grep -c '^  [A-Z]'
//! awk '/^assert_not_vanilla_european_call!\(/,/^\);/' pricer_registry.rs | grep -c '^  [A-Z]'
//! awk '/^assert_pricing_engine!\(/,/^\);/'            pricer_registry.rs | grep -c '^  ('
//! awk '/^const NO_TRAIT_BY_DESIGN/,/^\];/'            pricer_registry.rs | grep -c '^  ('
//! ```

use std::marker::PhantomData;

use stochastic_rs_quant::bonds::Cir;
use stochastic_rs_quant::bonds::HullWhite;
use stochastic_rs_quant::bonds::Vasicek;
use stochastic_rs_quant::calibration::levy::LevyModel;
use stochastic_rs_quant::instruments::DigitalOption;
use stochastic_rs_quant::instruments::EuropeanOption;
use stochastic_rs_quant::lattice::equity::CrrModel;
use stochastic_rs_quant::pricing::AnalyticBsEngine;
use stochastic_rs_quant::pricing::AnalyticHestonEngine;
use stochastic_rs_quant::pricing::AssetOrNothingPricer;
use stochastic_rs_quant::pricing::BjerksundStensland2002Pricer;
use stochastic_rs_quant::pricing::BsmPricer;
use stochastic_rs_quant::pricing::CashOrNothingPricer;
use stochastic_rs_quant::pricing::GapPricer;
use stochastic_rs_quant::pricing::HestonPricer;
use stochastic_rs_quant::pricing::HestonSlvPricer;
use stochastic_rs_quant::pricing::HestonStochCorrPricer;
use stochastic_rs_quant::pricing::RBergomiPricer;
use stochastic_rs_quant::pricing::SuperSharePricer;
use stochastic_rs_quant::pricing::asian::AsianPricer;
use stochastic_rs_quant::pricing::bachelier::BachelierPricer;
use stochastic_rs_quant::pricing::finite_difference::FiniteDifferencePricer;
use stochastic_rs_quant::pricing::heston_adi::HestonAdiPricer;
use stochastic_rs_quant::pricing::malliavin_gbm::GbmMalliavinPricer;
use stochastic_rs_quant::pricing::merton_jump::Merton1976Pricer;
use stochastic_rs_quant::pricing::quanto::QuantoPricer;
use stochastic_rs_quant::pricing::sabr::SabrPricer;
use stochastic_rs_quant::pricing::snell_envelope::SnellEnvelopePricer;
use stochastic_rs_quant::traits::ModelPricer;
use stochastic_rs_quant::traits::PricingEngine;
use stochastic_rs_quant::traits::ShortRatePricer;
use stochastic_rs_quant::traits::VanillaEuropeanCall;

mod pricer_inventory;

/// Asserts at compile time that `$t` implements [`ModelPricer`] and publishes the list to the
/// runtime audit, so the two are one list.
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

/// Asserts `$t` implements [`VanillaEuropeanCall`], a European vanilla `price_call` that earns
/// `ModelSurface::vol_surface`, and publishes the list to the audit.
macro_rules! assert_vanilla_european_call {
  ($($t:ty),* $(,)?) => {
    $(const _: fn() = || {
      fn assert_impl<T: VanillaEuropeanCall + ?Sized>() {}
      assert_impl::<$t>();
    };)*
    const REGISTERED_VANILLA_EUROPEAN_CALL: &[&str] = &[$(stringify!($t)),*];
  };
}

/// Probe whose constant is `true` exactly for [`VanillaEuropeanCall`] implementors: the inherent
/// impl applies only under the bound and shadows the trait's `false`, as Rust has no `T: !Trait`.
struct VanillaProbe<T: ?Sized>(PhantomData<T>);

trait MaybeVanillaEuropeanCall {
  const IS_VANILLA_EUROPEAN_CALL: bool = false;
}

impl<T: ?Sized> MaybeVanillaEuropeanCall for VanillaProbe<T> {}

impl<T: VanillaEuropeanCall + ?Sized> VanillaProbe<T> {
  const IS_VANILLA_EUROPEAN_CALL: bool = true;
}

/// Asserts `$t` lacks [`VanillaEuropeanCall`] and publishes the list, which with the positive one
/// must tile `assert_model_pricer!`; the positive control below keeps it from passing vacuously.
macro_rules! assert_not_vanilla_european_call {
  ($($t:ty),* $(,)?) => {
    $(const _: () = assert!(!<VanillaProbe<$t>>::IS_VANILLA_EUROPEAN_CALL);)*
    const REGISTERED_NOT_VANILLA_EUROPEAN_CALL: &[&str] = &[$(stringify!($t)),*];
  };
}

// The `price_call(s, k, r, q, tau)` implementors; most override `price_put` because their carry is
// not `r − q`, their put carries an early-exercise premium, or parity would lose a closed form.
assert_model_pricer!(
  AsianPricer,
  AssetOrNothingPricer,
  BachelierPricer,
  BjerksundStensland2002Pricer,
  BsmPricer,
  CashOrNothingPricer,
  CrrModel<f64>,
  FiniteDifferencePricer,
  GapPricer,
  GbmMalliavinPricer,
  HestonAdiPricer,
  HestonPricer,
  HestonSlvPricer,
  HestonStochCorrPricer,
  LevyModel,
  Merton1976Pricer,
  QuantoPricer,
  RBergomiPricer,
  SabrPricer,
  SnellEnvelopePricer,
  SuperSharePricer,
);

// The pricers whose `price_call` is a European vanilla call, which `ModelSurface` may invert; a
// `FourierModelExt` model has it through the blanket `BLANKET_IMPLS` pins, by Lewis's integral.
assert_vanilla_european_call!(
  BachelierPricer,
  BsmPricer,
  CrrModel<f64>,
  FiniteDifferencePricer,
  GbmMalliavinPricer,
  HestonAdiPricer,
  HestonPricer,
  HestonSlvPricer,
  HestonStochCorrPricer,
  LevyModel,
  Merton1976Pricer,
  RBergomiPricer,
  SabrPricer,
);

// `ModelPricer`s a Black inversion cannot describe, though each inverts to a finite vol: the
// digitals, the geometric Asian, the two American pricers and the quanto.
assert_not_vanilla_european_call!(
  AsianPricer,
  AssetOrNothingPricer,
  BjerksundStensland2002Pricer,
  CashOrNothingPricer,
  GapPricer,
  QuantoPricer,
  SnellEnvelopePricer,
  SuperSharePricer,
);

// The probe's positive control (see `assert_not_vanilla_european_call!`), outside the list it
// guards.
const _: () = assert!(<VanillaProbe<HestonPricer>>::IS_VANILLA_EUROPEAN_CALL);

// The `Instrument` + `PricingEngine` engines, one entry per `(engine, instrument)` impl, as the
// audit diffs pairs.
assert_pricing_engine!(
  (AnalyticBsEngine, DigitalOption),
  (AnalyticBsEngine, EuropeanOption),
  (AnalyticHestonEngine, EuropeanOption),
);

// The short-rate bond family, named for their model, so only the trait signal sees them.
assert_short_rate_pricer!(Cir, HullWhite, Vasicek);

/// Blanket impls of a tracked trait as `(trait, selecting bound)`, pinned because a blanket hands a
/// trait to types no list enumerates.
const BLANKET_IMPLS: &[(&str, &str)] = &[
  ("ModelPricer", "FourierModelExt"),
  ("VanillaEuropeanCall", "FourierModelExt"),
];

const MULTI_ASSET: &str = "multi-asset: no single (s, k) query point";
const PATH_DEPENDENT: &str = "path-dependent: contract parameters live in the struct";
const EXOTIC: &str = "exotic: the contract and the query it reads are arguments";
const FOURIER_ENGINE: &str = "Fourier engine: takes the model per call rather than being one";
const NON_OPTION: &str = "not a single-underlying option pricer";

// Pricers whose call shape cannot be `price_call(s, k, r, q, tau)`: multi-asset, path-dependent and
// exotic contracts, Fourier engines that take the model per call, and non-option pricers.
const NO_TRAIT_BY_DESIGN: &[(&str, &str)] = &[
  ("ArithmeticBasketLevyPricer", MULTI_ASSET),
  ("AutocallablePricer", PATH_DEPENDENT),
  ("BarrierPricer", EXOTIC),
  ("BermudanLsmPricer", PATH_DEPENDENT),
  ("CarrMadanPricer", FOURIER_ENGINE),
  ("CashflowPricer", NON_OPTION),
  ("CgmysvPricer", FOURIER_ENGINE),
  ("ChooserPricer", EXOTIC),
  ("CliquetPricer", EXOTIC),
  ("CompoundPricer", EXOTIC),
  ("CosEngine", FOURIER_ENGINE),
  ("CosPricer", FOURIER_ENGINE),
  ("DoubleBarrierPricer", EXOTIC),
  ("FrftCarrMadanPricer", FOURIER_ENGINE),
  ("GeometricBasketPricer", MULTI_ASSET),
  ("GilPelaezPricer", FOURIER_ENGINE),
  ("KirkSpreadPricer", MULTI_ASSET),
  ("LewisPricer", FOURIER_ENGINE),
  ("LookbackPricer", EXOTIC),
  ("McBarrierPricer", PATH_DEPENDENT),
  ("MargrabePricer", MULTI_ASSET),
  ("McBasketPricer", MULTI_ASSET),
  ("McCliquetPricer", PATH_DEPENDENT),
  ("McRainbowPricer", MULTI_ASSET),
  ("McSpreadPricer", MULTI_ASSET),
  ("PortfolioEngine", NON_OPTION),
  ("StulzRainbowPricer", MULTI_ASSET),
  ("VarianceSwapPricer", EXOTIC),
  ("VolatilitySwapPricer", EXOTIC),
];

/// Diffs the lists against the inventory re-derived from `src/**/*.rs`, failing on a pricer the
/// file has never heard of.
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
