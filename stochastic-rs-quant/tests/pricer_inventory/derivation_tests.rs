//! Pins what the scan in the parent module extracts.
//!
//! The guard in `pricer_registry.rs` is only as good as this derivation, so
//! the derivation gets its own fixture rather than being trusted because it
//! agrees with the registry today. Every shape below is one a line-based
//! inventory gets wrong: a decoy inside a string literal or a doc comment, a
//! `*PricerBuilder` helper, a `*EngineConfig`, a `#[cfg(test)]` module, a
//! blanket impl, and an impl nested inside a function body.

use std::collections::BTreeSet;

use super::Registered;
use super::audit;
use super::base_name;
use super::diff;
use super::parse_source;
use super::scan_crate_source;

/// One file exercising every shape the scan has to separate.
const FIXTURE: &str = r##"
pub struct AlphaPricer;

pub struct BetaEngine {
  pub field: f64,
}

pub struct GammaEngineConfig;
pub struct DeltaPricerBuilder;
pub(crate) struct EpsilonPricer;
struct ZetaPricer;

impl ModelPricer for AlphaPricer {}
impl crate::traits::ModelPricer for QualifiedPricer {}
impl ModelPricer for GenericArgPricer<f64> {}
impl<T: Clone> ModelPricer for WrapperPricer<T> {}
impl<T: FourierModelExt> ModelPricer for T {}
impl<T> VanillaEuropeanCall for T where T: FourierModelExt {}
impl<T: ?Sized + SomeMarker> ShortRatePricer for T {}
impl PricingEngine<EuropeanOption> for BetaEngine {}
impl PricingEngine<DigitalOption> for BetaEngine {}
impl SomeUntrackedTrait for AlphaPricer {}
impl !ModelPricer for NegatedPricer {}
impl ModelPricer for (f64, f64) {}

/// Doc comment naming `pub struct DocCommentPricer` and
/// `impl ModelPricer for DocCommentThing`, which must not be seen.
pub fn decoy() -> &'static str {
  "pub struct StringLiteralPricer; impl ModelPricer for StringLiteralThing {}"
}

pub fn function_scope() {
  pub struct FnBodyPricer;
  impl ModelPricer for FnBodyPricer {}
}

pub mod inner {
  pub struct InnerPricer;
  impl ModelPricer for InnerPricer {}
}

#[cfg(test)]
mod tests {
  pub struct MockPricer;
  impl ModelPricer for MockPricer {}
}

#[cfg(test)]
pub struct DirectlyGatedPricer;

#[cfg(test)]
impl ModelPricer for DirectlyGatedPricer {}
"##;

fn set(names: &[&str]) -> BTreeSet<String> {
  names.iter().map(|name| (*name).to_string()).collect()
}

#[test]
fn scan_names_only_public_pricer_suffixed_structs() {
  let inventory = parse_source(FIXTURE);
  assert_eq!(
    inventory.named_structs,
    set(&["AlphaPricer", "BetaEngine", "InnerPricer"]),
    "`GammaEngineConfig` and `DeltaPricerBuilder` are suffix misses, \
     `EpsilonPricer`/`ZetaPricer` are not `pub`, `MockPricer` and \
     `DirectlyGatedPricer` are `#[cfg(test)]`, `FnBodyPricer` is function-\
     local, and the rest are decoys inside a doc comment or a string literal"
  );
}

#[test]
fn scan_separates_concrete_impls_from_blanket_ones() {
  let inventory = parse_source(FIXTURE);
  assert_eq!(
    inventory.implementors("ModelPricer"),
    set(&[
      "AlphaPricer",
      "GenericArgPricer",
      "InnerPricer",
      "QualifiedPricer",
      "WrapperPricer",
    ]),
    "a path-qualified trait, a generic argument on the self type and a \
     generic parameter on the impl all still name a concrete implementor; a \
     negated impl, a `#[cfg(test)]` impl and a function-local impl do not"
  );
  assert_eq!(inventory.implementors("VanillaEuropeanCall"), set(&[]));
  assert_eq!(inventory.implementors("ShortRatePricer"), set(&[]));
  assert_eq!(
    inventory.implementors("PricingEngine"),
    set(&["BetaEngine"])
  );
}

#[test]
fn scan_records_the_bound_that_drives_each_blanket_impl() {
  let inventory = parse_source(FIXTURE);
  assert_eq!(
    inventory.blanket.get("ModelPricer"),
    Some(&set(&["FourierModelExt"])),
    "an inline bound names the blanket"
  );
  assert_eq!(
    inventory.blanket.get("VanillaEuropeanCall"),
    Some(&set(&["FourierModelExt"])),
    "a `where` clause names it just as well"
  );
  assert_eq!(
    inventory.blanket.get("ShortRatePricer"),
    Some(&set(&["SomeMarker"])),
    "`?Sized` is a bound but not the one that selects implementors"
  );
  assert_eq!(inventory.blanket.get("PricingEngine"), None);
}

#[test]
fn scan_pairs_each_engine_with_every_instrument_it_prices() {
  let inventory = parse_source(FIXTURE);
  let pairs = inventory
    .engine_pairs
    .iter()
    .map(|(engine, instrument)| format!("{engine}/{instrument}"))
    .collect::<BTreeSet<_>>();
  assert_eq!(
    pairs,
    set(&["BetaEngine/DigitalOption", "BetaEngine/EuropeanOption"]),
    "one engine serving two instruments is two entries, not one"
  );
}

#[test]
fn scan_reports_rather_than_drops_an_unforeseen_self_type() {
  let inventory = parse_source(FIXTURE);
  assert_eq!(
    inventory.unclassified,
    set(&["impl ModelPricer for <tuple>"]),
    "a self type that is neither a named path nor a generic parameter has to \
     surface somewhere, or the inventory shrinks in silence"
  );
}

#[test]
fn base_name_strips_path_and_generic_arguments() {
  assert_eq!(base_name("HestonPricer"), "HestonPricer");
  assert_eq!(base_name("CrrModel<f64>"), "CrrModel");
  assert_eq!(base_name("CrrModel < f64 >"), "CrrModel");
  assert_eq!(
    base_name("crate::lattice::equity::CrrModel<f64>"),
    "CrrModel"
  );
  assert_eq!(base_name(" Spaced "), "Spaced");
}

#[test]
fn diff_reports_both_directions_and_stays_quiet_on_agreement() {
  let derived = set(&["A", "B"]);
  assert_eq!(diff("l", &derived, &set(&["A", "B"])), None);

  let report = diff("l", &derived, &set(&["B", "C"])).expect("disagreement");
  assert!(
    report.contains("missing from the registry: [\"A\"]"),
    "{report}"
  );
  assert!(
    report.contains("absent from the source: [\"C\"]"),
    "{report}"
  );
}

#[test]
fn crate_scan_skips_the_python_wrappers_and_finds_the_real_tree() {
  let inventory = scan_crate_source();
  assert!(
    inventory.named_structs.len() > 40,
    "the crate scan collapsed to {} named structs",
    inventory.named_structs.len()
  );
  let wrappers = inventory
    .named_structs
    .iter()
    .filter(|name| name.starts_with("Py"))
    .collect::<Vec<_>>();
  assert!(
    wrappers.is_empty(),
    "`src/python/` is excluded by path, yet these arrived: {wrappers:?}"
  );
}

/// A crate small enough to hand-check, holding one properly registered pricer,
/// one that carries a trait under a model's name, and one orphan that carries
/// nothing at all — the three shapes the audit has to react to differently.
const AUDIT_FIXTURE: &str = r#"
pub struct RegisteredPricer;
impl ModelPricer for RegisteredPricer {}
impl VanillaEuropeanCall for RegisteredPricer {}

pub struct ReviewProbeOrphanPricer;

pub struct LevyModel;
impl ModelPricer for LevyModel {}
impl VanillaEuropeanCall for LevyModel {}
"#;

fn audit_fixture(
  model_pricer: &'static [&'static str],
  vanilla_european_call: &'static [&'static str],
  no_trait_by_design: &'static [(&'static str, &'static str)],
) -> Vec<String> {
  audit(
    &parse_source(AUDIT_FIXTURE),
    &Registered {
      model_pricer,
      vanilla_european_call,
      not_vanilla_european_call: &[],
      short_rate_pricer: &[],
      pricing_engine: &[],
      no_trait_by_design,
      blanket_impls: &[],
    },
  )
}

#[test]
fn audit_is_quiet_when_every_pricer_is_accounted_for() {
  let problems = audit_fixture(
    &["LevyModel", "RegisteredPricer"],
    &["LevyModel", "RegisteredPricer"],
    &[("ReviewProbeOrphanPricer", "the orphan, excused")],
  );
  assert!(problems.is_empty(), "{problems:#?}");
}

#[test]
fn audit_catches_an_orphan_the_registry_never_heard_of() {
  let problems = audit_fixture(
    &["LevyModel", "RegisteredPricer"],
    &["LevyModel", "RegisteredPricer"],
    &[],
  );
  assert!(
    problems
      .iter()
      .any(|problem| problem.contains("ReviewProbeOrphanPricer")),
    "a pricer-shaped struct carrying no trait and named nowhere has to fail \
     the audit — that is the blind spot this file exists to close: {problems:#?}"
  );
}

#[test]
fn audit_catches_a_trait_implementor_hiding_behind_a_model_name() {
  let problems = audit_fixture(
    &["RegisteredPricer"],
    &["RegisteredPricer"],
    &[("ReviewProbeOrphanPricer", "the orphan, excused")],
  );
  assert!(
    problems.iter().any(|problem| problem.contains("LevyModel")),
    "`LevyModel` is not named `*Pricer`, so only the trait signal sees it: \
     {problems:#?}"
  );
}

#[test]
fn audit_catches_a_stale_excuse_and_a_contradicted_one() {
  let problems = audit_fixture(
    &["LevyModel", "RegisteredPricer"],
    &["LevyModel", "RegisteredPricer"],
    &[
      ("ReviewProbeOrphanPricer", "the orphan, excused"),
      ("DeletedPricer", "excused after it was deleted"),
      ("RegisteredPricer", "excused although it carries the trait"),
    ],
  );
  let joined = problems.join("\n");
  assert!(joined.contains("DeletedPricer"), "{joined}");
  assert!(
    joined.contains("yet the source gives them one: [\"RegisteredPricer\"]"),
    "{joined}"
  );
}
