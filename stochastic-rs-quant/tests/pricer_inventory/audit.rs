//! Diffs the hand-written registry against the inventory derived from source.
//!
//! Kept beside the derivation rather than in `pricer_registry.rs` so the
//! comparison and the thing it compares against can be exercised together on
//! synthetic inventories, without waiting on a scan of the real tree.

use std::collections::BTreeSet;

use super::Inventory;
use super::base_name;
use super::diff;

/// The lists `pricer_registry.rs` maintains by hand, as the audit sees them.
pub struct Registered<'a> {
  /// Entries of `assert_model_pricer!`.
  pub model_pricer: &'a [&'a str],
  /// Entries of `assert_vanilla_european_call!`.
  pub vanilla_european_call: &'a [&'a str],
  /// Entries of `assert_not_vanilla_european_call!`.
  pub not_vanilla_european_call: &'a [&'a str],
  /// Entries of `assert_short_rate_pricer!`.
  pub short_rate_pricer: &'a [&'a str],
  /// `(engine, instrument)` entries of `assert_pricing_engine!`.
  pub pricing_engine: &'a [(&'a str, &'a str)],
  /// `(struct, reason)` entries of the deliberate-omission list.
  pub no_trait_by_design: &'a [(&'a str, &'a str)],
  /// `(trait, bound)` for each blanket impl the registry knows of. A blanket
  /// impl hands a pricing trait to types no list can name, so it is pinned
  /// here instead: a new one is a change in what the registry even covers.
  pub blanket_impls: &'a [(&'a str, &'a str)],
}

/// Every way the registry and the source disagree, empty when they do not.
pub fn audit(inventory: &Inventory, registered: &Registered<'_>) -> Vec<String> {
  let mut problems = Vec::new();

  for (trait_name, list) in [
    ("ModelPricer", registered.model_pricer),
    ("VanillaEuropeanCall", registered.vanilla_european_call),
    ("ShortRatePricer", registered.short_rate_pricer),
  ] {
    let label = format!("`{trait_name}` implementors");
    if let Some(report) = diff(&label, &inventory.implementors(trait_name), &names(list)) {
      problems.push(report);
    }
  }

  let derived_pairs = inventory
    .engine_pairs
    .iter()
    .map(|(engine, instrument)| format!("{engine} / {instrument}"))
    .collect::<BTreeSet<_>>();
  let registered_pairs = registered
    .pricing_engine
    .iter()
    .map(|(engine, instrument)| format!("{} / {}", base_name(engine), base_name(instrument)))
    .collect::<BTreeSet<_>>();
  if let Some(report) = diff(
    "`PricingEngine<I>` engine/instrument pairs",
    &derived_pairs,
    &registered_pairs,
  ) {
    problems.push(report);
  }

  let derived_blanket = inventory
    .blanket
    .iter()
    .flat_map(|(trait_name, bounds)| {
      bounds
        .iter()
        .map(move |bound| format!("{trait_name} via {bound}"))
    })
    .collect::<BTreeSet<_>>();
  let registered_blanket = registered
    .blanket_impls
    .iter()
    .map(|(trait_name, bound)| format!("{trait_name} via {bound}"))
    .collect::<BTreeSet<_>>();
  if let Some(report) = diff("blanket impls", &derived_blanket, &registered_blanket) {
    problems.push(report);
  }

  if !inventory.unclassified.is_empty() {
    problems.push(format!(
      "tracked-trait impls the scan could not classify: {:?}",
      inventory.unclassified
    ));
  }

  problems.extend(partition_problems(registered));
  problems.extend(coverage_problems(inventory, registered));
  problems
}

/// `assert_vanilla_european_call!` and its complement have to tile
/// `assert_model_pricer!` exactly: every `ModelPricer` is either invertible
/// through the Black formula or explicitly is not, and none is both.
fn partition_problems(registered: &Registered<'_>) -> Vec<String> {
  let mut problems = Vec::new();
  let model = names(registered.model_pricer);
  let vanilla = names(registered.vanilla_european_call);
  let not_vanilla = names(registered.not_vanilla_european_call);

  let overlap = vanilla
    .intersection(&not_vanilla)
    .cloned()
    .collect::<Vec<_>>();
  if !overlap.is_empty() {
    problems.push(format!(
      "listed as both a vanilla European call and not one: {overlap:?}"
    ));
  }

  let union = vanilla
    .union(&not_vanilla)
    .cloned()
    .collect::<BTreeSet<_>>();
  if let Some(report) = diff(
    "vanilla / not-vanilla partition of `assert_model_pricer!`",
    &model,
    &union,
  ) {
    problems.push(report);
  }
  problems
}

/// The union of both pricer signals has to be exactly what the file accounts
/// for: a type in neither a trait list nor the deliberate-omission list is
/// the orphan this guard exists to catch, and a name in the omission list
/// that the source no longer has is a stale excuse.
fn coverage_problems(inventory: &Inventory, registered: &Registered<'_>) -> Vec<String> {
  let mut problems = Vec::new();
  let no_trait = registered
    .no_trait_by_design
    .iter()
    .map(|(name, _)| base_name(name))
    .collect::<BTreeSet<_>>();

  let mut universe = inventory.named_structs.clone();
  universe.extend(inventory.all_implementors());

  let mut accounted = names(registered.model_pricer);
  accounted.extend(names(registered.short_rate_pricer));
  accounted.extend(
    registered
      .pricing_engine
      .iter()
      .map(|(engine, _)| base_name(engine)),
  );
  accounted.extend(no_trait.iter().cloned());

  if let Some(report) = diff(
    "pricers in the source vs. pricers this file accounts for",
    &universe,
    &accounted,
  ) {
    problems.push(report);
  }

  let contradicted = no_trait
    .intersection(&inventory.all_implementors())
    .cloned()
    .collect::<Vec<_>>();
  if !contradicted.is_empty() {
    problems.push(format!(
      "excused as carrying no pricing trait, yet the source gives them one: {contradicted:?}"
    ));
  }

  let duplicated = registered
    .no_trait_by_design
    .len()
    .checked_sub(no_trait.len())
    .filter(|count| *count > 0);
  if let Some(count) = duplicated {
    problems.push(format!(
      "the deliberate-omission list repeats {count} name(s)"
    ));
  }
  problems
}

fn names(list: &[&str]) -> BTreeSet<String> {
  list.iter().map(|entry| base_name(entry)).collect()
}
