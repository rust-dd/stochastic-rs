//! Re-derives this crate's pricing inventory from `src/**/*.rs`.
//!
//! `pricer_registry.rs` holds a hand-written list; this module produces the
//! list the source actually implies, so the two can be diffed and a pricer
//! added without a registry entry fails a test instead of passing unnoticed.
//!
//! The reader is `syn` — the parser procedural macros use — and not a regex
//! over lines. That choice is the whole point: a line regex sees `impl` inside
//! a doc comment or a string literal, cannot separate `struct FooPricer` from
//! `struct FooPricerBuilder` without a word-boundary trick this repository has
//! already got wrong, and has no way at all to skip a `#[cfg(test)]` module.
//! `derivation_tests` feeds every one of those shapes through `parse_source`
//! and pins what comes out, so the derivation is checked rather than trusted.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

pub use audit::Registered;
pub use audit::audit;
use syn::Attribute;
use syn::Item;
use syn::ItemImpl;
use syn::ItemStruct;
use syn::Meta;
use syn::PathArguments;
use syn::Token;
use syn::Type;
use syn::TypeParamBound;
use syn::Visibility;
use syn::punctuated::Punctuated;

mod audit;
mod derivation_tests;

/// The traits `pricer_registry.rs` enumerates implementors of. A trait absent
/// from this list is invisible to the scan, which is why adding one to the
/// registry means adding it here too.
pub const TRACKED_TRAITS: [&str; 4] = [
  "ModelPricer",
  "PricingEngine",
  "ShortRatePricer",
  "VanillaEuropeanCall",
];

/// Identifier *suffixes* that make a `pub struct` a pricer by name. Matching
/// the suffix of the parsed identifier rather than a substring of the source
/// line is what excludes `KirkSpreadPricerBuilder` and `PortfolioEngineConfig`
/// without needing a word-boundary trick to be spelled correctly.
pub const PRICER_NAME_SUFFIXES: [&str; 2] = ["Pricer", "Engine"];

/// Directory under `src/` the scan skips: the PyO3 wrappers are
/// `#[cfg(feature = "python")]`, hold their subject in an `inner` field and
/// carry no trait of their own, so they belong to no pricing family.
const EXCLUDED_DIR: &str = "python";

const PRICING_ENGINE: &str = "PricingEngine";

/// What the source says about pricing traits and pricer-shaped structs.
#[derive(Debug, Default)]
pub struct Inventory {
  /// `impl <tracked trait> for <named type>`, keyed by trait, valued by the
  /// self type's base identifier — generic arguments dropped, so
  /// `impl ModelPricer for CrrModel<f64>` lands under `CrrModel`.
  pub concrete: BTreeMap<String, BTreeSet<String>>,
  /// `impl<T: Bound> <tracked trait> for T`, keyed by trait, valued by the
  /// bound that selects the implementors. A blanket impl hands the trait to
  /// types no per-type list can enumerate, so it is pinned rather than
  /// expanded.
  pub blanket: BTreeMap<String, BTreeSet<String>>,
  /// `(engine, instrument)` for every `impl PricingEngine<I> for E`.
  pub engine_pairs: BTreeSet<(String, String)>,
  /// `pub struct`s whose identifier ends in one of [`PRICER_NAME_SUFFIXES`].
  pub named_structs: BTreeSet<String>,
  /// Tracked-trait impls whose self type is neither a named path nor a
  /// generic parameter. Recorded instead of dropped: an unforeseen shape
  /// should fail a test, not quietly leave the inventory.
  pub unclassified: BTreeSet<String>,
}

impl Inventory {
  /// Concrete implementors of `trait_name`, empty if there are none.
  pub fn implementors(&self, trait_name: &str) -> BTreeSet<String> {
    self.concrete.get(trait_name).cloned().unwrap_or_default()
  }

  /// Every concrete implementor of every tracked trait.
  pub fn all_implementors(&self) -> BTreeSet<String> {
    self.concrete.values().flatten().cloned().collect()
  }
}

/// Parses every non-excluded `.rs` file under this crate's `src/`.
pub fn scan_crate_source() -> Inventory {
  let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
  let mut files = Vec::new();
  collect_rs_files(&root, &mut files);
  files.sort();
  assert!(
    !files.is_empty(),
    "no source files found under {} — the scan would vacuously agree with any registry",
    root.display()
  );

  let mut inventory = Inventory::default();
  for file in &files {
    let text =
      fs::read_to_string(file).unwrap_or_else(|err| panic!("reading {}: {err}", file.display()));
    let parsed =
      syn::parse_file(&text).unwrap_or_else(|err| panic!("parsing {}: {err}", file.display()));
    walk(&parsed.items, &mut inventory);
  }
  inventory
}

/// Parses one source text, for the fixtures in `derivation_tests`.
pub fn parse_source(text: &str) -> Inventory {
  let mut inventory = Inventory::default();
  let parsed = syn::parse_file(text).expect("fixture parses");
  walk(&parsed.items, &mut inventory);
  inventory
}

/// Reduces a `stringify!`d type to the base identifier the scan reports, so
/// `crate::lattice::equity::CrrModel<f64>` and `CrrModel < f64 >` both become
/// `CrrModel`.
pub fn base_name(rendered: &str) -> String {
  let head = rendered.split('<').next().unwrap_or(rendered).trim();
  head.rsplit("::").next().unwrap_or(head).trim().to_string()
}

/// Describes how `derived` and `registered` disagree, or `None` if they do
/// not. Both directions are reported: a missing entry is an unregistered
/// pricer, a stale one is a registry claim about a type that no longer
/// implements the trait.
pub fn diff(
  label: &str,
  derived: &BTreeSet<String>,
  registered: &BTreeSet<String>,
) -> Option<String> {
  let missing = derived.difference(registered).cloned().collect::<Vec<_>>();
  let stale = registered.difference(derived).cloned().collect::<Vec<_>>();
  if missing.is_empty() && stale.is_empty() {
    return None;
  }
  let mut message = format!("{label}:");
  if !missing.is_empty() {
    message.push_str(&format!(
      "\n  in the source, missing from the registry: {missing:?}"
    ));
  }
  if !stale.is_empty() {
    message.push_str(&format!(
      "\n  in the registry, absent from the source: {stale:?}"
    ));
  }
  Some(message)
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
  let entries = fs::read_dir(dir).unwrap_or_else(|err| panic!("listing {}: {err}", dir.display()));
  for entry in entries {
    let path = entry.expect("directory entry").path();
    if path.is_dir() {
      if path.file_name().is_some_and(|name| name == EXCLUDED_DIR) {
        continue;
      }
      collect_rs_files(&path, out);
    } else if path.extension().is_some_and(|ext| ext == "rs") {
      out.push(path);
    }
  }
}

fn walk(items: &[Item], inventory: &mut Inventory) {
  for item in items {
    match item {
      Item::Mod(node) if !is_cfg_test(&node.attrs) => {
        if let Some((_, inner)) = &node.content {
          walk(inner, inventory);
        }
      }
      Item::Impl(node) if !is_cfg_test(&node.attrs) => record_impl(node, inventory),
      Item::Struct(node) if !is_cfg_test(&node.attrs) => record_struct(node, inventory),
      _ => {}
    }
  }
}

fn record_impl(node: &ItemImpl, inventory: &mut Inventory) {
  if node.modifiers.polarity.is_some() {
    return;
  }
  let Some((trait_path, _)) = &node.trait_ else {
    return;
  };
  let Some(segment) = trait_path.segments.last() else {
    return;
  };
  let trait_name = segment.ident.to_string();
  if !TRACKED_TRAITS.contains(&trait_name.as_str()) {
    return;
  }

  let type_params = node
    .generics
    .type_params()
    .map(|param| param.ident.to_string())
    .collect::<BTreeSet<_>>();

  match self_ty_shape(&node.self_ty) {
    SelfTy::Named(name) if type_params.contains(&name) => {
      let bound = selecting_bound(node, &name);
      inventory
        .blanket
        .entry(trait_name)
        .or_default()
        .insert(bound);
    }
    SelfTy::Named(name) => {
      if trait_name == PRICING_ENGINE
        && let Some(instrument) = first_type_argument(&segment.arguments)
      {
        inventory.engine_pairs.insert((name.clone(), instrument));
      }
      inventory
        .concrete
        .entry(trait_name)
        .or_default()
        .insert(name);
    }
    SelfTy::Other(shape) => {
      inventory
        .unclassified
        .insert(format!("impl {trait_name} for <{shape}>"));
    }
  }
}

fn record_struct(node: &ItemStruct, inventory: &mut Inventory) {
  if !matches!(node.vis, Visibility::Public(_)) {
    return;
  }
  let name = node.ident.to_string();
  if PRICER_NAME_SUFFIXES
    .iter()
    .any(|suffix| name.ends_with(suffix))
  {
    inventory.named_structs.insert(name);
  }
}

enum SelfTy {
  Named(String),
  Other(&'static str),
}

fn self_ty_shape(ty: &Type) -> SelfTy {
  match ty {
    Type::Path(path) if path.qself.is_none() => match path.path.segments.last() {
      Some(segment) => SelfTy::Named(segment.ident.to_string()),
      None => SelfTy::Other("empty path"),
    },
    Type::Path(_) => SelfTy::Other("qualified path"),
    Type::Reference(_) => SelfTy::Other("reference"),
    Type::Tuple(_) => SelfTy::Other("tuple"),
    Type::Slice(_) => SelfTy::Other("slice"),
    Type::Array(_) => SelfTy::Other("array"),
    Type::TraitObject(_) => SelfTy::Other("trait object"),
    _ => SelfTy::Other("unrecognised"),
  }
}

/// The first non-`Sized` trait bound on generic parameter `param`, looked for
/// inline and then in the `where` clause.
fn selecting_bound(node: &ItemImpl, param: &str) -> String {
  let inline = node
    .generics
    .type_params()
    .filter(|candidate| candidate.ident == param)
    .flat_map(|candidate| candidate.bounds.iter())
    .find_map(bound_name);
  if let Some(name) = inline {
    return name;
  }
  if let Some(clause) = &node.generics.where_clause {
    for predicate in &clause.predicates {
      let syn::WherePredicate::Type(predicate) = predicate else {
        continue;
      };
      if !matches!(self_ty_shape(&predicate.bounded_ty), SelfTy::Named(ref name) if name == param) {
        continue;
      }
      if let Some(name) = predicate.bounds.iter().find_map(bound_name) {
        return name;
      }
    }
  }
  "unbounded".to_string()
}

fn bound_name(bound: &TypeParamBound) -> Option<String> {
  let TypeParamBound::Trait(bound) = bound else {
    return None;
  };
  let name = bound.path.segments.last()?.ident.to_string();
  (name != "Sized").then_some(name)
}

fn first_type_argument(arguments: &PathArguments) -> Option<String> {
  let PathArguments::AngleBracketed(arguments) = arguments else {
    return None;
  };
  arguments.args.iter().find_map(|argument| {
    let syn::GenericArgument::Type(ty) = argument else {
      return None;
    };
    match self_ty_shape(ty) {
      SelfTy::Named(name) => Some(name),
      SelfTy::Other(_) => None,
    }
  })
}

fn is_cfg_test(attrs: &[Attribute]) -> bool {
  attrs.iter().any(|attr| {
    attr.path().is_ident("cfg")
      && attr
        .parse_args::<Meta>()
        .is_ok_and(|meta| mentions_test(&meta))
  })
}

fn mentions_test(meta: &Meta) -> bool {
  match meta {
    Meta::Path(path) => path.is_ident("test"),
    Meta::List(list) => list
      .parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
      .is_ok_and(|nested| nested.iter().any(mentions_test)),
    Meta::NameValue(_) => false,
  }
}
