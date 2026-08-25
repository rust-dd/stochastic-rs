//! Guards `stochastic_rs::prelude`'s documented item list (CLAUDE.md's
//! "Key traits"/"Prelude" sections, `website/content/docs/concepts/prelude.mdx`)
//! against the failure mode that already happened once: `VolterraKernel`
//! was added to the prelude and the docs kept saying "20 items" for
//! several releases afterward, because nothing forced anyone to touch the
//! docs when the prelude's contents changed.
//!
//! Naming every documented item explicitly (rather than `use
//! stochastic_rs::prelude::*;`) means removing or renaming one is a
//! compile error here, not a silent doc/reality mismatch. This cannot
//! catch the opposite direction (a new item added to the prelude but
//! never documented) — for that, re-run the derivation below and compare
//! against the seven group lists in `CLAUDE.md` and `prelude.mdx`:
//!
//! `awk '/pub mod prelude/,/^}/' src/lib.rs | grep -c "^  pub use"`

#![allow(unused_imports)]

use stochastic_rs::prelude::Backend;
use stochastic_rs::prelude::BivariateExt;
use stochastic_rs::prelude::CalibrationResult;
use stochastic_rs::prelude::Calibrator;
use stochastic_rs::prelude::Cpu;
use stochastic_rs::prelude::DiffusionModel;
use stochastic_rs::prelude::DistributionExt;
use stochastic_rs::prelude::DistributionSampler;
use stochastic_rs::prelude::FloatExt;
use stochastic_rs::prelude::FractalDimEstimator;
use stochastic_rs::prelude::GreeksExt;
use stochastic_rs::prelude::HurstEstimator;
use stochastic_rs::prelude::HypothesisTest;
use stochastic_rs::prelude::Instrument;
use stochastic_rs::prelude::InstrumentExt;
use stochastic_rs::prelude::ModelPricer;
use stochastic_rs::prelude::Moneyness;
use stochastic_rs::prelude::OptionStyle;
use stochastic_rs::prelude::OptionType;
use stochastic_rs::prelude::PathSampler;
use stochastic_rs::prelude::PricingEngine;
use stochastic_rs::prelude::PricingResult;
use stochastic_rs::prelude::ProcessExt;
use stochastic_rs::prelude::SimdFloatExt;
use stochastic_rs::prelude::TailDependence;
use stochastic_rs::prelude::TimeExt;
use stochastic_rs::prelude::ToModel;
use stochastic_rs::prelude::VolterraKernel;

#[test]
fn all_twenty_eight_documented_prelude_items_resolve() {
  // The import above is the assertion: if it compiles, every name CLAUDE.md
  // and prelude.mdx list is still a real prelude export. Nothing to run.
}

/// The other half of the documented contract: a trait kept **out** of the
/// prelude is still reachable via `stochastic_rs::traits::*`. CLAUDE.md says
/// that for `MalliavinExt`, `MultivariateExt` and `CallableDist`, and
/// `prelude.mdx`'s "What is *not* in the prelude (and why)" section repeats
/// it — but nothing forced the hub to keep the promise, and `ShortRatePricer`
/// (half of the headline `ModelPricer`/`ShortRatePricer` pair) and
/// `VanillaEuropeanCall` had both fallen through it, reachable only as the
/// much longer `stochastic_rs::quant::traits::…`.
///
/// Every bullet of that section is named below, the two feature-gated ones
/// behind the same gates the hub uses — so this compiles on a default build
/// and still covers `MultivariateExt` / `CallableDist` when those features
/// are on.
mod prelude_excluded_traits_stay_hub_reachable {
  #[cfg(feature = "python")]
  use stochastic_rs::traits::CallableDist;
  use stochastic_rs::traits::Malliavin2DExt;
  use stochastic_rs::traits::MalliavinExt;
  #[cfg(feature = "openblas")]
  use stochastic_rs::traits::MultivariateExt;
  use stochastic_rs::traits::ShortRatePricer;
  use stochastic_rs::traits::ToShortRateModel;
  use stochastic_rs::traits::VanillaEuropeanCall;

  #[test]
  fn every_prelude_excluded_trait_resolves_through_the_hub() {
    // The imports above are the assertion. Nothing to run.
  }
}
