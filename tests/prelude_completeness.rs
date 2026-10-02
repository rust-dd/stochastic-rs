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
use stochastic_rs::prelude::HurstEstimator;
use stochastic_rs::prelude::HypothesisTest;
use stochastic_rs::prelude::ModelPricer;
use stochastic_rs::prelude::Moneyness;
use stochastic_rs::prelude::MultivariateExt;
use stochastic_rs::prelude::OptionStyle;
use stochastic_rs::prelude::OptionType;
use stochastic_rs::prelude::PathSampler;
use stochastic_rs::prelude::ProcessExt;
use stochastic_rs::prelude::RealExt;
use stochastic_rs::prelude::SimdFloatExt;
use stochastic_rs::prelude::TailDependence;
use stochastic_rs::prelude::TimeExt;
use stochastic_rs::prelude::ToModel;
use stochastic_rs::prelude::VolterraKernel;

#[test]
fn all_twenty_five_documented_prelude_items_resolve() {
  // The import above is the assertion: if it compiles, every name CLAUDE.md
  // and prelude.mdx list is still a real prelude export. Nothing to run.
}

/// A trait kept out of the prelude stays reachable via `stochastic_rs::traits::*`, as CLAUDE.md and
/// `prelude.mdx` promise; every bullet of "What is *not* in the prelude" is named below.
mod prelude_excluded_traits_stay_hub_reachable {
  use stochastic_rs::traits::FgnBackend;
  use stochastic_rs::traits::GreeksExt;
  use stochastic_rs::traits::Instrument;
  use stochastic_rs::traits::InstrumentExt;
  use stochastic_rs::traits::PricingEngine;
  use stochastic_rs::traits::PricingResult;
  use stochastic_rs::traits::SheetBackend;
  use stochastic_rs::traits::ShortRatePricer;
  use stochastic_rs::traits::ToShortRateModel;
  use stochastic_rs::traits::VanillaEuropeanCall;

  #[test]
  fn every_prelude_excluded_trait_resolves_through_the_hub() {
    // The imports above are the assertion. Nothing to run.
  }
}
