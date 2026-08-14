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
//! against the six group lists in `CLAUDE.md` and `prelude.mdx`:
//!
//! `awk '/pub mod prelude/,/^}/' src/lib.rs | grep -c "pub use"`

#![allow(unused_imports)]

use stochastic_rs::prelude::Backend;
use stochastic_rs::prelude::BivariateExt;
use stochastic_rs::prelude::CalibrationResult;
use stochastic_rs::prelude::Calibrator;
use stochastic_rs::prelude::Cpu;
use stochastic_rs::prelude::DistributionExt;
use stochastic_rs::prelude::DistributionSampler;
use stochastic_rs::prelude::FloatExt;
use stochastic_rs::prelude::GreeksExt;
use stochastic_rs::prelude::Instrument;
use stochastic_rs::prelude::InstrumentExt;
use stochastic_rs::prelude::ModelPricer;
use stochastic_rs::prelude::Moneyness;
use stochastic_rs::prelude::OptionStyle;
use stochastic_rs::prelude::OptionType;
use stochastic_rs::prelude::PathSampler;
use stochastic_rs::prelude::PricerExt;
use stochastic_rs::prelude::PricingEngine;
use stochastic_rs::prelude::PricingResult;
use stochastic_rs::prelude::ProcessExt;
use stochastic_rs::prelude::SimdFloatExt;
use stochastic_rs::prelude::TimeExt;
use stochastic_rs::prelude::ToModel;
use stochastic_rs::prelude::VolterraKernel;

#[test]
fn all_twenty_four_documented_prelude_items_resolve() {
  // The import above is the assertion: if it compiles, every name CLAUDE.md
  // and prelude.mdx list is still a real prelude export. Nothing to run.
}
