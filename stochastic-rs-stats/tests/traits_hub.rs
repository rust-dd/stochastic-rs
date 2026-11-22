//! The crate's own `traits` module names every trait it exports; the umbrella hub reads them from
//! here.

#![allow(unused_imports)]

use stochastic_rs_stats::traits::DiffusionModel;
use stochastic_rs_stats::traits::FractalDimEstimator;
use stochastic_rs_stats::traits::HurstEstimator;
use stochastic_rs_stats::traits::HypothesisTest;

#[test]
fn the_stats_traits_resolve_through_the_crate_hub_compile() {}
