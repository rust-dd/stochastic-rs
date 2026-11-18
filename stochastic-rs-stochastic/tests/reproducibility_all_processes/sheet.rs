//! The `sheet/` slice of the reproducibility guard: one process type.

use stochastic_rs_stochastic::sheet::fbs::Fbs;

use crate::common::guard;

guard!(fbs, "Fbs", |s| Fbs::new(0.7, 8, 8, 0.5, s));
