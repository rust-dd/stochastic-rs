//! `sheet/` slice of the exhaustive reproducibility guard — see
//! `../reproducibility_all_processes.rs` for the full rationale, the
//! derivation of the 130-type list, and shared methodology notes. The
//! smallest slice: `sheet/` contributes exactly one process type.

use stochastic_rs_stochastic::sheet::fbs::Fbs;

use crate::common::guard;

guard!(fbs, "Fbs", |s| Fbs::new(0.7, 8, 8, 0.5, s));
