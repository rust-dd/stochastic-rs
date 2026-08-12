//! `noise/` slice of the exhaustive reproducibility guard — see
//! `../reproducibility_all_processes.rs` for the full rationale, the
//! derivation of the 124-type list, and shared methodology notes. `Cfgns`
//! and `Fgn` are two of the nine backend-generic types named there,
//! instantiated on the default `Cpu` backend (their only inherent `new()`).

use stochastic_rs_stochastic::noise::cfgns::Cfgns;
use stochastic_rs_stochastic::noise::cgns::Cgns;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::noise::gn::Gn;
use stochastic_rs_stochastic::noise::wn::Wn;

use crate::common::N;
use crate::common::guard;

guard!(cfgns, "Cfgns", |s| Cfgns::new(0.7, 0.3, N, Some(1.0), s));
guard!(cgns, "Cgns", |s| Cgns::new(0.3, N, Some(1.0), s));
guard!(fgn, "Fgn", |s| Fgn::new(0.7, N, Some(1.0), s));
guard!(gn, "Gn", |s| Gn::new(N, Some(1.0), s));
guard!(wn, "Wn", |s| Wn::new(N, Some(0.0), Some(1.0), s));
