//! Driving-noise generators the process modules consume: Gaussian increments (`gn`, `wn`),
//! correlated pairs (`cgns`) and matrices (`mcgns`), fractional Gaussian noise (`fgn`, `cfgns`).
//!
pub mod cfgns;
pub mod cgns;
pub mod fgn;
pub mod gn;
pub mod mcgns;
pub mod wn;
