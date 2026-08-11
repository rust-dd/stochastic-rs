//! # Noise
//!
//! Raw driving-noise generators consumed by the process modules rather than
//! simulated as standalone SDEs: i.i.d. Gaussian increments (`gn`, `wn`),
//! correlated Gaussian pairs (`cgns`), fractional-Gaussian-noise for
//! rough/fractional processes (`fgn`, and its correlated pair `cfgns`).
//! Each module's own header states its concrete increment law.
//!
pub mod cfgns;
pub mod cgns;
pub mod fgn;
pub mod gn;
pub mod wn;
