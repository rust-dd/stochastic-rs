//! # Jump
//!
//! Discontinuous-path processes: a continuous (Brownian, possibly zero)
//! component plus a jump component driven by a Poisson-type counting
//! process or a subordinated time-change. Covers compound-Poisson
//! jump-diffusions on top of GBM/Heston (Merton, Kou, Bates, HKDE,
//! self-exciting Hawkes-driven jumps), pure-jump Lévy processes built by
//! Brownian subordination (VG, NIG, bilateral Gamma), and tempered-stable
//! / CGMY-family processes built from a truncated Rosiński series. Each
//! module's own header states its concrete jump mechanism.
//!
pub mod bates;
pub mod bilateral_gamma;
pub mod cgmy;
pub mod cts;
pub mod hawkes_jd;
pub mod ig;
pub mod jump_fou;
pub mod jump_fou_custom;
pub mod kobol;
pub mod kou;
pub mod levy_diffusion;
pub mod merton;
pub mod mjd_log;
pub mod nig;
pub mod rdts;
pub mod vg;
