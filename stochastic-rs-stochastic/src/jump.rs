//! Discontinuous paths: compound-Poisson jump diffusions (Merton, Kou, Bates, Hawkes), Brownian
//! subordination (VG, NIG, bilateral Gamma) and tempered-stable series (CGMY family, Rosiński).
//!
pub mod bates;
pub mod bilateral_gamma;
pub mod cgmy;
pub mod cts;
pub mod hawkes_jd;
pub mod jump_fou;
pub mod jump_fou_custom;
pub mod kobol;
pub mod kou;
pub mod levy_diffusion;
pub mod merton;
pub mod mjd_log;
pub mod nig;
pub mod rdts;
pub(crate) mod series_remainder;
pub mod vg;
