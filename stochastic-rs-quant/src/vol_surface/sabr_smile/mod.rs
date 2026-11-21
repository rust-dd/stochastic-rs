//! FX smile calibration under Hagan's SABR expansion with general β.
//!
//! `α` follows from the ATM vol (eq. A.69b), so the search is `(ρ, ν)` and four 25-delta strikes:
//!
//! $$
//! \sigma_{imp}(K)\approx \sigma_{Hagan}(K;\alpha(\sigma_{ATM},\rho,\nu),\beta,\rho,\nu)
//! $$
//!
//! Reference: Hagan et al. (2002), Wilmott Mag., Sept., 84–108

mod calibrate;
mod objective;
mod types;

#[cfg(test)]
mod tests;

pub use calibrate::strike_for_delta;
pub use types::SabrSmileCalibrator;
pub use types::SabrSmileQuotes;
pub use types::SmileStrikes;
