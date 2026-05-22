//! # Sabr Smile
//!
//! FX volatility smile calibrator using the Hagan (2002) Sabr approximation
//! with general β support.
//!
//! **Reference:** P. S. Hagan, D. Kumar, A. S. Lesniewski, D. E. Woodward,
//! *Managing Smile Risk*, Wilmott Magazine, pp. 84–108, 2002.
//!
//! α is derived analytically from the ATM vol (Eq. A.69b) so the
//! optimizer only searches over (ρ, ν) plus the four 25-delta strike levels.
//!
//! $$
//! \sigma_{imp}(K)\approx \sigma_{Hagan}(K;\alpha(\sigma_{ATM},\rho,\nu),\beta,\rho,\nu)
//! $$

mod calibrate;
mod objective;
mod types;

#[cfg(test)]
mod tests;

pub use calibrate::strike_for_delta;
pub use types::SabrSmileCalibrator;
pub use types::SabrSmileQuotes;
pub use types::SabrSmileResult;
