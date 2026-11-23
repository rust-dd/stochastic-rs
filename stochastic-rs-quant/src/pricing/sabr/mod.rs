//! SABR implied volatility by Hagan's expansion, and pricing at that volatility.
//!
//! $$
//! dF_t=\alpha_t F_t^\beta dW_t^1,\quad d\alpha_t=\nu\alpha_t dW_t^2,\ d\langle W^1,W^2\rangle_t=\rho dt
//! $$
//!
//! Reference: Hagan et al. (2002), Wilmott Mag., 84–108, eqs. A.69a–A.69c (general β)
mod hagan;
mod params;
mod pricer;

#[cfg(test)]
mod tests;

pub use hagan::alpha_from_atm_vol;
pub(crate) use hagan::bs_price_fx;
pub use hagan::forward_fx;
pub use hagan::fx_delta_from_forward;
pub use hagan::hagan_implied_vol;
pub(crate) use hagan::model_price_hagan;
pub use params::SabrParams;
pub use pricer::SabrPricer;
