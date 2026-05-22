//! # Sabr
//!
//! Implied-volatility approximation and pricing under the Sabr stochastic
//! volatility model.
//!
//! **Reference:** P. S. Hagan, D. Kumar, A. S. Lesniewski, D. E. Woodward,
//! *Managing Smile Risk*, Wilmott Magazine, pp. 84–108, 2002.
//! (Eq. A.69a–A.69c for the general-β lognormal vol expansion.)
//!
//! $$
//! dF_t=\alpha_t F_t^\beta dW_t^1,\quad d\alpha_t=\nu\alpha_t dW_t^2,\ d\langle W^1,W^2\rangle_t=\rho dt
//! $$
//!
mod hagan;
mod pricer;

#[cfg(test)]
mod tests;

pub use hagan::alpha_from_atm_vol;
pub use hagan::bs_price_fx;
pub use hagan::forward_fx;
pub use hagan::fx_delta_from_forward;
pub use hagan::hagan_implied_vol;
pub use hagan::hagan_implied_vol_beta1;
pub use hagan::model_price_hagan;
pub use hagan::model_price_hagan_general;
pub use pricer::SabrModel;
pub use pricer::SabrPricer;
pub use pricer::SabrPricerBuilder;
