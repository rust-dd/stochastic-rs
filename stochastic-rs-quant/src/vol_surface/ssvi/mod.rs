//! SSVI, Gatheral & Jacquier's (2012) full-surface parameterisation, `θ_t` the ATM total variance:
//!
//! $$
//! w(k,\theta_t)=\frac{\theta_t}{2}\Bigl(1+\rho\,\varphi(\theta_t)\,k
//!   +\sqrt{\bigl(\varphi(\theta_t)\,k+\rho\bigr)^2+(1-\rho^2)}\Bigr)
//! $$
//!
//! with the power-law mixing `φ` of their eq. 4.1:
//!
//! $$
//! \varphi(\theta)=\frac{\eta}{\theta^\gamma\,(1+\theta)^{1-\gamma}},
//! \quad \eta>0,\;\gamma\in\left[0,1\right]
//! $$
//!
//! Reference: Gatheral & Jacquier (2012), arXiv:1204.0646
//! Reference: Corbetta et al. (2019), Decis. Econ. Finance 42(2), 665–677, arXiv:1804.04924

mod calibrate;
mod params;
mod surface;

#[cfg(test)]
mod tests;

pub use calibrate::calibrate_ssvi;
pub use params::SsviParams;
pub use params::SsviSlice;
pub use surface::SsviSurface;
