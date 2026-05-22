//! # SSVI — Surface SVI
//!
//! Full-surface parameterization from Gatheral & Jacquier (2012):
//!
//! $$
//! w(k,\theta_t)=\frac{\theta_t}{2}\Bigl(1+\rho\,\varphi(\theta_t)\,k
//!   +\sqrt{\bigl(\varphi(\theta_t)\,k+\rho\bigr)^2+(1-\rho^2)}\Bigr)
//! $$
//!
//! where $\theta_t$ is the ATM total variance at time $t$ and $\varphi$ is
//! a mixing function.
//!
//! Power-law mixing (Gatheral & Jacquier, 2012, Eq. 4.1):
//!
//! $$
//! \varphi(\theta)=\frac{\eta}{\theta^\gamma\,(1+\theta)^{1-\gamma}},
//! \quad \eta>0,\;\gamma\in[0,1]
//! $$
//!
//! Calibration based on Cohort, Corbetta, Martini & Laachir (2018),
//! arXiv:1804.04924.
//!
//! Reference: Gatheral & Jacquier (2012), arXiv:1204.0646

mod calibrate;
mod params;
mod surface;

#[cfg(test)]
mod tests;

pub use calibrate::calibrate_ssvi;
pub use params::SsviParams;
pub use params::SsviSlice;
pub use surface::SsviSurface;
