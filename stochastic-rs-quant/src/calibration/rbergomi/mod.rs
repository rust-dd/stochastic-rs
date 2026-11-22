//! rBergomi calibration by distribution matching: the mean empirical Wasserstein-1 distance between
//! the model's mSOE terminal samples and the market's, per maturity.
//!
//! $$
//! dS_t=rS_t\,dt+S_t\sqrt{V_t}\left(\rho\,dW_t+\sqrt{1-\rho^2}\,dW_t^\perp\right),
//! $$
//! $$
//! V_t=\xi_0(t)\exp\left(\eta I_t-\frac{\eta^2}{2}t^{2H}\right),\quad
//! I_t=\sqrt{2H}\int_0^t (t-s)^{H-\frac12}\,dW_s.
//! $$
//! $$
//! L(\theta)=\frac1M\sum_{j=1}^M W_1\left(S_{T_j}(\theta),S_{T_j}^{\mathrm{MKT}}\right).
//! $$
//! $$
//! W_1\approx\frac1m\sum_{i=1}^m\left|X_{(i)}-Y_{(i)}\right|.
//! $$
//!
//! Reference: Bayer, Friz & Gatheral (2016), Quant. Finance 16(6), 887–904

mod calibrator;
mod loss;
mod params;
mod simulation;

pub(super) const H_MIN: f64 = 1e-3;
pub(super) const H_MAX: f64 = 0.499;
pub(super) const RHO_BOUND: f64 = 0.999;
pub(super) const ETA_MIN: f64 = 1e-8;
pub(super) const XI0_MIN: f64 = 1e-8;

pub use calibrator::RBergomiCalibrator;
pub use loss::bid_ask_tolerance;
pub use loss::empirical_wasserstein_1;
pub use params::RBergomiCalibrationConfig;
pub use params::RBergomiMarketSlice;
pub use params::RBergomiParams;
pub use params::RBergomiXi0;
pub(crate) use simulation::simulate_rbergomi_antithetic_pairs;
pub use simulation::simulate_rbergomi_terminal_samples;

#[cfg(test)]
mod tests;
