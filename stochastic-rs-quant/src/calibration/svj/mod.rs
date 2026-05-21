//! # SVJ (Bates) Calibration
//!
//! $$
//! \begin{aligned}
//! dS_t &= (r-q-\lambda\bar k)S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S + J_t\,S_t\,dN_t \\
//! dv_t &= \kappa(\theta-v_t)\,dt + \sigma_v\sqrt{v_t}\,dW_t^v, \quad
//! d\langle W^S,W^v\rangle_t = \rho\,dt
//! \end{aligned}
//! $$
//!
//! where $J_t\sim\mathcal{N}(\mu_J,\sigma_J^2)$ are i.i.d. log-jump sizes and
//! $N_t$ is a Poisson process with intensity $\lambda$.
//!
//! The characteristic function is
//! $$
//! \phi_T(\xi) = \phi_T^{\mathrm{Heston}}(\xi)\cdot
//! \exp\!\bigl[T\,\lambda\bigl(e^{i\mu_J\xi - \tfrac12\sigma_J^2\xi^2} - 1\bigr)\bigr].
//! $$
//!
//! Source:
//! - Bates, D. (1996), "Jumps and Stochastic Volatility"
//!   https://doi.org/10.1093/rfs/9.1.69
//! - Heston, S. L. (1993)
//!   https://doi.org/10.1093/rfs/6.2.327

mod calibrator;
mod loss;
mod params;
mod result;

pub use calibrator::SVJCalibrator;
pub use params::SVJParams;
pub use result::SVJCalibrationResult;

#[cfg(test)]
mod tests;
