//! # Double Heston Calibration
//!
//! $$
//! \begin{aligned}
//! dS_t &= (r-q)S_t\,dt + \sqrt{v_{1,t}}\,S_t\,dW_{1,t}^S + \sqrt{v_{2,t}}\,S_t\,dW_{2,t}^S \\
//! dv_{1,t} &= \kappa_1(\theta_1 - v_{1,t})\,dt + \sigma_1\sqrt{v_{1,t}}\,dW_{1,t}^v \\
//! dv_{2,t} &= \kappa_2(\theta_2 - v_{2,t})\,dt + \sigma_2\sqrt{v_{2,t}}\,dW_{2,t}^v
//! \end{aligned}
//! $$
//! with $d\langle W_1^S,W_1^v\rangle_t=\rho_1\,dt$,
//! $d\langle W_2^S,W_2^v\rangle_t=\rho_2\,dt$, and every other Brownian
//! motion pair independent.
//!
//! The characteristic function of $\ln(S_T/S_0)$ factorises into a sum of
//! two Heston contributions:
//! $$
//! \phi_T(u) = \exp\!\left(iu(r-q)T + \sum_{j=1}^2\bigl[C_j(u,T) + D_j(u,T)\,v_{j,0}\bigr]\right)
//! $$
//!
//! Source:
//! - Christoffersen, Heston & Jacobs (2009), "The Shape and Term Structure of
//!   the Index Option Smirk: Why Multifactor Stochastic Volatility Models Work
//!   So Well", Management Science 55(12), 1914-1932,
//!   <https://doi.org/10.1287/mnsc.1090.1065>
//! - Mehrdoust, Noorani & Hamdi (2021), "Calibration of the double Heston
//!   model and an analytical formula in pricing American put option",
//!   J. Comput. Appl. Math. 392, 113422,
//!   <https://doi.org/10.1016/j.cam.2021.113422>
//! - Levenberg (1944), <https://doi.org/10.1090/qam/10666>
//! - Marquardt (1963), <https://doi.org/10.1137/0111030>

mod calibrator;
mod loss;
mod params;
mod result;

pub use calibrator::DoubleHestonCalibrator;
pub use params::DoubleHestonParams;
pub use result::DoubleHestonCalibrationResult;

#[cfg(test)]
mod quadrature_tests;

#[cfg(test)]
mod tests;
