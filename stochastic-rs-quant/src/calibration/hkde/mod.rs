//! # Hkde (Heston + Kou Double-Exponential) Calibration
//!
//! $$
//! \begin{aligned}
//! dS_t &= (r-q-\lambda\bar k)S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S + S_t(e^{J_t}-1)\,dN_t \\
//! dv_t &= \kappa(\theta-v_t)\,dt + \sigma_v\sqrt{v_t}\,dW_t^v, \quad
//! d\langle W^S,W^v\rangle_t = \rho\,dt
//! \end{aligned}
//! $$
//!
//! where $N_t$ is a Poisson process with intensity $\lambda$ and jump sizes
//! follow an asymmetric double-exponential distribution
//!
//! $$
//! f_J(y) = p\,\eta_1 e^{-\eta_1 y}\,\mathbb{1}_{y\ge 0}
//!        + (1-p)\,\eta_2 e^{\eta_2 y}\,\mathbb{1}_{y<0},\qquad \eta_1>1,\ \eta_2>0.
//! $$
//!
//! The characteristic function factorises as
//! $\phi_{\mathrm{Hkde}}(\xi,t) = \phi_{\mathrm{Hes}}(\xi,t)\,\phi_{\mathrm{Kou}}(\xi,t)$
//! and is provided by [`HKDEFourier`](crate::pricing::fourier::HKDEFourier).
//!
//! The nine-dimensional parameter vector $\theta=(v_0,\kappa,\theta,\sigma_v,\rho,
//! \lambda,p,\eta_1,\eta_2)$ is calibrated against a set of market quotes by
//! minimising the vega-weighted least-squares objective (Eq. 12 of the paper):
//!
//! $$
//! \theta^{\*} = \arg\min_{\theta\in\Theta}
//! \sum_{n=1}^{N}\sum_{j=1}^{N_n} w_j^{(n)}\bigl(\mathcal V_j^{(n)}(\theta)-v_j^{(n)}\bigr)^2,
//! $$
//!
//! with vega weights (Eq. 13)
//!
//! $$
//! w_j^{(n)} := \bigl(S_0\,\psi(d_1)\,\sqrt{T_n}\bigr)^{-1},\qquad
//! d_1 = \frac{\log(S_0/K_j^{(n)}) + T_n\bigl(r_n-q_n+\tfrac12(\sigma_j^{(n)})^2\bigr)}
//!            {\sigma_j^{(n)}\sqrt{T_n}},
//! $$
//!
//! where $\psi$ is the standard-normal pdf and $\sigma_j^{(n)}$ is the market
//! implied volatility of the quote. The weights are computed once from the
//! observed market prices and kept fixed during the Levenberg-Marquardt
//! iterations.
//!
//! Source:
//! - Agazzotti, Aglieri Rinella, Aguilar, Kirkby (2025),
//!   "Calibration and Option Pricing with stochastic volatility and double
//!   exponential jumps", arXiv: 2502.13824
//! - Kou, S. G. (2002), "A jump-diffusion model for option pricing",
//!   Management Science 48(8), https://doi.org/10.1287/mnsc.48.8.1086.166
//! - Heston, S. L. (1993), "A closed-form solution for options with stochastic
//!   volatility", https://doi.org/10.1093/rfs/6.2.327

mod calibrator;
mod loss;
mod params;
mod result;

pub use calibrator::HKDECalibrator;
pub use loss::paper_table1;
pub use loss::paper_table2;
pub use params::HKDEParams;
pub use result::HKDECalibrationResult;

#[cfg(test)]
mod tests;
