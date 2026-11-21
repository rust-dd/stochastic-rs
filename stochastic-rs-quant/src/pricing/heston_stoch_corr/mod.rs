//! Heston with an Ornstein–Uhlenbeck correlation `ρ_t`, priced from its characteristic function.
//!
//! $$
//! dS = rS\,dt + \sqrt{v}\,S\,dW^S, \quad
//! dv = \kappa_v(\theta_v - v)\,dt + \sigma_v\sqrt{v}\,dW^v, \quad
//! d\rho = \kappa_\rho(\mu_\rho - \rho)\,dt + \sigma_\rho\,dW^\rho,
//! \quad dW^S dW^v = \rho_t\,dt,\ dW^v dW^\rho = \rho_2\,dt
//! $$
//!
//! Carr–Madan's damped transform by double-exponential quadrature, as each chf costs an RK4 solve.
//! Reference: Teng et al. (2016), Int. J. Theor. Appl. Finance 19(6), DOI 10.1142/S0219024916500333
//! Reference: Carr & Madan (1999), J. Comput. Finance 2(4), 61–73
//! Reference: Takahasi & Mori (1974), Publ. RIMS 9, 721–741, DOI 10.2977/prims/1195192451
//! Source: <https://github.com/tanasr/HestonStochCorr>

pub mod cf;
pub mod model;
pub mod pricer;

#[cfg(test)]
mod tests;

pub use model::HestonStochCorrPricer;
