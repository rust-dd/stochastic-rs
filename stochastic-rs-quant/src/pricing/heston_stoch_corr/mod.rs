//! # Heston with Stochastic Correlation
//!
//! Characteristic-function based pricer for the Heston model where the
//! price-variance correlation ρ_t follows a mean-reverting Ou process
//! (Teng, Ehrhardt & Günther, 2016).
//!
//! $$
//! dS = rS\,dt + \sqrt{v}\,S\,dW^S, \quad
//! dv = \kappa_v(\theta_v - v)\,dt + \sigma_v\sqrt{v}\,dW^v, \quad
//! d\rho = \kappa_\rho(\mu_\rho - \rho)\,dt + \sigma_\rho\,dW^\rho
//! $$
//!
//! with  dW^S·dW^v = ρ_t dt  and  dW^v·dW^ρ = ρ₂ dt.
//!
//! ## Pricing methods
//!
//! - **Dampened Fourier inversion** via double-exponential quadrature on
//!   the Carr-Madan-style modified call transform. The transform itself
//!   is the Carr-Madan (1999) construction (damping factor α on the call
//!   payoff to make the integrand integrable along the real axis); the
//!   inversion is computed by Takahasi-Mori (1974) double-exponential
//!   quadrature rather than the FFT, since the chf is computed from a
//!   numerical Riccati ODE per evaluation point and an FFT grid would
//!   amortise to a single-point cost only at very high strike counts.
//!
//! ## References
//!
//! - Teng, Ehrhardt & Günther (2016), *On the Heston model with stochastic
//!   correlation*, Int. J. Theor. Appl. Finance 19(6).
//! - Carr & Madan (1999), *Option valuation using the FFT* — the dampened
//!   transform; here applied without the FFT step.
//! - Takahasi & Mori (1974), *Double exponential formulas for numerical
//!   integration*, Publ. Res. Inst. Math. Sci. 9(3) — the inversion
//!   quadrature scheme actually used.
//! - Tanas, R. — <https://github.com/tanasr/HestonStochCorr> (reference
//!   Python implementation).

pub mod cf;
pub mod model;
pub mod pricer;

#[cfg(test)]
mod tests;

pub use model::HestonStochCorrPricer;
pub use model::HestonStochCorrPricerBuilder;
pub use pricer::HscmModel;
