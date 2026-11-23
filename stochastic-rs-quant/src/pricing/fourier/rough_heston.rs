//! Rough Heston by its characteristic function, three ways: the fractional Riccati equation by
//! the Adams scheme, its rational approximation, and the lifted model's ordinary Riccati system.
//!
//! $$
//! V_t = V_0 + \int_0^t K(t-s)\,\kappa(\theta - V_s)\,ds + \int_0^t K(t-s)\,\nu\sqrt{V_s}\,dW^v_s,
//! \qquad K(t) = \frac{t^{\alpha-1}}{\Gamma(\alpha)},\quad \alpha = H + \tfrac12
//! $$
//!
//! $$
//! \ln\phi(t,\xi) = \kappa\theta\,I^{1}h(t) + V_0\,I^{1-\alpha}h(t),\qquad
//! D^{\alpha}h = F(z,h) = \tfrac12(z^2-z) + (\rho\nu z-\kappa)h + \tfrac12\nu^2h^2,\quad z = i\xi
//! $$
//!
//! Reference: El Euch & Rosenbaum (2019), Math. Finance 29(1), 3–38, DOI 10.1111/mafi.12173

mod params;
mod scalar;

#[cfg(test)]
mod tests;

pub use params::RoughHestonParams;
