//! # Bsm
//!
//! $$
//! C=S_0e^{(b-r)T}N(d_1)-Ke^{-rT}N(d_2),\quad d_{1,2}=\frac{\ln(S_0/K)+(b\pm\tfrac12\sigma^2)T}{\sigma\sqrt T}
//! $$
//!
mod greeks;
mod pricer;

#[cfg(test)]
mod tests;

pub use pricer::BSMCoc;
pub use pricer::BSMPricer;
pub use pricer::BSMPricerBuilder;
