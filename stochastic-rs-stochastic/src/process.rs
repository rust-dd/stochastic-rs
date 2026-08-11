//! # Process
//!
//! Foundational point processes and building blocks that don't fit the
//! diffusion/volatility/jump taxonomy: plain and correlated Brownian
//! motion (`bm`, `cbms`), the endpoint-pinned Brownian bridge
//! (`brownian_bridge`), fractional Brownian motion and its correlated
//! pair (`fbm`, `cfbms`), Poisson and Hawkes counting processes
//! (`poisson`, `hawkes`, `multivariate_hawkes`), compound-jump generators
//! with custom inter-arrival laws (`cpoisson`, `customjt`, `ccustom`),
//! linear fractional stable motion (`lfsm`), the generic Volterra-kernel
//! Gaussian process (`volterra`), and monotone Lévy subordinators
//! (`subordinator`). Each module's own header states its concrete law.
//!
pub mod bm;
pub mod brownian_bridge;
pub mod cbms;
pub mod ccustom;
pub mod cfbms;
pub mod cpoisson;
pub mod customjt;
pub mod fbm;
pub mod hawkes;
pub mod lfsm;
pub mod multivariate_hawkes;
pub mod poisson;
pub mod subordinator;
pub mod volterra;
