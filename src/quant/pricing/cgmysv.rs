//! # CGMYSV — Stochastic Volatility CGMY Process
//!
//! $$
//! L_t = Z_{\mathcal{V}_t} + \rho\,v_t,\quad
//! dv_t = \kappa(\eta - v_t)\,dt + \zeta\sqrt{v_t}\,dW_t
//! $$
//!
//! Monte Carlo option pricing via sample path generation of the CGMYSV process
//! using the Rosiński series representation with CIR-driven stochastic time change.
//!
//! Supports European, American (LSM), Asian, and Barrier option pricing.
//!
//! Reference: Kim, Y. S. (2021), "Sample path generation of the stochastic volatility
//! CGMY process and its application to path-dependent option pricing",
//! arXiv:2101.11001

pub mod model;
pub mod path_gen;
pub mod pricer;

pub use model::{CgmysvModel, CgmysvParams};
pub use path_gen::CgmysvPathGen;
pub use pricer::{CgmysvPricer, McResult};
