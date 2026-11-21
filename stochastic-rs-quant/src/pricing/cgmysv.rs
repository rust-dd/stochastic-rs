//! CGMYSV, a CGMY process on a CIR clock, priced by Monte Carlo over Rosiński series paths.
//!
//! $$
//! L_t = Z_{\mathcal{V}_t} + \rho\,v_t,\quad
//! dv_t = \kappa(\eta - v_t)\,dt + \zeta\sqrt{v_t}\,dW_t
//! $$
//!
//! European, American (least-squares Monte Carlo), Asian and barrier options.
//! Reference: Kim (2021), arXiv:2101.11001

pub mod model;
pub mod pricer;

pub use model::CgmysvModel;
pub use model::CgmysvParams;
pub use pricer::CgmysvPricer;
