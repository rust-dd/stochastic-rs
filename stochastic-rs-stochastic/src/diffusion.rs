//! # Diffusion
//!
//! Continuous-path Itô diffusions `dX_t = a(t, X_t)dt + b(t, X_t)dW_t`
//! driven by a single standard Brownian motion, with no jump component.
//! Covers mean-reverting families (Ornstein–Uhlenbeck, CIR, Jacobi, Pearson
//! and their fractional/nonlinear variants), non-mean-reverting log-normal
//! families (GBM and its inhomogeneous/log-domain variants), and
//! population-growth diffusions (logistic, Gompertz, Verhulst, Kimura).
//! Each module's own header states its concrete `a(t, X_t)` / `b(t, X_t)`.
//!
pub mod ait_sahalia;
pub mod cev;
pub mod cfou;
pub mod cir;
pub mod ckls;
pub mod fcir;
pub mod feller;
pub mod feller_root;
pub mod fgbm;
pub mod fjacobi;
pub mod fou;
pub mod fouque;
pub mod gbm;
pub mod gbm_ih;
pub mod gbm_log;
pub mod gompertz;
pub mod hyperbolic;
pub mod hyperbolic2;
pub mod jacobi;
pub mod kimura;
pub mod linear_sde;
pub mod logistic;
pub mod modified_cir;
pub mod nonlinear_sde;
pub mod ou;
pub mod pearson;
pub mod quadratic;
pub mod radial_ou;
pub mod regime_switching;
pub mod three_half;
pub mod verhulst;
