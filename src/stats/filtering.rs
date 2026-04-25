//! # Bayesian Filtering & Sampling
//!
//! Sequential state-space inference (bootstrap particle filter, Unscented
//! Kalman Filter) and posterior sampling (Random-Walk Metropolis-Hastings).
//!
//! $$
//! x_t = f(x_{t-1}, v_t),\qquad y_t = h(x_t, w_t),
//! $$
//! with $v_t, w_t$ independent noise.
//!
//! Most algorithms require the `openblas` feature for the linear-algebra
//! routines on the state-covariance matrix.
//!
//! # References
//! - Gordon, Salmond, Smith, "Novel Approach to Nonlinear/Non-Gaussian
//!   Bayesian State Estimation", IEE Proceedings F, 140(2), 107-113 (1993).
//!   DOI: 10.1049/ip-f-2.1993.0015
//! - Julier, Uhlmann, "Unscented Filtering and Nonlinear Estimation",
//!   Proceedings of the IEEE, 92(3), 401-422 (2004).
//!   DOI: 10.1109/JPROC.2003.823141
//! - Wan, van der Merwe, "The Unscented Kalman Filter for Nonlinear
//!   Estimation", IEEE Adaptive Systems for Signal Processing,
//!   Communications, and Control Symposium 2000, 153-158.
//!   DOI: 10.1109/ASSPCC.2000.882463
//! - Metropolis, Rosenbluth, Rosenbluth, Teller, Teller, "Equation of State
//!   Calculations by Fast Computing Machines", Journal of Chemical Physics,
//!   21(6), 1087-1092 (1953). DOI: 10.1063/1.1699114
//! - Hastings, "Monte Carlo Sampling Methods Using Markov Chains and Their
//!   Applications", Biometrika, 57(1), 97-109 (1970). DOI: 10.1093/biomet/57.1.97

pub mod mcmc;
pub mod particle;
#[cfg(feature = "openblas")]
pub mod ukf;

pub use mcmc::MhResult;
pub use mcmc::random_walk_metropolis;
pub use particle::ParticleFilter;
pub use particle::ResamplingScheme;
#[cfg(feature = "openblas")]
pub use ukf::UkfState;
#[cfg(feature = "openblas")]
pub use ukf::unscented_kalman_step;
