//! Bayesian filtering and sampling: bootstrap particle filter, unscented Kalman filter, Metropolis.
//!
//! $$
//! x_t = f(x_{t-1}, v_t),\qquad y_t = h(x_t, w_t),\qquad v_t \perp w_t
//! $$
//!
//! Reference: Gordon et al. (1993), IEE Proc. F 140(2), 107-113, DOI: 10.1049/ip-f-2.1993.0015
//! Reference: Julier & Uhlmann (2004), Proc. IEEE 92(3), 401-422, DOI: 10.1109/JPROC.2003.823141
//! Reference: Wan & van der Merwe (2000), IEEE AS-SPCC, 153-158, DOI: 10.1109/ASSPCC.2000.882463
//! Reference: Metropolis et al. (1953), J. Chem. Phys. 21(6), 1087-1092, DOI: 10.1063/1.1699114
//! Reference: Hastings (1970), Biometrika 57(1), 97-109, DOI: 10.1093/biomet/57.1.97

pub mod mcmc;
pub mod particle;
pub mod ukf;

pub use mcmc::MhResult;
pub use mcmc::random_walk_metropolis;
pub use particle::ParticleFilter;
pub use particle::ResamplingScheme;
pub use ukf::UkfState;
pub use ukf::UkfTuning;
pub use ukf::unscented_kalman_step;
