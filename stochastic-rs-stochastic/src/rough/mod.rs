//! Riemann–Liouville fBM and its Volterra SDEs by Bilokon & Wong's (2026) Markov lift.
//!
//! $$
//! X_t = X_0 + \frac{1}{\Gamma(H+1/2)}\int_0^t (t-s)^{H-1/2} f(X_s)\,ds
//!          + \frac{1}{\Gamma(H+1/2)}\int_0^t (t-s)^{H-1/2} g(X_s)\,dW_s
//! $$
//!
//! Gauss–Laguerre where Ma & Wu (2021) use Gauss–Legendre: `N' ≈ log N` factors, no stored history.
//!
//! $$
//! \Gamma(1/2-H)\, t^{H-1/2} \approx \sum_{l=1}^{N'} w_l\, e^{-x_l t},
//! $$
//!
//! Reference: Bilokon & Wong (2026), J. Appl. Probab., doi:10.1017/jpr.2025.10071
//! Reference: Ma & Wu (2021), SIAM Review 22, 447–462
//! Reference: Abi Jaber & El Euch (2018), arXiv:1801.10359
pub mod kernel;
mod markov_lift;
pub mod rl_bs;
pub mod rl_fbm;
pub mod rl_fou;
pub mod rl_heston;

pub use kernel::RlKernel;
pub use markov_lift::MarkovLift;
pub use rl_bs::RlBlackScholes;
pub use rl_fbm::RlFbm;
pub use rl_fou::RlFou;
pub use rl_heston::RlHeston;

/// [`MarkovLift`] has one path, `rough::MarkovLift`.
///
/// ```compile_fail,E0603
/// use stochastic_rs_stochastic::rough::markov_lift::MarkovLift;
/// ```
///
/// ```compile_fail,E0432
/// use stochastic_rs_stochastic::numerics::MarkovLift;
/// ```
#[cfg(doctest)]
pub struct MarkovLiftHasOnePath;
