//! # Rough (Riemann–Liouville) Volterra processes
//!
//! $$
//! X_t = X_0 + \frac{1}{\Gamma(H+1/2)}\int_0^t (t-s)^{H-1/2} f(X_s)\,ds
//!          + \frac{1}{\Gamma(H+1/2)}\int_0^t (t-s)^{H-1/2} g(X_s)\,dW_s
//! $$
//!
//! Riemann–Liouville fBM and its Volterra-type driven SDEs, simulated via the
//! modified fast algorithm of Bilokon & Wong (2026), which replaces the
//! Gauss–Legendre quadrature of Ma & Wu (2021) with a generalised Gauss–Laguerre
//! quadrature. The power-law kernel is approximated by an exponential sum
//!
//! $$
//! \Gamma(1/2-H)\, t^{H-1/2} \approx \sum_{l=1}^{N'} w_l\, e^{-x_l t},
//! $$
//!
//! which turns every Volterra-type SDE into a superposition of $N' \approx \log N$
//! independent Ornstein–Uhlenbeck-like Markov factors. No cumulative Euler,
//! no full-history storage — the whole path is built from a bounded state.
//!
//! # References
//! - Bilokon P. A., Wong Y. C. C. *Efficient Simulation of Fractional Brownian Motion*,
//!   J. Appl. Probab. (2026), doi:10.1017/jpr.2025.10071.
//! - Ma J., Wu H. *A Fast Algorithm for Simulation of Rough Volatility Models*,
//!   SIAM Review 22 (2021), 447–462.
//! - Abi Jaber E., El Euch O. *Multi-factor approximation of rough volatility models*,
//!   arXiv:1801.10359 (2018).
pub mod kernel;
pub mod markov_lift;
pub mod rl_bs;
pub mod rl_fbm;
pub mod rl_fou;
pub mod rl_heston;

pub use kernel::RlKernel;
pub use markov_lift::{MarkovLift, RoughSimd};
pub use rl_bs::RlBlackScholes;
pub use rl_fbm::RlFBm;
pub use rl_fou::RlFOU;
pub use rl_heston::RlHeston;
