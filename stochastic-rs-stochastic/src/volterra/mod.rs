//! Stochastic Volterra equations, stepped at `O(nN')` through an exponential-sum kernel.
//!
//! $$
//! X_t = X_0 + \int_0^t K(t-s)\,b(s,X_s)\,ds + \int_0^t K(t-s)\,\sigma(s,X_s)\,dW_s
//! $$
//!
//! $X$ is neither Markov nor a semimartingale; [`VolterraLift`] steps [`VolterraKernel`]'s factors.
//!
//! $$
//! K(t) \approx \sum_{l=1}^{N'} w_l\, e^{-x_l t},
//! $$
//!
//! $$
//! r_{\theta\text{-EM}} = \min\{1-\alpha,\ \tfrac12-\beta\},\qquad
//! r_{\mathrm{Milstein}} = \min\{1-\alpha,\ 1-2\beta\}
//! $$
//!
//! Reference: Abi Jaber & El Euch (2018), arXiv:1801.10359
//! Reference: Alfonsi & Kebaier (2021), arXiv:2102.13505
//! Reference: Li, Huang & Hu (2020), arXiv:2004.04916, the strong rates of θ-EM and Milstein
//! Reference: Liu, Hu & Gao (2024), arXiv:2412.11126, the Milstein rate $n^{-2H}$ is optimal
//! Reference: Bayer & Breneis (2021), arXiv:2108.05048, superpolynomial strong convergence in $N'$
//! Reference: Bayer & Breneis (2023), arXiv:2309.07023, the weak error follows the $L^1$ error

pub mod fit;
mod gaussian;
pub mod gaussian_polynomial;
pub mod kernel;
pub mod lift;
pub mod reference;
pub mod square_root;
pub mod volterra_sde;

pub use fit::fit_l1;
pub use fit::l1_error;
#[cfg(feature = "python")]
#[doc(hidden)]
pub use gaussian::PyVolterra;
pub use gaussian::Volterra;
pub use gaussian::VolterraKernelSpec;
pub use gaussian_polynomial::GaussianPolynomialVolatility;
pub use kernel::ExponentialKernel;
pub use kernel::GammaKernel;
pub use kernel::SumOfExponentials;
pub use kernel::VolterraKernel;
pub use lift::VolterraLift;
pub use reference::reference_path;
pub use square_root::VolterraSquareRoot;
pub use volterra_sde::VolterraSde;
