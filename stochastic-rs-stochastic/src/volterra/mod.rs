//! # Kernel abstraction for stochastic Volterra equations
//!
//! A stochastic Volterra equation (SVE)
//!
//! $$
//! X_t = X_0 + \int_0^t K(t-s)\,b(s,X_s)\,ds + \int_0^t K(t-s)\,\sigma(s,X_s)\,dW_s
//! $$
//!
//! is driven by a kernel $K$ that, for the weakly singular kernels used by
//! rough-volatility models, makes $X$ neither Markov nor a semimartingale:
//! every step needs the whole path history, so naive simulation costs
//! $O(n^2)$ on an $n$-point grid. The standard remedy approximates $K$ by a
//! finite sum of exponentials
//!
//! $$
//! K(t) \approx \sum_{l=1}^{N'} w_l\, e^{-x_l t},
//! $$
//!
//! which turns the SVE into a superposition of $N'$ coupled
//! Ornstein–Uhlenbeck-like factors, each updatable from its own previous
//! value alone — reducing cost to $O(n N')$ with $N' \ll n$.
//!
//! [`VolterraKernel`] is the trait every such exponential-sum kernel
//! implements: nodes and weights for the sum itself, plus the two
//! closed-form boundary quantities — [`VolterraKernel::evaluate`] and
//! [`VolterraKernel::integral_from_zero`] — that a Markov-lift-style
//! stepper needs at the current step, before any history has accumulated.
//! [`crate::rough::kernel::RlKernel`], the Riemann–Liouville kernel behind
//! [`crate::rough`]'s rough-volatility processes, implements it too (see
//! that module for the `impl` block), so [`VolterraLift`] — the
//! Markov-lift-style stepper built on top of this trait — drives both the
//! fractional and the exponential-family kernels through one interface.
//! [`crate::rough::markov_lift::MarkovLift`] is `VolterraLift<T,
//! RlKernel<T>>` kept as a thin, backward-compatible wrapper.
//!
//! [`VolterraSde`] promotes [`VolterraLift`] to a first-class
//! [`ProcessExt`](crate::traits::ProcessExt): general $b$/$\sigma$
//! coefficients over any [`VolterraKernel`], not just the $(x)$-only,
//! zero-drift special cases [`MarkovLift`](crate::rough::markov_lift::MarkovLift)'s
//! own callers use. [`reference::reference_path`] is its permanent $O(n^2)$
//! cross-implementation oracle — see that module's docs for why it is kept,
//! not scaffolding.
//!
//! # References
//! - Abi Jaber E., El Euch O. *Multi-factor approximation of rough
//!   volatility models*, arXiv:1801.10359 (2018).
//! - Alfonsi A., Kebaier A. *Approximation of Stochastic Volterra Equations
//!   with kernels of completely monotone type*, arXiv:2102.13505 (2021).
//! - Li M., Huang C., Hu Y. *Numerical methods for stochastic Volterra
//!   integral equations with weakly singular kernels*, arXiv:2004.04916
//!   (2020).
pub mod kernel;
pub mod lift;
pub mod reference;
pub mod sve;

pub use kernel::ExponentialKernel;
pub use kernel::GammaKernel;
pub use kernel::SumOfExponentials;
pub use kernel::VolterraKernel;
pub use lift::VolterraLift;
pub use reference::reference_path;
pub use sve::VolterraSde;
