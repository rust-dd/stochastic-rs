//! Malliavin–Thalmaier multi-dimensional Greeks.
//!
//! $$
//! \frac{\partial}{\partial\mu_k}\mathbb E[f(F^\mu)]
//! =\sum_{i,j}^d\mathbb E\!\bigl[g^h_{i,j}(F^\mu)\,H_{(i)}\!\bigl(F^\mu;\tfrac{\partial F^\mu_j}{\partial\mu_k}\bigr)\bigr]
//! $$
//!
//! The classical Malliavin integration-by-parts formula (Nualart, Prop. 2.1.5)
//! uses *d* nested Skorohod integrals for a *d*-dimensional density, which is
//! exponentially expensive and numerically unstable. Malliavin and Thalmaier
//! (2006, Theorem 4.23) replace the nested integrals with the **Poisson kernel**
//! `Q_d`, the fundamental solution of the Laplace equation, reducing it to a
//! **single** Skorohod integral regardless of dimension.
//!
//! The Poisson kernel `Q_d`:
//! - `d = 2`: `Q₂(x) = (1/2π) ln|x|`
//! - `d ≥ 3`: `Q_d(x) = −1 / (a_d · |x|^{d−2})`
//!
//! where `a_d = 2π^{d/2} / Γ(d/2)` is the surface area of the unit sphere.
//!
//! The raw formula has infinite variance at `h = 0` (Kohatsu-Higa & Yasuda,
//! 2008). Replacing `|x|` with `|x|_h = √(Σx² + h)` gives controllable
//! bias `O(h ln 1/h)` and variance `O(ln 1/h)` for `d = 2`.
//!
//! Works for any Itô diffusion satisfying the Hörmander condition: Heston,
//! SABR, 3/2, Stein–Stein, rough Bergomi, CEV, Bates, etc.
//!
//! # References
//!
//! - Malliavin, P. & Thalmaier, A. (2006). *Stochastic Calculus of Variations
//!   in Mathematical Finance*. Springer. Theorem 4.23.
//! - Kohatsu-Higa, A. & Yasuda, K. (2008). RIMS Kôkyûroku 1580, 124–135.
//! - Bally, V. & Caramellino, L. *Lower bounds for the density of Itô
//!   processes under weak regularity assumptions*. Preprint.

pub mod engine;
pub mod heston;
pub mod kernel;
pub mod traits;

pub use engine::MtGreeks;
pub use engine::MtPayoff;
pub use heston::AssetParams;
pub use heston::MultiHestonParams;
pub use heston::MultiHestonPaths;
pub use kernel::g_digital_put_2d;
pub use traits::MultiSvPaths;
