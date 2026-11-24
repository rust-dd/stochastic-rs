//! Malliavin–Thalmaier multi-dimensional Greeks: the Poisson kernel `Q_d` turns `d` nested Skorohod
//! integrals into one, whatever the dimension.
//!
//! $$
//! \frac{\partial}{\partial\mu_k}\mathbb E[f(F^\mu)]
//! =\sum_{i,j}^d\mathbb E\!\bigl[g^h_{i,j}(F^\mu)\,H_{(i)}\!\bigl(F^\mu;\tfrac{\partial F^\mu_j}{\partial\mu_k}\bigr)\bigr]
//! $$
//!
//! $$
//! Q_2(x) = \tfrac{1}{2\pi}\ln|x|,\qquad Q_d(x) = -\frac{|x|^{2-d}}{(d-2)\,a_d},\qquad
//! a_d = \frac{2\pi^{d/2}}{\Gamma(d/2)}
//! $$
//!
//! Reference: Malliavin & Thalmaier (2006), Springer, Theorem 4.23
//! Reference: Kohatsu-Higa & Yasuda (2008), RIMS Kôkyûroku 1580, 124–135
//! Reference: Nualart, The Malliavin Calculus and Related Topics, Prop. 2.1.5
//! Reference: Bally & Caramellino (2013), Potential Anal. 39(2), DOI 10.1007/s11118-012-9324-7
//! Source: <https://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1580-11.pdf>

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
pub use kernel::g_kernel_numerical_nd;
pub use traits::MultiSvPaths;
