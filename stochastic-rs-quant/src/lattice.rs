//! Recombining binomial and trinomial trees for discrete-time valuation.
//!
//! $$
//! V_i(x)=e^{-r_i(x)\Delta t}\,\mathbb{E}^{\mathbb Q}[V_{i+1}(X_{i+1})\mid X_i=x]
//! $$
//!
//! Reference: Hull & White (1994), J. Deriv. 2(1), 7–16, DOI 10.3905/jod.1994.407902
//! Reference: Black & Karasinski (1991), Financ. Anal. J. 47(4), 52–59
//! Reference: Yamakami & Takeuchi (2022), arXiv:2212.08250

pub mod equity;
pub mod short_rate;
pub mod tree;

pub use equity::CrrModel;
pub use short_rate::BlackKarasinskiTree;
pub use short_rate::BlackKarasinskiTreeModel;
pub use short_rate::CallableBondPrice;
pub use short_rate::CallableBondSpec;
pub use short_rate::CurveFittedHullWhiteModel;
pub use short_rate::CurveFittedHullWhiteTree;
pub use short_rate::G2ppTree;
pub use short_rate::G2ppTreeModel;
pub use short_rate::HullWhiteTree;
pub use short_rate::HullWhiteTreeModel;
pub use short_rate::OneFactorShortRateModel;
pub use short_rate::TwoFactorShortRateModel;
pub use short_rate::price_callable_bond;
pub use tree::BinomialTree;
pub use tree::TrinomialBranch;
pub use tree::TrinomialTree;
