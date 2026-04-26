//! # Lattice Framework
//!
//! $$
//! V_i(x)=e^{-r_i(x)\Delta t}\,\mathbb{E}^{\mathbb Q}[V_{i+1}(X_{i+1})\mid X_i=x]
//! $$
//!
//! Recombining binomial and trinomial trees for discrete-time valuation.
//!
//! Reference: Hull & White, "Numerical Procedures for Implementing Term Structure
//! Models I: Single-Factor Models", Journal of Derivatives 2(1) (1994).
//!
//! Reference: Black & Karasinski, "Bond and Option Pricing when Short Rates are
//! Lognormal", Financial Analysts Journal 47(4) (1991).
//!
//! Reference: Yamakami & Takeuchi, "Pricing Bermudan Swaption under Two Factor
//! Hull-White Model with Fast Gauss Transform", arXiv:2212.08250 (2022).

use crate::traits::FloatExt;

pub mod short_rate;
pub mod tree;

pub use short_rate::BlackKarasinskiTree;
pub use short_rate::BlackKarasinskiTreeModel;
pub use short_rate::CurveFittedHullWhiteModel;
pub use short_rate::CurveFittedHullWhiteTree;
pub use short_rate::G2ppTree;
pub use short_rate::G2ppTreeModel;
pub use short_rate::HullWhiteTree;
pub use short_rate::HullWhiteTreeModel;
pub use short_rate::OneFactorShortRateModel;
pub use short_rate::TwoFactorShortRateModel;
pub use tree::BinomialTree;
pub use tree::TrinomialBranch;
pub use tree::TrinomialTree;

/// Helper trait for one-dimensional node discounting.
pub trait NodeDiscount<T: FloatExt>: Send + Sync {
  /// Discount factor applied over one time step at a node state.
  fn node_discount(&self, dt: T, state: T) -> T;
}
