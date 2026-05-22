//! Short-rate tree engines built on the generic lattice primitives.
//!
//! Reference: Hull & White (1994), Black & Karasinski (1991).

use crate::traits::FloatExt;

mod black_karasinski;
mod common;
mod curve_fitted_hw;
mod g2pp;
mod hull_white;

pub use black_karasinski::BlackKarasinskiTree;
pub use black_karasinski::BlackKarasinskiTreeModel;
pub(crate) use common::correlated_joint_probabilities;
pub use curve_fitted_hw::CurveFittedHullWhiteModel;
pub use curve_fitted_hw::CurveFittedHullWhiteTree;
pub use g2pp::G2ppTree;
pub use g2pp::G2ppTreeModel;
pub use hull_white::HullWhiteTree;
pub use hull_white::HullWhiteTreeModel;

/// Extensibility point for one-factor short-rate lattice models.
pub trait OneFactorShortRateModel<T: FloatExt>: Clone + Send + Sync {
  /// Initial state variable.
  fn initial_state(&self) -> T;

  /// Drift of the state process.
  fn drift(&self, time: T, state: T) -> T;

  /// Instantaneous diffusion of the state process.
  fn diffusion(&self, time: T, state: T) -> T;

  /// Short rate implied by the state variable.
  fn short_rate(&self, time: T, state: T) -> T;
}

/// Extensibility point for two-factor short-rate lattice models.
pub trait TwoFactorShortRateModel<T: FloatExt>: Clone + Send + Sync {
  /// Initial first factor.
  fn initial_x(&self) -> T;

  /// Initial second factor.
  fn initial_y(&self) -> T;

  /// Drift of the first factor.
  fn drift_x(&self, time: T, x: T) -> T;

  /// Drift of the second factor.
  fn drift_y(&self, time: T, y: T) -> T;

  /// Instantaneous diffusion of the first factor.
  fn diffusion_x(&self, time: T, x: T) -> T;

  /// Instantaneous diffusion of the second factor.
  fn diffusion_y(&self, time: T, y: T) -> T;

  /// Instantaneous factor correlation.
  fn correlation(&self) -> T;

  /// Short rate implied by the factor pair.
  fn short_rate(&self, time: T, x: T, y: T) -> T;
}
