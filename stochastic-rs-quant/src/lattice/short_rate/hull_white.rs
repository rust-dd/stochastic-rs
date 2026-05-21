//! One-factor Hull-White / Vasicek trinomial-tree engine.

use super::common::build_one_factor_trinomial_tree;
use super::common::price_one_factor_zcb;
use super::OneFactorShortRateModel;
use crate::lattice::tree::TrinomialTree;
use crate::traits::FloatExt;

/// One-factor Hull-White / Vasicek tree model.
#[derive(Debug, Clone)]
pub struct HullWhiteTreeModel<T: FloatExt> {
  /// Initial short rate.
  pub initial_rate: T,
  /// Mean reversion speed.
  pub mean_reversion: T,
  /// Long-run level in the short-rate SDE.
  pub theta: T,
  /// Volatility.
  pub sigma: T,
}

impl<T: FloatExt> HullWhiteTreeModel<T> {
  /// Construct a Hull-White tree model.
  pub fn new(initial_rate: T, mean_reversion: T, theta: T, sigma: T) -> Self {
    Self {
      initial_rate,
      mean_reversion,
      theta,
      sigma,
    }
  }
}

impl<T: FloatExt> OneFactorShortRateModel<T> for HullWhiteTreeModel<T> {
  fn initial_state(&self) -> T {
    self.initial_rate
  }

  fn drift(&self, _time: T, state: T) -> T {
    self.mean_reversion * (self.theta - state)
  }

  fn diffusion(&self, _time: T, _state: T) -> T {
    self.sigma
  }

  fn short_rate(&self, _time: T, state: T) -> T {
    state
  }
}

/// Hull-White tree engine.
#[derive(Debug, Clone)]
pub struct HullWhiteTree<T: FloatExt> {
  /// Underlying model.
  pub model: HullWhiteTreeModel<T>,
  /// Trinomial lattice.
  pub tree: TrinomialTree<T>,
  /// Maturity horizon.
  pub horizon: T,
}

impl<T: FloatExt> HullWhiteTree<T> {
  /// Build a Hull-White tree with `steps` time steps up to `horizon`.
  pub fn new(model: HullWhiteTreeModel<T>, horizon: T, steps: usize) -> Self {
    let tree = build_one_factor_trinomial_tree(&model, horizon, steps);
    Self {
      model,
      tree,
      horizon,
    }
  }

  /// Price a zero-coupon bond maturing at the tree horizon.
  pub fn zero_coupon_bond_price(&self) -> T {
    price_one_factor_zcb(&self.tree, &self.model)
  }
}
