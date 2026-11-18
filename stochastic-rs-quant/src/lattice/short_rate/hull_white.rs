//! One-factor Hull-White / Vasicek trinomial-tree engine.

use super::OneFactorShortRateModel;
use super::common::build_one_factor_trinomial_tree;
use super::common::price_one_factor_zcb;
use crate::calibration::hw_swaption::HullWhiteParams;
use crate::instruments::option::BermudanSwaption;
use crate::instruments::option::BermudanSwaptionValuation;
use crate::instruments::option::BermudanTree;
use crate::instruments::option::bermudan::one_factor_valuation;
use crate::lattice::tree::TrinomialTree;
use crate::traits::RealExt;

/// One-factor Hull-White / Vasicek tree model.
#[derive(Debug, Clone)]
pub struct HullWhiteTreeModel<T: RealExt> {
  /// Initial short rate.
  pub initial_rate: T,
  /// Mean reversion speed.
  pub mean_reversion: T,
  /// Long-run level in the short-rate SDE.
  pub theta: T,
  /// Absolute (normal) volatility σ of the short rate.
  pub sigma: T,
}

impl<T: RealExt> HullWhiteTreeModel<T> {
  pub fn new(params: HullWhiteParams) -> Self {
    Self {
      initial_rate: T::from_f64_fast(params.initial_rate),
      mean_reversion: T::from_f64_fast(params.mean_reversion),
      theta: T::from_f64_fast(params.theta),
      sigma: T::from_f64_fast(params.sigma),
    }
  }
}

impl<T: RealExt> OneFactorShortRateModel<T> for HullWhiteTreeModel<T> {
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
pub struct HullWhiteTree<T: RealExt> {
  /// Underlying model.
  pub model: HullWhiteTreeModel<T>,
  /// Trinomial lattice.
  pub tree: TrinomialTree<T>,
  /// Maturity horizon.
  pub horizon: T,
}

impl<T: RealExt> HullWhiteTree<T> {
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

impl<T: RealExt> BermudanTree<T> for HullWhiteTree<T> {
  fn bermudan_valuation(&self, swaption: &BermudanSwaption<T>) -> BermudanSwaptionValuation<T> {
    one_factor_valuation(swaption, &self.tree, &self.model)
  }
}
