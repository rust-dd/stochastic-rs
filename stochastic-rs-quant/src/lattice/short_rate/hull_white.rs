//! One-factor Hull-White / Vasicek trinomial-tree engine.

use super::OneFactorShortRateModel;
use super::common::OrnsteinUhlenbeckFactor;
use super::common::build_one_factor_trinomial_tree;
use super::common::price_one_factor_zcb;
use super::common::require_finite;
use super::common::require_non_negative;
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
  /// Panics unless `mean_reversion` and `sigma` are finite and `≥ 0` and both rates are finite.
  pub fn new(params: HullWhiteParams) -> Self {
    let model = Self {
      initial_rate: T::from_f64_fast(params.initial_rate),
      mean_reversion: T::from_f64_fast(params.mean_reversion),
      theta: T::from_f64_fast(params.theta),
      sigma: T::from_f64_fast(params.sigma),
    };
    model.check();
    model
  }

  fn check(&self) {
    require_non_negative("mean_reversion", self.mean_reversion);
    require_non_negative("sigma", self.sigma);
    require_finite("initial_rate", self.initial_rate);
    require_finite("theta", self.theta);
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
  /// `steps` levels up to `horizon`; panics unless `horizon` is finite and positive, `steps > 0`
  /// and the model passes the checks of [`HullWhiteTreeModel::new`], which a struct literal skips.
  pub fn new(model: HullWhiteTreeModel<T>, horizon: T, steps: usize) -> Self {
    model.check();
    let factor = OrnsteinUhlenbeckFactor {
      initial_state: model.initial_rate,
      long_run_mean: model.theta,
      mean_reversion: model.mean_reversion,
      sigma: model.sigma,
    };
    let tree = build_one_factor_trinomial_tree(&factor, horizon, steps);
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
