//! Black-Karasinski log-short-rate trinomial-tree engine.

use super::OneFactorShortRateModel;
use super::common::OrnsteinUhlenbeckFactor;
use super::common::build_one_factor_trinomial_tree;
use super::common::price_one_factor_zcb;
use super::common::require_finite;
use super::common::require_non_negative;
use super::common::require_positive;
use crate::calibration::tree_swaption::BlackKarasinskiParams;
use crate::instruments::option::BermudanSwaption;
use crate::instruments::option::BermudanSwaptionValuation;
use crate::instruments::option::BermudanTree;
use crate::instruments::option::bermudan::one_factor_valuation;
use crate::lattice::tree::TrinomialTree;
use crate::traits::RealExt;

/// Black-Karasinski log-short-rate tree model.
#[derive(Debug, Clone)]
pub struct BlackKarasinskiTreeModel<T: RealExt> {
  /// Initial short rate.
  pub initial_rate: T,
  /// Mean reversion speed of the log-rate.
  pub mean_reversion: T,
  /// Long-run level of the log-rate.
  pub theta_log: T,
  /// Volatility of the log-rate.
  pub sigma: T,
}

impl<T: RealExt> BlackKarasinskiTreeModel<T> {
  /// `theta_log` is the log of `params.long_run_rate`; panics unless both rates are finite and
  /// positive, and `mean_reversion` and `sigma` finite and `≥ 0`.
  pub fn new(params: BlackKarasinskiParams) -> Self {
    require_positive("long_run_rate", params.long_run_rate);
    let model = Self {
      initial_rate: T::from_f64_fast(params.initial_rate),
      mean_reversion: T::from_f64_fast(params.mean_reversion),
      theta_log: T::from_f64_fast(params.long_run_rate).ln(),
      sigma: T::from_f64_fast(params.sigma),
    };
    model.check();
    model
  }

  fn check(&self) {
    require_non_negative("mean_reversion", self.mean_reversion);
    require_non_negative("sigma", self.sigma);
    require_positive("initial_rate", self.initial_rate);
    require_finite("theta_log", self.theta_log);
  }
}

impl<T: RealExt> OneFactorShortRateModel<T> for BlackKarasinskiTreeModel<T> {
  fn initial_state(&self) -> T {
    self.initial_rate.ln()
  }

  fn drift(&self, _time: T, state: T) -> T {
    self.mean_reversion * (self.theta_log - state)
  }

  fn diffusion(&self, _time: T, _state: T) -> T {
    self.sigma
  }

  fn short_rate(&self, _time: T, state: T) -> T {
    state.exp()
  }
}

/// Black-Karasinski tree engine.
#[derive(Debug, Clone)]
pub struct BlackKarasinskiTree<T: RealExt> {
  /// Underlying model.
  pub model: BlackKarasinskiTreeModel<T>,
  /// Trinomial lattice.
  pub tree: TrinomialTree<T>,
  /// Maturity horizon.
  pub horizon: T,
}

impl<T: RealExt> BlackKarasinskiTree<T> {
  /// `steps` levels up to `horizon`; panics unless `horizon` is finite and positive, `steps > 0`
  /// and the model passes [`BlackKarasinskiTreeModel::new`]'s checks, which a struct literal skips.
  pub fn new(model: BlackKarasinskiTreeModel<T>, horizon: T, steps: usize) -> Self {
    model.check();
    let factor = OrnsteinUhlenbeckFactor {
      initial_state: model.initial_rate.ln(),
      long_run_mean: model.theta_log,
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

impl<T: RealExt> BermudanTree<T> for BlackKarasinskiTree<T> {
  fn bermudan_valuation(&self, swaption: &BermudanSwaption<T>) -> BermudanSwaptionValuation<T> {
    one_factor_valuation(swaption, &self.tree, &self.model)
  }
}
