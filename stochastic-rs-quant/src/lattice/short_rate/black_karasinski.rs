//! Black-Karasinski log-short-rate trinomial-tree engine.

use super::OneFactorShortRateModel;
use super::common::build_one_factor_trinomial_tree;
use super::common::price_one_factor_zcb;
use crate::lattice::tree::TrinomialTree;
use crate::traits::FloatExt;

/// Black-Karasinski log-short-rate tree model.
#[derive(Debug, Clone)]
pub struct BlackKarasinskiTreeModel<T: FloatExt> {
  /// Initial short rate.
  pub initial_rate: T,
  /// Mean reversion speed of the log-rate.
  pub mean_reversion: T,
  /// Long-run level of the log-rate.
  pub theta_log: T,
  /// Volatility of the log-rate.
  pub sigma: T,
}

impl<T: FloatExt> BlackKarasinskiTreeModel<T> {
  /// Construct a Black-Karasinski tree model from rate levels.
  pub fn new(initial_rate: T, mean_reversion: T, long_run_rate: T, sigma: T) -> Self {
    Self {
      initial_rate,
      mean_reversion,
      theta_log: long_run_rate.ln(),
      sigma,
    }
  }
}

impl<T: FloatExt> OneFactorShortRateModel<T> for BlackKarasinskiTreeModel<T> {
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
pub struct BlackKarasinskiTree<T: FloatExt> {
  /// Underlying model.
  pub model: BlackKarasinskiTreeModel<T>,
  /// Trinomial lattice.
  pub tree: TrinomialTree<T>,
  /// Maturity horizon.
  pub horizon: T,
}

impl<T: FloatExt> BlackKarasinskiTree<T> {
  /// Build a Black-Karasinski tree with `steps` time steps up to `horizon`.
  pub fn new(model: BlackKarasinskiTreeModel<T>, horizon: T, steps: usize) -> Self {
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
