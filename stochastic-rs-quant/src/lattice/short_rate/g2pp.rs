//! Two-factor Gaussian G2++ trinomial-tree engine.

use ndarray::Array1;
use ndarray::Array2;

use super::common::build_one_factor_trinomial_tree;
use super::common::correlated_joint_probabilities;
use super::common::OrnsteinUhlenbeckFactor;
use super::TwoFactorShortRateModel;
use crate::lattice::tree::TrinomialTree;
use crate::traits::FloatExt;

/// Two-factor Gaussian G2++ tree model.
#[derive(Debug, Clone)]
pub struct G2ppTreeModel<T: FloatExt> {
  /// Initial first factor.
  pub initial_x: T,
  /// Initial second factor.
  pub initial_y: T,
  /// Constant deterministic shift.
  pub phi: T,
  /// Mean reversion of the first factor.
  pub mean_reversion_x: T,
  /// Mean reversion of the second factor.
  pub mean_reversion_y: T,
  /// Volatility of the first factor.
  pub sigma_x: T,
  /// Volatility of the second factor.
  pub sigma_y: T,
  /// Correlation between the two Gaussian factors.
  pub rho: T,
}

impl<T: FloatExt> G2ppTreeModel<T> {
  /// Construct a G2++ tree model.
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    initial_x: T,
    initial_y: T,
    phi: T,
    mean_reversion_x: T,
    mean_reversion_y: T,
    sigma_x: T,
    sigma_y: T,
    rho: T,
  ) -> Self {
    Self {
      initial_x,
      initial_y,
      phi,
      mean_reversion_x,
      mean_reversion_y,
      sigma_x,
      sigma_y,
      rho,
    }
  }
}

impl<T: FloatExt> TwoFactorShortRateModel<T> for G2ppTreeModel<T> {
  fn initial_x(&self) -> T {
    self.initial_x
  }

  fn initial_y(&self) -> T {
    self.initial_y
  }

  fn drift_x(&self, _time: T, x: T) -> T {
    -self.mean_reversion_x * x
  }

  fn drift_y(&self, _time: T, y: T) -> T {
    -self.mean_reversion_y * y
  }

  fn diffusion_x(&self, _time: T, _x: T) -> T {
    self.sigma_x
  }

  fn diffusion_y(&self, _time: T, _y: T) -> T {
    self.sigma_y
  }

  fn correlation(&self) -> T {
    self.rho
  }

  fn short_rate(&self, _time: T, x: T, y: T) -> T {
    x + y + self.phi
  }
}

/// G2++ two-factor tree engine.
///
/// The factor trees are built exactly as independent Gaussian Ou trees and
/// coupled at each step by a moment-matched 3x3 joint transition correction
/// that preserves the one-factor marginals and injects the requested
/// instantaneous correlation. This is sufficient as a Tier 0 foundation for
/// bond-style backward induction and can be refined later for calibration-grade
/// swaption work.
#[derive(Debug, Clone)]
pub struct G2ppTree<T: FloatExt> {
  /// Underlying model.
  pub model: G2ppTreeModel<T>,
  /// First-factor lattice.
  pub x_tree: TrinomialTree<T>,
  /// Second-factor lattice.
  pub y_tree: TrinomialTree<T>,
  /// Maturity horizon.
  pub horizon: T,
  /// Time step.
  pub dt: T,
}

impl<T: FloatExt> G2ppTree<T> {
  /// Build a G2++ tree with `steps` time steps up to `horizon`.
  pub fn new(model: G2ppTreeModel<T>, horizon: T, steps: usize) -> Self {
    let x_model = OrnsteinUhlenbeckFactor {
      initial_state: model.initial_x,
      mean_reversion: model.mean_reversion_x,
      sigma: model.sigma_x,
    };
    let y_model = OrnsteinUhlenbeckFactor {
      initial_state: model.initial_y,
      mean_reversion: model.mean_reversion_y,
      sigma: model.sigma_y,
    };
    let x_tree = build_one_factor_trinomial_tree(&x_model, horizon, steps);
    let y_tree = build_one_factor_trinomial_tree(&y_model, horizon, steps);
    Self {
      model,
      dt: horizon / T::from_usize_(steps),
      x_tree,
      y_tree,
      horizon,
    }
  }

  /// Price a zero-coupon bond maturing at the tree horizon.
  pub fn zero_coupon_bond_price(&self) -> T {
    let mut values = Array2::from_elem(
      (
        self.x_tree.states.last().map_or(0, Array1::len),
        self.y_tree.states.last().map_or(0, Array1::len),
      ),
      T::one(),
    );

    for level in (0..self.x_tree.branches.len()).rev() {
      let x_width = self.x_tree.states[level].len();
      let y_width = self.y_tree.states[level].len();
      let mut step_values = Array2::zeros((x_width, y_width));
      let time = T::from_usize_(level) * self.dt;

      for ix in 0..x_width {
        let x_branch = self.x_tree.branches[level][ix];
        let x_children = [
          x_branch.center_index - 1,
          x_branch.center_index,
          x_branch.center_index + 1,
        ];

        for iy in 0..y_width {
          let y_branch = self.y_tree.branches[level][iy];
          let y_children = [
            y_branch.center_index - 1,
            y_branch.center_index,
            y_branch.center_index + 1,
          ];
          let joint = correlated_joint_probabilities(x_branch, y_branch, self.model.correlation());

          let mut expected = T::zero();
          for ax in 0..3 {
            for ay in 0..3 {
              expected += joint[ax][ay] * values[[x_children[ax], y_children[ay]]];
            }
          }

          let rate = self.model.short_rate(
            time,
            self.x_tree.states[level][ix],
            self.y_tree.states[level][iy],
          );
          step_values[[ix, iy]] = (-rate * self.dt).exp() * expected;
        }
      }

      values = step_values;
    }

    values[[0, 0]]
  }
}
