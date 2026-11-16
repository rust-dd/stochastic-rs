//! Two-factor Gaussian G2++ trinomial-tree engine.

use ndarray::Array1;
use ndarray::Array2;

use super::TwoFactorShortRateModel;
use super::common::OrnsteinUhlenbeckFactor;
use super::common::build_one_factor_trinomial_tree;
use super::common::correlated_joint_probabilities;
use crate::lattice::tree::TrinomialTree;
use crate::traits::RealExt;

/// Two-factor Gaussian G2++ tree model.
#[derive(Debug, Clone)]
pub struct G2ppTreeModel<T: RealExt> {
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

impl<T: RealExt> G2ppTreeModel<T> {
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

impl<T: RealExt> TwoFactorShortRateModel<T> for G2ppTreeModel<T> {
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

/// G2++ two-factor tree: OU factor trees coupled per step by Brigo–Mercurio's `ρ/36` pattern, which
/// keeps both marginals and realises ρ, scaled down where a drifting branch leaves too little mass.
#[derive(Debug, Clone)]
pub struct G2ppTree<T: RealExt> {
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

impl<T: RealExt> G2ppTree<T> {
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

#[cfg(test)]
mod tests {
  use ndarray::Axis;

  use super::*;

  /// `Corr(x_T, y_T)` of the tree: the root's unit mass carried forward through the joint tables.
  fn terminal_correlation(a: f64, b: f64, rho: f64, t: f64, steps: usize) -> f64 {
    let tree = G2ppTree::new(
      G2ppTreeModel::new(0.0, 0.0, 0.0, a, b, 1.0, 1.0, rho),
      t,
      steps,
    );
    let mut mass = Array2::from_elem((1, 1), 1.0);
    for level in 0..steps {
      let mut next = Array2::zeros((2 * level + 3, 2 * level + 3));
      for ((ix, iy), &m) in mass.indexed_iter() {
        let (bx, by) = (
          tree.x_tree.branches[level][ix],
          tree.y_tree.branches[level][iy],
        );
        let joint = correlated_joint_probabilities(bx, by, rho);
        for (u, row) in joint.iter().enumerate() {
          for (v, p) in row.iter().enumerate() {
            next[[bx.center_index + u - 1, by.center_index + v - 1]] += m * p;
          }
        }
      }
      mass = next;
    }
    let (x, y) = (&tree.x_tree.states[steps], &tree.y_tree.states[steps]);
    let (px, py) = (mass.sum_axis(Axis(1)), mass.sum_axis(Axis(0)));
    let (mx, my) = (px.dot(x), py.dot(y));
    let (dx, dy) = (x - mx, y - my);
    let cov = dx.dot(&mass.dot(&dy));
    cov / (px.dot(&(&dx * &dx)) * py.dot(&(&dy * &dy))).sqrt()
  }

  /// Auer (2009)'s `Corr_{x,y}(t)`: within 0.05 for `|ρ| < 1` and converging where `a ≠ b`.
  #[test]
  fn the_terminal_factor_correlation_converges_to_auers() {
    for (a, b, rho, t, auer) in [
      (0.5, 0.05, -0.7, 3.0, -0.655_250_231_639_761_1),
      (0.05, 0.05, -0.9, 1.0, -0.9),
      (0.5, 0.05, -0.7, 1.0, -0.694_215_295_961_855_3),
      (0.5, 0.05, -1.0, 3.0, -0.936_071_759_485_373),
    ] {
      let coarse = terminal_correlation(a, b, rho, t, 30);
      let fine = terminal_correlation(a, b, rho, t, 120);
      if rho.abs() < 1.0 {
        assert!(
          (fine - auer).abs() < 0.05,
          "a {a} b {b} t {t}: {fine} vs {auer}"
        );
      }
      if a != b {
        assert!(
          (fine - auer).abs() < (coarse - auer).abs(),
          "{coarse} -> {fine} vs {auer}"
        );
      }
    }
  }
}
