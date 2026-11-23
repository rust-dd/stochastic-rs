//! Generic recombining binomial and trinomial trees.
//!
//! Reference: Cox et al. (1979), J. Financ. Econ. 7(3), 229–263, DOI 10.1016/0304-405X(79)90015-1

use ndarray::Array1;

use crate::traits::RealExt;

/// Recombining binomial tree.
#[derive(Debug, Clone)]
pub struct BinomialTree<T: RealExt> {
  /// Time step.
  pub dt: T,
  /// Node states for each level.
  pub states: Vec<Array1<T>>,
  /// Up probabilities for each node at each non-terminal level.
  pub up_probabilities: Vec<Array1<T>>,
}

impl<T: RealExt> BinomialTree<T> {
  /// Panics unless level `i` holds `i + 1` states and every up probability is in `[0, 1]`.
  pub fn new(dt: T, states: Vec<Array1<T>>, up_probabilities: Vec<Array1<T>>) -> Self {
    assert_eq!(
      states.len(),
      up_probabilities.len() + 1,
      "states must have one more level than up probabilities"
    );
    for (level, nodes) in states.iter().enumerate() {
      assert_eq!(
        nodes.len(),
        level + 1,
        "binomial level {level} must contain {} nodes",
        level + 1
      );
    }
    for (level, probs) in up_probabilities.iter().enumerate() {
      assert_eq!(
        probs.len(),
        level + 1,
        "probability level {level} must contain {} entries",
        level + 1
      );
      for (node, &p) in probs.iter().enumerate() {
        assert!(
          p.is_finite() && p >= T::zero() && p <= T::one(),
          "up_probabilities[{level}][{node}] must satisfy `p.is_finite() && p >= T::zero() && p <= \
           T::one()`, got up_probabilities[{level}][{node}] = {p:?}"
        );
      }
    }
    Self {
      dt,
      states,
      up_probabilities,
    }
  }

  /// Cox-Ross-Rubinstein stock-price tree; panics unless `probability` is in `[0, 1]`.
  pub fn from_crr(spot: T, up: T, down: T, probability: T, steps: usize, dt: T) -> Self {
    let states = (0..=steps)
      .map(|level| {
        Array1::from_iter((0..=level).map(|up_moves| {
          spot * up.powf(T::from_usize_(up_moves)) * down.powf(T::from_usize_(level - up_moves))
        }))
      })
      .collect::<Vec<_>>();

    let up_probabilities = (0..steps)
      .map(|level| Array1::from_elem(level + 1, probability))
      .collect::<Vec<_>>();

    Self::new(dt, states, up_probabilities)
  }

  /// Backward induction under node-specific discount factors.
  pub fn backward_induct<F>(&self, terminal_values: Array1<T>, mut discount: F) -> T
  where
    F: FnMut(usize, usize, T) -> T,
  {
    assert_eq!(
      terminal_values.len(),
      self.states.last().map_or(0, Array1::len),
      "terminal values must match terminal level width"
    );

    let mut values = terminal_values;
    for level in (0..self.up_probabilities.len()).rev() {
      let mut step_values = Array1::zeros(level + 1);
      for node in 0..=level {
        let p = self.up_probabilities[level][node];
        let expected = p * values[node + 1] + (T::one() - p) * values[node];
        step_values[node] = discount(level, node, self.states[level][node]) * expected;
      }
      values = step_values;
    }
    values[0]
  }
}

/// Trinomial branch centered on a node of the next level.
#[derive(Debug, Clone, Copy)]
pub struct TrinomialBranch<T: RealExt> {
  /// Index of the middle child on the next level.
  pub center_index: usize,
  /// Probability of moving to the child below `center_index`.
  pub down_probability: T,
  /// Probability of moving to the child at `center_index`.
  pub middle_probability: T,
  /// Probability of moving to the child above `center_index`.
  pub up_probability: T,
}

impl<T: RealExt> TrinomialBranch<T> {
  /// Probabilities as `[down, middle, up]`.
  pub fn probabilities(&self) -> [T; 3] {
    [
      self.down_probability,
      self.middle_probability,
      self.up_probability,
    ]
  }
}

/// Recombining trinomial tree with per-node centered branches.
#[derive(Debug, Clone)]
pub struct TrinomialTree<T: RealExt> {
  /// Time step.
  pub dt: T,
  /// Grid states for each level.
  pub states: Vec<Array1<T>>,
  /// Branch definitions for each non-terminal level.
  pub branches: Vec<Vec<TrinomialBranch<T>>>,
}

impl<T: RealExt> TrinomialTree<T> {
  /// Levels `2i + 1` wide; panics unless every branch's three probabilities are finite and `≥ 0`
  /// and sum to 1 within `8ε`.
  pub fn new(dt: T, states: Vec<Array1<T>>, branches: Vec<Vec<TrinomialBranch<T>>>) -> Self {
    assert_eq!(
      states.len(),
      branches.len() + 1,
      "states must have one more level than branches"
    );
    for (level, nodes) in states.iter().enumerate() {
      let expected = 2 * level + 1;
      assert_eq!(
        nodes.len(),
        expected,
        "trinomial level {level} must contain {expected} nodes"
      );
    }
    for (level, step_branches) in branches.iter().enumerate() {
      let expected = 2 * level + 1;
      assert_eq!(
        step_branches.len(),
        expected,
        "branch level {level} must contain {expected} nodes"
      );
      let next_width = 2 * (level + 1) + 1;
      for (node, branch) in step_branches.iter().enumerate() {
        assert!(
          branch.center_index > 0 && branch.center_index + 1 < next_width,
          "branch center must leave room for down and up children"
        );
        check_probabilities(level, node, branch);
      }
    }
    Self {
      dt,
      states,
      branches,
    }
  }

  /// Backward induction under node-specific discount factors.
  pub fn backward_induct<F>(&self, terminal_values: Array1<T>, mut discount: F) -> T
  where
    F: FnMut(usize, usize, T) -> T,
  {
    assert_eq!(
      terminal_values.len(),
      self.states.last().map_or(0, Array1::len),
      "terminal values must match terminal level width"
    );

    let mut values = terminal_values;
    for level in (0..self.branches.len()).rev() {
      let width = self.states[level].len();
      let mut step_values = Array1::zeros(width);
      for node in 0..width {
        let branch = self.branches[level][node];
        let expected = branch.down_probability * values[branch.center_index - 1]
          + branch.middle_probability * values[branch.center_index]
          + branch.up_probability * values[branch.center_index + 1];
        step_values[node] = discount(level, node, self.states[level][node]) * expected;
      }
      values = step_values;
    }
    values[0]
  }
}

fn check_probabilities<T: RealExt>(level: usize, node: usize, branch: &TrinomialBranch<T>) {
  let fields = [
    ("down_probability", branch.down_probability),
    ("middle_probability", branch.middle_probability),
    ("up_probability", branch.up_probability),
  ];
  for (name, p) in fields {
    assert!(
      p.is_finite() && p >= T::zero(),
      "branches[{level}][{node}].{name} must satisfy `{name}.is_finite() && {name} >= T::zero()`, \
       got branches[{level}][{node}].{name} = {p:?}"
    );
  }
  let total = branch.down_probability + branch.middle_probability + branch.up_probability;
  assert!(
    (total - T::one()).abs() <= T::from_f64_fast(8.0) * T::epsilon(),
    "branches[{level}][{node}] must satisfy `(down + middle + up - T::one()).abs() <= 8.0 * T::epsilon()`, \
     got down + middle + up = {total:?}"
  );
}
