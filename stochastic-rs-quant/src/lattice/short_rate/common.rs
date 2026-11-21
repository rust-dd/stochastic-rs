//! Internal helpers shared by the short-rate tree engines.

use ndarray::Array1;

use super::OneFactorShortRateModel;
use crate::lattice::tree::TrinomialBranch;
use crate::lattice::tree::TrinomialTree;
use crate::traits::RealExt;

/// The state every short-rate tree is built for, `dx = a(m − x)dt + σ dW` from `initial_state`.
#[derive(Debug, Clone, Copy)]
pub(super) struct OrnsteinUhlenbeckFactor<T: RealExt> {
  pub(super) initial_state: T,
  pub(super) long_run_mean: T,
  pub(super) mean_reversion: T,
  pub(super) sigma: T,
}

/// Each level on the state's exact mean, each branch on the nearest node: no branch below 1/24.
///
/// $$
/// M = x\,e^{-a\Delta t},\quad V = \sigma^2\,\frac{1-e^{-2a\Delta t}}{2a},\quad
/// \Delta x = \sqrt{3V},\quad k = \operatorname{round}(M/\Delta x),\quad \eta = M - k\,\Delta x
/// $$
/// $$
/// p_u = \frac16 + \frac{\eta^2}{6V} + \frac{\eta}{2\sqrt{3V}},\qquad
/// p_m = \frac23 - \frac{\eta^2}{3V},\qquad
/// p_d = \frac16 + \frac{\eta^2}{6V} - \frac{\eta}{2\sqrt{3V}}
/// $$
///
/// Reference: Hull & White (1994), J. Deriv. 2(1), 7–16, §II, note 5, DOI 10.3905/jod.1994.407902
/// Reference: Brigo & Mercurio (2006), §3.3.3, p. 78, read in Baldvinsdóttir & Palmborg (2011)
/// Reference: Hull & White (1996), J. Deriv. 3(3), 26–36, §I: each level on the analytic mean
pub(super) fn build_one_factor_trinomial_tree<T: RealExt>(
  factor: &OrnsteinUhlenbeckFactor<T>,
  horizon: T,
  steps: usize,
) -> TrinomialTree<T> {
  require_positive("horizon", horizon);
  assert!(
    steps > 0,
    "steps must satisfy `steps > 0`, got steps = {steps}"
  );
  let OrnsteinUhlenbeckFactor {
    initial_state,
    long_run_mean,
    mean_reversion,
    sigma,
  } = *factor;
  let dt = horizon / T::from_usize_(steps);
  let two_a_dt = (mean_reversion + mean_reversion) * dt;
  // `(1 − e^{−2aΔt})/(2aΔt)`, so `σ` is never squared and `a = 0` is `V = σ²Δt`.
  let variance_share = if two_a_dt > T::zero() {
    -(-two_a_dt).exp_m1() / two_a_dt
  } else {
    T::one()
  };
  let dx = sigma * (T::from_f64_fast(3.0) * dt * variance_share).sqrt();
  let decay = (-mean_reversion * dt).exp();
  let offset = |j: usize, i: usize| T::from_f64_fast(j as f64 - i as f64);

  let states = (0..=steps)
    .map(|i| {
      let time = T::from_usize_(i) * dt;
      let mean = long_run_mean + (initial_state - long_run_mean) * (-mean_reversion * time).exp();
      Array1::from_iter((0..=2 * i).map(|j| mean + offset(j, i) * dx))
    })
    .collect::<Vec<_>>();

  let branches = (0..steps)
    .map(|i| {
      (0..=2 * i)
        .map(|j| nearest_node_branch(offset(j, i) * decay, i + 1))
        .collect::<Vec<_>>()
    })
    .collect::<Vec<_>>();

  TrinomialTree::new(dt, states, branches)
}

pub(super) fn require_finite<T: RealExt>(name: &str, value: T) {
  assert!(
    value.is_finite(),
    "{name} must satisfy `{name}.is_finite()`, got {name} = {value:?}"
  );
}

pub(super) fn require_non_negative<T: RealExt>(name: &str, value: T) {
  assert!(
    value.is_finite() && value >= T::zero(),
    "{name} must satisfy `{name}.is_finite() && {name} >= T::zero()`, got {name} = {value:?}"
  );
}

pub(super) fn require_positive<T: RealExt>(name: &str, value: T) {
  assert!(
    value.is_finite() && value > T::zero(),
    "{name} must satisfy `{name}.is_finite() && {name} > T::zero()`, got {name} = {value:?}"
  );
}

/// The branch whose exact conditional mean lies `target = M/Δx` spacings off the next centre.
fn nearest_node_branch<T: RealExt>(target: T, next_center: usize) -> TrinomialBranch<T> {
  let k = target.round();
  // `e = η/Δx`: with `Δx² = 3V` the formulas above are `1/6 + e²/2 ± e/2` and `2/3 − e²`.
  let e = target - k;
  let half = T::from_f64_fast(0.5);
  let outer = T::from_f64_fast(1.0 / 6.0) + half * e * e;
  let k = k.to_isize().expect("a finite target rounds to an integer");
  TrinomialBranch {
    center_index: next_center
      .checked_add_signed(k)
      .expect("|k| is at most the node's offset"),
    down_probability: outer - half * e,
    middle_probability: T::from_f64_fast(2.0 / 3.0) - e * e,
    up_probability: outer + half * e,
  }
}

pub(super) fn price_one_factor_zcb<T: RealExt, M: OneFactorShortRateModel<T>>(
  tree: &TrinomialTree<T>,
  model: &M,
) -> T {
  let terminal = Array1::from_elem(tree.states.last().map_or(0, Array1::len), T::one());
  tree.backward_induct(terminal, |level, _node, state| {
    let time = T::from_usize_(level) * tree.dt;
    (-model.short_rate(time, state) * tree.dt).exp()
  })
}

/// G2++ joint probabilities `Π⁰ + s̄·(|ρ|/36)·M`, Brigo–Mercurio's `M` +5 on the corners ρ favours,
/// −1 on the other two, −4 on the edges, +8 at the centre; `s̄ ≤ 1`, the largest keeping `Π ≥ 0`.
/// Reference: Auer (2009), Proc. World Congress on Engineering II, 1388–1389, Brigo–Mercurio's BM′
pub(crate) fn correlated_joint_probabilities<T: RealExt>(
  x_branch: TrinomialBranch<T>,
  y_branch: TrinomialBranch<T>,
  rho: T,
) -> [[T; 3]; 3] {
  let px = x_branch.probabilities();
  let py = y_branch.probabilities();
  let mut joint = [[T::zero(); 3]; 3];
  for i in 0..3 {
    for j in 0..3 {
      joint[i][j] = px[i] * py[j];
    }
  }

  let (corner, edge, centre) = (
    T::from_f64_fast(5.0),
    T::from_f64_fast(-4.0),
    T::from_f64_fast(8.0),
  );
  let (same, opposite) = if rho > T::zero() {
    (corner, -T::one())
  } else {
    (-T::one(), corner)
  };
  let pattern = [
    [same, edge, opposite],
    [edge, centre, edge],
    [opposite, edge, same],
  ];
  let scale = rho.abs() / T::from_f64_fast(36.0);
  let mut s_bar = T::one();
  for i in 0..3 {
    for j in 0..3 {
      let delta = pattern[i][j] * scale;
      if delta < T::zero() && joint[i][j] + delta < T::zero() {
        s_bar = s_bar.min_or_nan(joint[i][j] / -delta);
      }
    }
  }
  // Rows and columns of `M` sum to zero, so the marginals hold; the floor only meets a −0 or the
  // binding cell's last-ulp negative.
  for i in 0..3 {
    for j in 0..3 {
      joint[i][j] = (joint[i][j] + s_bar * pattern[i][j] * scale).max_or_nan(T::zero());
    }
  }
  joint
}

#[cfg(test)]
mod tests {
  use super::*;

  fn symmetric() -> TrinomialBranch<f64> {
    TrinomialBranch {
      center_index: 1,
      down_probability: 1.0 / 6.0,
      middle_probability: 2.0 / 3.0,
      up_probability: 1.0 / 6.0,
    }
  }

  /// One-step correlation of the moves `−1, 0, +1` on both axes, about their means.
  fn realised_correlation(joint: &[[f64; 3]; 3]) -> f64 {
    let moves = [-1.0, 0.0, 1.0];
    let (mut mx, mut my) = (0.0, 0.0);
    for i in 0..3 {
      for j in 0..3 {
        mx += joint[i][j] * moves[i];
        my += joint[i][j] * moves[j];
      }
    }
    let (mut cov, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for i in 0..3 {
      for j in 0..3 {
        let (dx, dy) = (moves[i] - mx, moves[j] - my);
        cov += joint[i][j] * dx * dy;
        vx += joint[i][j] * dx * dx;
        vy += joint[i][j] * dy * dy;
      }
    }
    cov / (vx * vy).sqrt()
  }

  #[test]
  fn the_symmetric_branch_realises_rho_exactly_up_to_one() {
    for rho in [-1.0, -0.9, -0.7, -1.0 / 3.0, 0.0, 0.5, 0.95, 1.0] {
      let joint = correlated_joint_probabilities(symmetric(), symmetric(), rho);
      let realised = realised_correlation(&joint);
      assert!(
        (realised - rho).abs() < 1e-14,
        "rho {rho}: realised {realised}"
      );
      let (mut rows, mut cols) = ([0.0; 3], [0.0; 3]);
      for i in 0..3 {
        for j in 0..3 {
          assert!(
            joint[i][j] >= 0.0,
            "rho {rho}: cell ({i}, {j}) = {}",
            joint[i][j]
          );
          rows[i] += joint[i][j];
          cols[j] += joint[i][j];
        }
      }
      let marginal = symmetric().probabilities();
      for k in 0..3 {
        assert!((rows[k] - marginal[k]).abs() < 1e-15 && (cols[k] - marginal[k]).abs() < 1e-15);
      }
    }
  }

  #[test]
  fn a_drift_shifted_branch_scales_the_pattern_instead_of_dropping_it() {
    // Auer's s̄ = 35/36 on the branch (1/4, 7/12, 1/6) at rho = −0.9 realises −7√177/118 exactly.
    let shifted = TrinomialBranch {
      center_index: 1,
      down_probability: 0.25,
      middle_probability: 7.0 / 12.0,
      up_probability: 1.0 / 6.0,
    };
    let joint = correlated_joint_probabilities(shifted, symmetric(), -0.9);
    let min_cell = f64::min_or_nan_fold(joint.iter().flatten().copied());
    assert!(
      (0.0..1e-15).contains(&min_cell),
      "the binding cell is zero: {min_cell}"
    );
    let realised = realised_correlation(&joint);
    assert!(
      (realised + 7.0 * 177.0_f64.sqrt() / 118.0).abs() < 1e-14,
      "{realised}"
    );
  }
}
