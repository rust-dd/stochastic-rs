//! Internal helpers shared by the short-rate tree engines.

use ndarray::Array1;

use super::OneFactorShortRateModel;
use crate::lattice::tree::TrinomialBranch;
use crate::lattice::tree::TrinomialTree;
use crate::traits::RealExt;

/// Zero-mean Ornstein-Uhlenbeck factor used as a primitive for Gaussian
/// short-rate constructions (G2++, curve-fitted Hull-White, …).
#[derive(Debug, Clone)]
pub(super) struct OrnsteinUhlenbeckFactor<T: RealExt> {
  pub(super) initial_state: T,
  pub(super) mean_reversion: T,
  pub(super) sigma: T,
}

impl<T: RealExt> OneFactorShortRateModel<T> for OrnsteinUhlenbeckFactor<T> {
  fn initial_state(&self) -> T {
    self.initial_state
  }

  fn drift(&self, _time: T, state: T) -> T {
    -self.mean_reversion * state
  }

  fn diffusion(&self, _time: T, _state: T) -> T {
    self.sigma
  }

  fn short_rate(&self, _time: T, state: T) -> T {
    state
  }
}

pub(super) fn build_one_factor_trinomial_tree<T: RealExt, M: OneFactorShortRateModel<T>>(
  model: &M,
  horizon: T,
  steps: usize,
) -> TrinomialTree<T> {
  assert!(steps > 0, "steps must be strictly positive");
  let dt = horizon / T::from_usize_(steps);
  let sigma = model.diffusion(T::zero(), model.initial_state()).abs();
  let x0 = model.initial_state();
  let diffusion_dx = sigma * (T::from_f64_fast(3.0) * dt).sqrt();
  let drift_dx = model.drift(T::zero(), x0).abs() * dt;
  let dx = diffusion_dx.max(drift_dx).max(T::from_f64_fast(1e-6));

  let states = (0..=steps)
    .map(|level| {
      let center = level as isize;
      Array1::from_iter((0..(2 * level + 1)).map(|idx| {
        let j = idx as isize - center;
        x0 + T::from_f64_fast(j as f64) * dx
      }))
    })
    .collect::<Vec<_>>();

  let branches = (0..steps)
    .map(|level| {
      let time = T::from_usize_(level) * dt;
      let next_center = (level + 1) as isize;
      (0..(2 * level + 1))
        .map(|idx| {
          let state = states[level][idx];
          let mean = state + model.drift(time, state) * dt;
          let variance = model.diffusion(time, state).powi(2) * dt;

          if variance <= T::min_positive_val() {
            // mean/x0/dx are all derived from finite model coefficients here,
            // but defensively fall back to the centre node (j=0) if any
            // intermediate produces NaN (e.g. zero-diffusion *and* zero-drift
            // pathological inputs that bypass earlier validation).
            let j = (((mean - x0) / dx).round().to_f64().unwrap_or(0.0)) as isize;
            let j = j.clamp(-next_center + 1, next_center - 1);
            return TrinomialBranch {
              center_index: (j + next_center) as usize,
              down_probability: T::zero(),
              middle_probability: T::one(),
              up_probability: T::zero(),
            };
          }

          let j = (((mean - x0) / dx).round().to_f64().unwrap_or(0.0)) as isize;
          let j = j.clamp(-next_center + 1, next_center - 1);
          let target = x0 + T::from_f64_fast(j as f64) * dx;
          let eta = mean - target;
          let dx2 = dx * dx;
          let q = (variance + eta * eta) / dx2;
          let mut down = T::from_f64_fast(0.5) * (q - eta / dx);
          let mut up = T::from_f64_fast(0.5) * (q + eta / dx);
          let mut middle = T::one() - q;
          sanitize_probabilities(&mut down, &mut middle, &mut up);

          TrinomialBranch {
            center_index: (j + next_center) as usize,
            down_probability: down,
            middle_probability: middle,
            up_probability: up,
          }
        })
        .collect::<Vec<_>>()
    })
    .collect::<Vec<_>>();

  TrinomialTree::new(dt, states, branches)
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

fn sanitize_probabilities<T: RealExt>(down: &mut T, middle: &mut T, up: &mut T) {
  let zero = T::zero();
  let tol = T::from_f64_fast(1e-12);
  if *down < zero && *down > -tol {
    *down = zero;
  }
  if *middle < zero && *middle > -tol {
    *middle = zero;
  }
  if *up < zero && *up > -tol {
    *up = zero;
  }
  let sum = *down + *middle + *up;
  if sum > zero {
    *down = *down / sum;
    *middle = *middle / sum;
    *up = *up / sum;
  } else {
    *down = zero;
    *middle = T::one();
    *up = zero;
  }
}

/// G2++ joint probabilities `Π⁰ + s̄·(|ρ|/36)·M`, Brigo–Mercurio's `M` +5 on the corners ρ favours,
/// −1 on the other two, −4 on the edges, +8 at the centre; `s̄ ≤ 1`, the largest keeping `Π ≥ 0`.
/// Auer (2009), Proc. World Congress on Engineering II, 1388–1389: Brigo–Mercurio's pattern, BM′.
pub(crate) fn correlated_joint_probabilities<T: RealExt>(
  x_branch: TrinomialBranch<T>,
  y_branch: TrinomialBranch<T>,
  rho: T,
) -> [[T; 3]; 3] {
  let px = non_negative(x_branch.probabilities());
  let py = non_negative(y_branch.probabilities());
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

/// A branch the grid cannot centre (a vol under its `1e-6` floor, a mean reversion `a·dt > 2`)
/// has a negative probability; its non-negative part, renormalised, enters the product instead.
fn non_negative<T: RealExt>(p: [T; 3]) -> [T; 3] {
  if p.iter().all(|&q| q >= T::zero()) {
    return p;
  }
  let clipped = p.map(|q| q.max_or_nan(T::zero()));
  let total = clipped[0] + clipped[1] + clipped[2];
  clipped.map(|q| q / total)
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

  #[test]
  fn a_branch_with_a_negative_probability_enters_by_its_non_negative_part() {
    // An off-centre node of a factor whose vol is under the grid floor: (3/8, 3/4, −1/8).
    let off_grid = TrinomialBranch {
      center_index: 1,
      down_probability: 0.375,
      middle_probability: 0.75,
      up_probability: -0.125,
    };
    for rho in [0.0, -0.9, 0.9] {
      let joint = correlated_joint_probabilities(symmetric(), off_grid, rho);
      let cells = joint.iter().flatten().copied();
      assert!(
        f64::min_or_nan_fold(cells.clone()) >= 0.0,
        "rho {rho}: {joint:?}"
      );
      assert!(
        (cells.sum::<f64>() - 1.0).abs() < 1e-15,
        "rho {rho}: {joint:?}"
      );
      let up = joint[0][2] + joint[1][2] + joint[2][2];
      assert!(up.abs() < 1e-15, "rho {rho}: the clipped move keeps {up}");
    }
  }
}
