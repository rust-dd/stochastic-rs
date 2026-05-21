//! Internal helpers shared by the short-rate tree engines.

use ndarray::Array1;

use super::OneFactorShortRateModel;
use crate::lattice::tree::TrinomialBranch;
use crate::lattice::tree::TrinomialTree;
use crate::traits::FloatExt;

/// Zero-mean Ornstein-Uhlenbeck factor used as a primitive for Gaussian
/// short-rate constructions (G2++, curve-fitted Hull-White, …).
#[derive(Debug, Clone)]
pub(super) struct OrnsteinUhlenbeckFactor<T: FloatExt> {
  pub(super) initial_state: T,
  pub(super) mean_reversion: T,
  pub(super) sigma: T,
}

impl<T: FloatExt> OneFactorShortRateModel<T> for OrnsteinUhlenbeckFactor<T> {
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

pub(super) fn build_one_factor_trinomial_tree<T: FloatExt, M: OneFactorShortRateModel<T>>(
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

pub(super) fn price_one_factor_zcb<T: FloatExt, M: OneFactorShortRateModel<T>>(
  tree: &TrinomialTree<T>,
  model: &M,
) -> T {
  let terminal = Array1::from_elem(tree.states.last().map_or(0, Array1::len), T::one());
  tree.backward_induct(terminal, |level, _node, state| {
    let time = T::from_usize_(level) * tree.dt;
    (-model.short_rate(time, state) * tree.dt).exp()
  })
}

fn sanitize_probabilities<T: FloatExt>(down: &mut T, middle: &mut T, up: &mut T) {
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

/// Build correlated joint probabilities for a 2D trinomial lattice (G2++ /
/// two-factor Hull-White) using the **symmetric-branch corner correction**:
/// $\lambda = \rho / 12$ on the four corners.
///
/// **Tier-0 / sketch-grade:** the $\lambda = \rho/12$ correction matches
/// the requested factor covariance **exactly** only for the symmetric
/// trinomial branch ($p_u = p_d = 1/6$, $p_m = 2/3$, i.e. zero local drift).
/// For drift-shifted branches ($p_u \neq p_d$, the usual case off the lattice
/// midline), the correction does not hit the requested covariance — bias is
/// proportional to the local drift. For typical $a \leq 0.05$, $dt \leq 0.25$,
/// the resulting joint-probability error is on the order of $10^{-3}$,
/// translating to a few-bps bias on multi-year products.
///
/// For a per-cell exact match across asymmetric branches, replace the corner
/// correction with the linear system in Hull-White (2000), "The General
/// Hull-White Model and Super-calibration", equations 16–18.
pub(crate) fn correlated_joint_probabilities<T: FloatExt>(
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

  let mut lambda = rho / T::from_f64_fast(12.0);
  if lambda > T::zero() {
    let max_positive = joint[0][2].min(joint[2][0]);
    lambda = lambda.min(max_positive);
  } else if lambda < T::zero() {
    let max_negative = joint[0][0].min(joint[2][2]);
    lambda = lambda.max(-max_negative);
  }

  joint[0][0] += lambda;
  joint[2][2] += lambda;
  joint[0][2] -= lambda;
  joint[2][0] -= lambda;

  let mut sum = T::zero();
  for row in &joint {
    for value in row {
      sum += *value;
    }
  }
  if sum > T::zero() {
    for row in &mut joint {
      for value in row {
        *value = (*value).max(T::zero());
      }
    }
    let mut normalized_sum = T::zero();
    for row in &joint {
      for value in row {
        normalized_sum += *value;
      }
    }
    if normalized_sum > T::zero() {
      for row in &mut joint {
        for value in row {
          *value = *value / normalized_sum;
        }
      }
    }
  }

  joint
}
