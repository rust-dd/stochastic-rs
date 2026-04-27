//! Short-rate tree engines built on the generic lattice primitives.
//!
//! Reference: Hull & White (1994), Black & Karasinski (1991).

use ndarray::Array1;
use ndarray::Array2;

use super::tree::TrinomialBranch;
use super::tree::TrinomialTree;
use crate::curves::DiscountCurve;
use crate::traits::FloatExt;

/// Extensibility point for one-factor short-rate lattice models.
pub trait OneFactorShortRateModel<T: FloatExt>: Clone + Send + Sync {
  /// Initial state variable.
  fn initial_state(&self) -> T;

  /// Drift of the state process.
  fn drift(&self, time: T, state: T) -> T;

  /// Instantaneous diffusion of the state process.
  fn diffusion(&self, time: T, state: T) -> T;

  /// Short rate implied by the state variable.
  fn short_rate(&self, time: T, state: T) -> T;
}

/// Extensibility point for two-factor short-rate lattice models.
pub trait TwoFactorShortRateModel<T: FloatExt>: Clone + Send + Sync {
  /// Initial first factor.
  fn initial_x(&self) -> T;

  /// Initial second factor.
  fn initial_y(&self) -> T;

  /// Drift of the first factor.
  fn drift_x(&self, time: T, x: T) -> T;

  /// Drift of the second factor.
  fn drift_y(&self, time: T, y: T) -> T;

  /// Instantaneous diffusion of the first factor.
  fn diffusion_x(&self, time: T, x: T) -> T;

  /// Instantaneous diffusion of the second factor.
  fn diffusion_y(&self, time: T, y: T) -> T;

  /// Instantaneous factor correlation.
  fn correlation(&self) -> T;

  /// Short rate implied by the factor pair.
  fn short_rate(&self, time: T, x: T, y: T) -> T;
}

/// One-factor Hull-White / Vasicek tree model.
#[derive(Debug, Clone)]
pub struct HullWhiteTreeModel<T: FloatExt> {
  /// Initial short rate.
  pub initial_rate: T,
  /// Mean reversion speed.
  pub mean_reversion: T,
  /// Long-run level in the short-rate SDE.
  pub theta: T,
  /// Volatility.
  pub sigma: T,
}

impl<T: FloatExt> HullWhiteTreeModel<T> {
  /// Construct a Hull-White tree model.
  pub fn new(initial_rate: T, mean_reversion: T, theta: T, sigma: T) -> Self {
    Self {
      initial_rate,
      mean_reversion,
      theta,
      sigma,
    }
  }
}

impl<T: FloatExt> OneFactorShortRateModel<T> for HullWhiteTreeModel<T> {
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

/// Hull-White tree engine.
#[derive(Debug, Clone)]
pub struct HullWhiteTree<T: FloatExt> {
  /// Underlying model.
  pub model: HullWhiteTreeModel<T>,
  /// Trinomial lattice.
  pub tree: TrinomialTree<T>,
  /// Maturity horizon.
  pub horizon: T,
}

impl<T: FloatExt> HullWhiteTree<T> {
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

#[derive(Debug, Clone)]
struct OrnsteinUhlenbeckFactor<T: FloatExt> {
  initial_state: T,
  mean_reversion: T,
  sigma: T,
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

/// Hull-White model with time-dependent $\theta(t)$ calibrated to an initial
/// yield curve. The short rate is decomposed as $r(t) = x(t) + \alpha(t)$ where
/// $x(t)$ is a zero-mean Ornstein-Uhlenbeck process and $\alpha(t)$ is the
/// deterministic shift that reproduces market zero-coupon bond prices.
#[derive(Debug, Clone)]
pub struct CurveFittedHullWhiteModel<T: FloatExt> {
  /// Mean reversion speed $a$.
  pub mean_reversion: T,
  /// Volatility $\sigma$.
  pub sigma: T,
  /// Deterministic shift $\alpha(t_m)$ for $m = 0, 1, \dots$.
  pub alpha: Array1<T>,
  /// Tree time step.
  pub dt: T,
}

impl<T: FloatExt> OneFactorShortRateModel<T> for CurveFittedHullWhiteModel<T> {
  fn initial_state(&self) -> T {
    T::zero()
  }

  fn drift(&self, _time: T, state: T) -> T {
    -self.mean_reversion * state
  }

  fn diffusion(&self, _time: T, _state: T) -> T {
    self.sigma
  }

  fn short_rate(&self, time: T, state: T) -> T {
    let raw_level = (time / self.dt).to_f64().unwrap_or(0.0).round();
    let level = raw_level.max(0.0) as usize;
    let alpha = self
      .alpha
      .get(level)
      .copied()
      .unwrap_or_else(|| *self.alpha.last().expect("alpha schedule must be non-empty"));
    alpha + state
  }
}

/// Curve-fitted Hull-White trinomial tree engine.
///
/// Reference: Hull & White, "Using Hull-White Interest Rate Trees",
/// Journal of Derivatives 3(3), 26-36 (1996).
#[derive(Debug, Clone)]
pub struct CurveFittedHullWhiteTree<T: FloatExt> {
  /// Calibrated model.
  pub model: CurveFittedHullWhiteModel<T>,
  /// Trinomial lattice of the zero-mean $x(t)$ process.
  pub tree: TrinomialTree<T>,
  /// Maturity horizon.
  pub horizon: T,
}

impl<T: FloatExt> CurveFittedHullWhiteTree<T> {
  /// Build a Hull-White tree calibrated to `curve` with mean reversion `a`,
  /// volatility `sigma`, and `steps` tree steps over `horizon`.
  pub fn new(
    mean_reversion: T,
    sigma: T,
    curve: &DiscountCurve<T>,
    horizon: T,
    steps: usize,
  ) -> Self {
    let factor = OrnsteinUhlenbeckFactor {
      initial_state: T::zero(),
      mean_reversion,
      sigma,
    };
    let tree = build_one_factor_trinomial_tree(&factor, horizon, steps);
    let dt = horizon / T::from_usize_(steps);
    let alpha = fit_alpha_schedule(&tree, curve, dt);
    let model = CurveFittedHullWhiteModel {
      mean_reversion,
      sigma,
      alpha,
      dt,
    };
    Self {
      model,
      tree,
      horizon,
    }
  }

  /// Price a zero-coupon bond maturing at the tree horizon. Used as a
  /// calibration self-check — the price should match `curve.discount_factor(horizon)`.
  pub fn zero_coupon_bond_price(&self) -> T {
    price_one_factor_zcb(&self.tree, &self.model)
  }
}

fn fit_alpha_schedule<T: FloatExt>(
  tree: &TrinomialTree<T>,
  curve: &DiscountCurve<T>,
  dt: T,
) -> Array1<T> {
  let n_levels = tree.states.len();
  let steps = n_levels - 1;
  let mut alpha = Array1::zeros(n_levels);

  let mut arrow_debreu: Vec<Array1<T>> = (0..n_levels)
    .map(|level| Array1::zeros(tree.states[level].len()))
    .collect();
  arrow_debreu[0][0] = T::one();

  for m in 0..steps {
    let width_m = tree.states[m].len();
    let p_next = curve.discount_factor(T::from_usize_(m + 1) * dt);

    let mut weighted_sum = T::zero();
    for j in 0..width_m {
      let x_j = tree.states[m][j];
      weighted_sum += arrow_debreu[m][j] * (-x_j * dt).exp();
    }

    let alpha_m = if weighted_sum > T::zero() && p_next > T::zero() {
      -(p_next / weighted_sum).ln() / dt
    } else {
      T::zero()
    };
    alpha[m] = alpha_m;

    let width_next = tree.states[m + 1].len();
    let mut next = Array1::zeros(width_next);
    for j in 0..width_m {
      let branch = tree.branches[m][j];
      let x_j = tree.states[m][j];
      let discount = (-(x_j + alpha_m) * dt).exp();
      let base = arrow_debreu[m][j] * discount;
      next[branch.center_index - 1] += base * branch.down_probability;
      next[branch.center_index] += base * branch.middle_probability;
      next[branch.center_index + 1] += base * branch.up_probability;
    }
    arrow_debreu[m + 1] = next;
  }

  if steps > 0 {
    alpha[steps] = alpha[steps - 1];
  }

  alpha
}

fn build_one_factor_trinomial_tree<T: FloatExt, M: OneFactorShortRateModel<T>>(
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
            let j = (((mean - x0) / dx).round().to_f64().unwrap()) as isize;
            let j = j.clamp(-next_center + 1, next_center - 1);
            return TrinomialBranch {
              center_index: (j + next_center) as usize,
              down_probability: T::zero(),
              middle_probability: T::one(),
              up_probability: T::zero(),
            };
          }

          let j = (((mean - x0) / dx).round().to_f64().unwrap()) as isize;
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

fn price_one_factor_zcb<T: FloatExt, M: OneFactorShortRateModel<T>>(
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
