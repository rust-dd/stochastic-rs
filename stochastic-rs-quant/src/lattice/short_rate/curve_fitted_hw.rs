//! Curve-fitted Hull-White trinomial-tree engine.
//!
//! Reference: Hull & White, "Using Hull-White Interest Rate Trees",
//! Journal of Derivatives 3(3), 26-36 (1996).

use ndarray::Array1;

use super::common::build_one_factor_trinomial_tree;
use super::common::price_one_factor_zcb;
use super::common::OrnsteinUhlenbeckFactor;
use super::OneFactorShortRateModel;
use crate::curves::DiscountCurve;
use crate::lattice::tree::TrinomialTree;
use crate::traits::FloatExt;

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
