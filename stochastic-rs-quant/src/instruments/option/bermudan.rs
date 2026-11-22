//! Bermudan swaption on a short-rate trinomial lattice, the swap reduced to zero-coupon bonds:
//!
//! $$
//! V_{\mathrm{float}}(t)=N\,\bigl[1-P(t,T_N)\bigr],\qquad
//! V_{\mathrm{fixed}}(t)=N\,K\,\sum_{T_i>t}\delta_i\,P(t,T_i)
//! $$
//! so `V_swap = V_float ± V_fixed` at every node by backward induction, exercise at chosen levels.
//! Reference: Hull & White (1994), J. Deriv. 2(1), 7–16, DOI 10.3905/jod.1994.407902
//! Reference: Brigo & Mercurio (2006), 2nd ed., §13, DOI 10.1007/978-3-540-34604-3

use chrono::NaiveDate;
use ndarray::Array1;
use ndarray::Array2;

use super::types::BermudanSwaptionValuation;
use super::types::ExerciseSchedule;
use super::types::SwaptionDirection;
use super::types::TreeCouponSchedule;
use crate::calendar::DayCountConvention;
use crate::lattice::G2ppTree;
use crate::lattice::OneFactorShortRateModel;
use crate::lattice::TrinomialTree;
use crate::lattice::TwoFactorShortRateModel;
use crate::lattice::short_rate::correlated_joint_probabilities;
use crate::traits::RealExt;

/// Bermudan swaption on tree-level schedules, priced on any [`BermudanTree`].
#[derive(Debug, Clone)]
pub struct BermudanSwaption<T: RealExt> {
  /// Payoff direction.
  pub direction: SwaptionDirection,
  /// Fixed strike $K$.
  pub strike: T,
  /// Swap notional.
  pub notional: T,
  /// Tree levels at which the holder may exercise.
  pub exercise_schedule: ExerciseSchedule,
  /// Tree levels carrying fixed-leg coupon payments.
  pub coupon_schedule: TreeCouponSchedule<T>,
}

impl<T: RealExt> BermudanSwaption<T> {
  /// Build a Bermudan swaption tied to the tree-level schedules.
  pub fn new(
    direction: SwaptionDirection,
    strike: T,
    notional: T,
    exercise_schedule: ExerciseSchedule,
    coupon_schedule: TreeCouponSchedule<T>,
  ) -> Self {
    Self {
      direction,
      strike,
      notional,
      exercise_schedule,
      coupon_schedule,
    }
  }

  pub fn valuation(&self, tree: &impl BermudanTree<T>) -> BermudanSwaptionValuation<T> {
    tree.bermudan_valuation(self)
  }

  pub fn npv(&self, tree: &impl BermudanTree<T>) -> T {
    self.valuation(tree).npv
  }

  /// Snaps each date to the nearest level of a tree stepping by `dt` years from the valuation
  /// date; panics unless `coupon_dates` and `accrual_factors` have one length.
  pub fn from_dates(
    direction: SwaptionDirection,
    strike: T,
    notional: T,
    dates: &BermudanDates<T>,
    dt: T,
  ) -> Self {
    let levels = |of: &[NaiveDate]| snap_to_levels(dates.valuation_date, dates.day_count, dt, of);
    Self::new(
      direction,
      strike,
      notional,
      ExerciseSchedule::new(levels(&dates.exercise_dates)),
      TreeCouponSchedule::new(levels(&dates.coupon_dates), dates.accrual_factors.clone()),
    )
  }
}

/// Calendar form of a Bermudan swaption's exercise and coupon schedule.
#[derive(Debug, Clone)]
pub struct BermudanDates<T: RealExt> {
  pub valuation_date: NaiveDate,
  /// Converts each date to a year fraction from `valuation_date`.
  pub day_count: DayCountConvention,
  pub exercise_dates: Vec<NaiveDate>,
  pub coupon_dates: Vec<NaiveDate>,
  /// One accrual factor per coupon date, in the same order.
  pub accrual_factors: Vec<T>,
}

/// A lattice a [`BermudanSwaption`] prices on by backward induction; a custom one-factor tree
/// answers with [`one_factor_valuation`].
pub trait BermudanTree<T: RealExt> {
  fn bermudan_valuation(&self, swaption: &BermudanSwaption<T>) -> BermudanSwaptionValuation<T>;
}

/// Backward induction on any one-factor trinomial short-rate tree.
pub fn one_factor_valuation<T: RealExt, M: OneFactorShortRateModel<T>>(
  swaption: &BermudanSwaption<T>,
  tree: &TrinomialTree<T>,
  model: &M,
) -> BermudanSwaptionValuation<T> {
  BermudanSwaptionValuation {
    npv: price_on_one_factor_tree(swaption, tree, model),
    exercise_count: swaption.exercise_schedule.levels.len(),
  }
}

/// Joint `(x, y)` backward induction on a G2++ tree with correlated branches.
pub(crate) fn g2pp_valuation<T: RealExt>(
  swaption: &BermudanSwaption<T>,
  tree: &G2ppTree<T>,
) -> BermudanSwaptionValuation<T> {
  BermudanSwaptionValuation {
    npv: price_on_two_factor_tree(swaption, tree),
    exercise_count: swaption.exercise_schedule.levels.len(),
  }
}

fn snap_to_levels<T: RealExt>(
  valuation_date: NaiveDate,
  day_count: DayCountConvention,
  dt: T,
  dates: &[NaiveDate],
) -> Vec<usize> {
  dates
    .iter()
    .map(|&date| {
      let tau: T = day_count.year_fraction(valuation_date, date);

      (tau / dt).to_f64().unwrap_or(0.0).round().max(0.0) as usize
    })
    .collect()
}

fn price_on_one_factor_tree<T: RealExt, M: OneFactorShortRateModel<T>>(
  swaption: &BermudanSwaption<T>,
  tree: &TrinomialTree<T>,
  model: &M,
) -> T {
  let BermudanSwaption {
    direction,
    strike,
    notional,
    exercise_schedule,
    coupon_schedule,
  } = swaption;
  let (direction, strike, notional) = (*direction, *strike, *notional);
  let n_levels = tree.states.len();
  assert!(n_levels > 0, "tree must contain at least one level");
  let terminal = n_levels - 1;
  let dt = tree.dt;

  let zcb = backward_expectation(tree, model, |level, _| {
    if level == terminal {
      Array1::from_elem(tree.states[terminal].len(), T::one())
    } else {
      Array1::zeros(tree.states[level].len())
    }
  });

  let coupon_injection = |level: usize| -> T {
    for (i, &l) in coupon_schedule.levels.iter().enumerate() {
      if l == level {
        return notional * strike * coupon_schedule.accrual_factors[i];
      }
    }
    T::zero()
  };

  let fixed_leg = backward_expectation_with_injection(tree, model, coupon_injection);

  let direction_sign = match direction {
    SwaptionDirection::Payer => T::one(),
    SwaptionDirection::Receiver => -T::one(),
  };

  let intrinsic_at = |level: usize, node: usize| -> T {
    let p = zcb[level][node];
    let fixed = fixed_leg[level][node];
    let float = notional * (T::one() - p);
    direction_sign * (float - fixed)
  };

  let mut option = Array1::zeros(tree.states[terminal].len());
  if exercise_schedule.contains(terminal) {
    for node in 0..tree.states[terminal].len() {
      option[node] = intrinsic_at(terminal, node).max(T::zero());
    }
  }

  for level in (0..terminal).rev() {
    let width = tree.states[level].len();
    let mut step = Array1::zeros(width);
    for node in 0..width {
      let branch = tree.branches[level][node];
      let expected = branch.down_probability * option[branch.center_index - 1]
        + branch.middle_probability * option[branch.center_index]
        + branch.up_probability * option[branch.center_index + 1];
      let time = T::from_usize_(level) * dt;
      let rate = model.short_rate(time, tree.states[level][node]);
      let discount = (-rate * dt).exp();
      step[node] = discount * expected;
    }

    if exercise_schedule.contains(level) {
      for node in 0..width {
        let intrinsic = intrinsic_at(level, node);
        step[node] = step[node].max(intrinsic);
      }
    }

    option = step;
  }

  option[0]
}

fn backward_expectation<T: RealExt, M: OneFactorShortRateModel<T>>(
  tree: &TrinomialTree<T>,
  model: &M,
  seed: impl Fn(usize, &TrinomialTree<T>) -> Array1<T>,
) -> Vec<Array1<T>> {
  let n_levels = tree.states.len();
  let terminal = n_levels - 1;
  let dt = tree.dt;

  let mut storage: Vec<Array1<T>> = (0..n_levels)
    .map(|level| Array1::zeros(tree.states[level].len()))
    .collect();
  storage[terminal] = seed(terminal, tree);

  let mut values = storage[terminal].clone();
  for level in (0..terminal).rev() {
    let width = tree.states[level].len();
    let mut step = Array1::zeros(width);
    for node in 0..width {
      let branch = tree.branches[level][node];
      let expected = branch.down_probability * values[branch.center_index - 1]
        + branch.middle_probability * values[branch.center_index]
        + branch.up_probability * values[branch.center_index + 1];
      let time = T::from_usize_(level) * dt;
      let rate = model.short_rate(time, tree.states[level][node]);
      let discount = (-rate * dt).exp();
      step[node] = discount * expected;
    }
    values = step.clone();
    storage[level] = step;
  }

  storage
}

fn backward_expectation_with_injection<T: RealExt, M: OneFactorShortRateModel<T>>(
  tree: &TrinomialTree<T>,
  model: &M,
  injection: impl Fn(usize) -> T,
) -> Vec<Array1<T>> {
  let n_levels = tree.states.len();
  let terminal = n_levels - 1;
  let dt = tree.dt;

  let mut storage: Vec<Array1<T>> = (0..n_levels)
    .map(|level| Array1::zeros(tree.states[level].len()))
    .collect();

  let terminal_inject = injection(terminal);
  let mut values = Array1::from_elem(tree.states[terminal].len(), terminal_inject);
  storage[terminal] = values.clone();

  for level in (0..terminal).rev() {
    let width = tree.states[level].len();
    let mut step = Array1::zeros(width);
    for node in 0..width {
      let branch = tree.branches[level][node];
      let expected = branch.down_probability * values[branch.center_index - 1]
        + branch.middle_probability * values[branch.center_index]
        + branch.up_probability * values[branch.center_index + 1];
      let time = T::from_usize_(level) * dt;
      let rate = model.short_rate(time, tree.states[level][node]);
      let discount = (-rate * dt).exp();
      step[node] = discount * expected;
    }

    let inject = injection(level);
    if inject != T::zero() {
      step.mapv_inplace(|v| v + inject);
    }
    values = step.clone();
    storage[level] = step;
  }

  storage
}

fn price_on_two_factor_tree<T: RealExt>(swaption: &BermudanSwaption<T>, tree: &G2ppTree<T>) -> T {
  let BermudanSwaption {
    direction,
    strike,
    notional,
    exercise_schedule,
    coupon_schedule,
  } = swaption;
  let (direction, strike, notional) = (*direction, *strike, *notional);
  let n_levels = tree.x_tree.states.len();
  assert!(n_levels > 0, "G2++ tree must contain at least one level");
  let terminal = n_levels - 1;
  let dt = tree.dt;

  let injection_at = |level: usize| -> T {
    for (i, &l) in coupon_schedule.levels.iter().enumerate() {
      if l == level {
        return notional * strike * coupon_schedule.accrual_factors[i];
      }
    }
    T::zero()
  };

  let zcb = backward_expectation_2d(tree, |level, _| {
    let x_width = tree.x_tree.states[level].len();
    let y_width = tree.y_tree.states[level].len();
    if level == terminal {
      Array2::from_elem((x_width, y_width), T::one())
    } else {
      Array2::zeros((x_width, y_width))
    }
  });

  let fixed_leg = backward_expectation_2d_with_injection(tree, injection_at);

  let direction_sign = match direction {
    SwaptionDirection::Payer => T::one(),
    SwaptionDirection::Receiver => -T::one(),
  };

  let intrinsic = |level: usize, ix: usize, iy: usize| -> T {
    let p = zcb[level][[ix, iy]];
    let fixed = fixed_leg[level][[ix, iy]];
    let float = notional * (T::one() - p);
    direction_sign * (float - fixed)
  };

  let x_width_terminal = tree.x_tree.states[terminal].len();
  let y_width_terminal = tree.y_tree.states[terminal].len();
  let mut option = Array2::zeros((x_width_terminal, y_width_terminal));
  if exercise_schedule.contains(terminal) {
    for ix in 0..x_width_terminal {
      for iy in 0..y_width_terminal {
        option[[ix, iy]] = intrinsic(terminal, ix, iy).max(T::zero());
      }
    }
  }

  for level in (0..terminal).rev() {
    let x_width = tree.x_tree.states[level].len();
    let y_width = tree.y_tree.states[level].len();
    let mut step = Array2::zeros((x_width, y_width));
    let time = T::from_usize_(level) * dt;
    let rho = tree.model.correlation();
    for ix in 0..x_width {
      let x_branch = tree.x_tree.branches[level][ix];
      let x_children = [
        x_branch.center_index - 1,
        x_branch.center_index,
        x_branch.center_index + 1,
      ];
      for iy in 0..y_width {
        let y_branch = tree.y_tree.branches[level][iy];
        let y_children = [
          y_branch.center_index - 1,
          y_branch.center_index,
          y_branch.center_index + 1,
        ];
        let joint = correlated_joint_probabilities(x_branch, y_branch, rho);
        let mut expected = T::zero();
        for ax in 0..3 {
          for ay in 0..3 {
            expected += joint[ax][ay] * option[[x_children[ax], y_children[ay]]];
          }
        }
        let rate = tree.model.short_rate(
          time,
          tree.x_tree.states[level][ix],
          tree.y_tree.states[level][iy],
        );
        step[[ix, iy]] = (-rate * dt).exp() * expected;
      }
    }

    if exercise_schedule.contains(level) {
      for ix in 0..x_width {
        for iy in 0..y_width {
          let ex = intrinsic(level, ix, iy);
          step[[ix, iy]] = step[[ix, iy]].max(ex);
        }
      }
    }

    option = step;
  }

  option[[0, 0]]
}

fn backward_expectation_2d<T: RealExt>(
  tree: &G2ppTree<T>,
  seed: impl Fn(usize, &G2ppTree<T>) -> Array2<T>,
) -> Vec<Array2<T>> {
  let n_levels = tree.x_tree.states.len();
  let terminal = n_levels - 1;
  let dt = tree.dt;
  let rho = tree.model.correlation();

  let mut storage: Vec<Array2<T>> = (0..n_levels)
    .map(|level| {
      Array2::zeros((
        tree.x_tree.states[level].len(),
        tree.y_tree.states[level].len(),
      ))
    })
    .collect();
  storage[terminal] = seed(terminal, tree);

  let mut values = storage[terminal].clone();
  for level in (0..terminal).rev() {
    let x_width = tree.x_tree.states[level].len();
    let y_width = tree.y_tree.states[level].len();
    let mut step = Array2::zeros((x_width, y_width));
    let time = T::from_usize_(level) * dt;
    for ix in 0..x_width {
      let x_branch = tree.x_tree.branches[level][ix];
      let x_children = [
        x_branch.center_index - 1,
        x_branch.center_index,
        x_branch.center_index + 1,
      ];
      for iy in 0..y_width {
        let y_branch = tree.y_tree.branches[level][iy];
        let y_children = [
          y_branch.center_index - 1,
          y_branch.center_index,
          y_branch.center_index + 1,
        ];
        let joint = correlated_joint_probabilities(x_branch, y_branch, rho);
        let mut expected = T::zero();
        for ax in 0..3 {
          for ay in 0..3 {
            expected += joint[ax][ay] * values[[x_children[ax], y_children[ay]]];
          }
        }
        let rate = tree.model.short_rate(
          time,
          tree.x_tree.states[level][ix],
          tree.y_tree.states[level][iy],
        );
        step[[ix, iy]] = (-rate * dt).exp() * expected;
      }
    }
    values = step.clone();
    storage[level] = step;
  }

  storage
}

fn backward_expectation_2d_with_injection<T: RealExt>(
  tree: &G2ppTree<T>,
  injection: impl Fn(usize) -> T,
) -> Vec<Array2<T>> {
  let n_levels = tree.x_tree.states.len();
  let terminal = n_levels - 1;
  let dt = tree.dt;
  let rho = tree.model.correlation();

  let mut storage: Vec<Array2<T>> = (0..n_levels)
    .map(|level| {
      Array2::zeros((
        tree.x_tree.states[level].len(),
        tree.y_tree.states[level].len(),
      ))
    })
    .collect();
  let terminal_inject = injection(terminal);
  let mut values = Array2::from_elem(
    (
      tree.x_tree.states[terminal].len(),
      tree.y_tree.states[terminal].len(),
    ),
    terminal_inject,
  );
  storage[terminal] = values.clone();

  for level in (0..terminal).rev() {
    let x_width = tree.x_tree.states[level].len();
    let y_width = tree.y_tree.states[level].len();
    let mut step = Array2::zeros((x_width, y_width));
    let time = T::from_usize_(level) * dt;
    for ix in 0..x_width {
      let x_branch = tree.x_tree.branches[level][ix];
      let x_children = [
        x_branch.center_index - 1,
        x_branch.center_index,
        x_branch.center_index + 1,
      ];
      for iy in 0..y_width {
        let y_branch = tree.y_tree.branches[level][iy];
        let y_children = [
          y_branch.center_index - 1,
          y_branch.center_index,
          y_branch.center_index + 1,
        ];
        let joint = correlated_joint_probabilities(x_branch, y_branch, rho);
        let mut expected = T::zero();
        for ax in 0..3 {
          for ay in 0..3 {
            expected += joint[ax][ay] * values[[x_children[ax], y_children[ay]]];
          }
        }
        let rate = tree.model.short_rate(
          time,
          tree.x_tree.states[level][ix],
          tree.y_tree.states[level][iy],
        );
        step[[ix, iy]] = (-rate * dt).exp() * expected;
      }
    }

    let inject = injection(level);
    if inject != T::zero() {
      step.mapv_inplace(|v| v + inject);
    }
    values = step.clone();
    storage[level] = step;
  }

  storage
}
