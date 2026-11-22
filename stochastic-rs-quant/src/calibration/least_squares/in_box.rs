//! The bounds side of `minimize_in_box`: the set a fit holds on its bounds through each solve,
//! after Bertsekas's projected Newton method, a restart standing for its step.
//!
//! Reference: Bertsekas (1982), SIAM J. Control Optim., eqs. (32), (81)–(84), DOI 10.1137/0320018

use std::cell::Cell;
use std::cell::RefCell;

use basin::Jacobian;
use basin::Residual;
use basin::TerminationReason;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use super::LeastSquaresProblem;
use super::Objective;
use super::Run;

/// How far each coordinate of the optimiser's point can fall and rise before its value meets a
/// bound its map folds or clamps at, and `strip`, its stencil step: the paper's `ε`.
pub(crate) struct Room {
  pub(crate) down: Array1<f64>,
  pub(crate) up: Array1<f64>,
  pub(crate) strip: Array1<f64>,
}

pub(super) type Rooms = Box<dyn Fn(ArrayView1<f64>) -> Room>;

/// The coordinates of (81)'s set whose diagonal step, `M = 1/‖Jⱼ‖²`, is cut at the bound: the
/// step onto it. Such a step is its own term of `wₖ`, so `εₖ` holds whatever the other terms.
pub(super) fn cut(
  jacobian: &Array2<f64>,
  residuals: &Array1<f64>,
  room: &Room,
) -> Vec<Option<f64>> {
  let columns = jacobian.columns().into_iter().enumerate();
  let onto = columns.map(|(j, column)| {
    let (slope, square) = (column.dot(residuals), column.dot(&column));
    let onto = if slope > 0.0 {
      -room.down[j]
    } else {
      room.up[j]
    };
    let reaches = slope.abs() >= onto.abs() * square && onto.abs() <= room.strip[j];
    (slope != 0.0 && reaches).then_some(onto)
  });
  onto.collect()
}

#[cfg(test)]
thread_local! {
  /// Each point a solve stepped from, with the set held there: what a test reads of a fit.
  pub(crate) static HELD: RefCell<Vec<(Array1<f64>, Vec<bool>)>> =
    const { RefCell::new(Vec::new()) };
}

/// The bounds' side of a fit: the set held on its bounds through one solve, and what ends it.
pub(super) struct Bounds {
  rooms: Rooms,
  /// The coordinates on a bound, their columns zero: the scaling matrix diagonal in the set.
  active: RefCell<Vec<bool>>,
  /// The set and the cost at the iterate whose set is not `active`, which stopped the solve.
  change: RefCell<Option<(Vec<Option<f64>>, f64)>>,
  pub(super) watch: Cell<bool>,
  /// The iterate whose step onto the bounds raised the cost: nothing enters the set there.
  refused: RefCell<Option<Array1<f64>>>,
}

impl Bounds {
  pub(super) fn new(rooms: Rooms, dimension: usize) -> Self {
    Self {
      rooms,
      active: RefCell::new(vec![false; dimension]),
      change: RefCell::new(None),
      watch: Cell::new(true),
      refused: RefCell::new(None),
    }
  }

  /// Zeroes the active columns; `Err`, with the change kept, where the set at `x` is another.
  pub(super) fn hold(
    &self,
    x: &Array1<f64>,
    jacobian: &mut Array2<f64>,
    residuals: &Array1<f64>,
  ) -> Result<(), ()> {
    let active = self.active.borrow();
    if self.watch.get() {
      let mut set = cut(jacobian, residuals, &(self.rooms)(x.view()));
      if self.refused.borrow().as_ref() == Some(x) {
        let entering = set
          .iter_mut()
          .zip(active.iter())
          .filter(|(_, active)| !**active);
        entering.for_each(|(step, _)| *step = None);
      }
      if set
        .iter()
        .zip(active.iter())
        .any(|(step, active)| step.is_some() != *active)
      {
        *self.change.borrow_mut() = Some((set, residuals.dot(residuals)));
        return Err(());
      }
      #[cfg(test)]
      HELD.with_borrow_mut(|held| held.push((x.clone(), active.to_vec())));
    }
    let columns = jacobian.columns_mut().into_iter().zip(active.iter());
    columns
      .filter(|(_, active)| **active)
      .for_each(|(mut column, _)| column.fill(0.0));
    Ok(())
  }

  /// A sum can miss the bound by a rounding; the rest of the way is the nearer room, twice.
  fn land(&self, x: &mut Array1<f64>, j: usize) {
    for _ in 0..2 {
      let room = (self.rooms)(x.view());
      x[j] += if room.down[j] <= room.up[j] {
        -room.down[j]
      } else {
        room.up[j]
      };
    }
  }

  /// The fit's point folded, where a bound's own `f64` is a raw value, with each held coordinate
  /// landed on it: far from its box a raw value reaches a bound only to its own spacing.
  pub(super) fn settled<P: LeastSquaresProblem>(
    &self,
    objective: &Objective<P>,
    best: Array1<f64>,
  ) -> Array1<f64> {
    let mut settled = objective.at(&best).params();
    let active = self.active.borrow();
    let held = (0..settled.len()).filter(|j| active[*j]);
    held.for_each(|j| self.land(&mut settled, j));
    settled
  }

  /// `best` with each entering coordinate of `set` on its bound, the projection of (82), where
  /// that does not raise `cost`; `best` otherwise, with nothing entering. The rest is released.
  fn seat<P: LeastSquaresProblem>(
    &self,
    objective: &Objective<P>,
    best: Array1<f64>,
    set: &[Option<f64>],
    cost: f64,
  ) -> Array1<f64> {
    let mut active = self.active.borrow_mut();
    let folded = objective.at(&best).params();
    let mut seated = best.clone();
    let entering = |j: usize| !active[j] && set[j].is_some();
    for j in (0..set.len()).filter(|j| entering(*j)) {
      seated[j] += set[j].unwrap_or(0.0);
      self.land(&mut seated, j);
    }
    let on = |x: &Array1<f64>| Residual::residual(&objective, x).map(|r| r.dot(&r));
    // An ulp of every folded value moves the cost by `2‖r‖·ε‖Dθ‖`: a seat worse by no more than
    // twice that is taken, as (82)'s projection onto the bound.
    let slopes = objective.slopes.borrow();
    let columns = slopes
      .iter()
      .flat_map(|slopes| slopes.columns().into_iter().zip(&folded));
    let reach = columns.map(|(column, value)| column.dot(&column) * value * value);
    let allowance = 4.0 * cost.sqrt() * f64::EPSILON * reach.sum::<f64>().sqrt();
    drop(slopes);
    let kept = seated == best || on(&seated).is_ok_and(|on| on <= cost + allowance);
    for (j, active) in active.iter_mut().enumerate() {
      *active = set[j].is_some() && (kept || *active);
    }
    *self.refused.borrow_mut() = (!kept).then(|| best.clone());
    if kept && seated != best {
      objective.model.borrow_mut().accept();
    }
    if kept { seated } else { best }
  }
}

/// `solve` from `start`, and again from each iterate whose set is not the one held, its entering
/// coordinates seated; a restart spends an evaluation, so the budget bounds their number.
pub(super) fn restarted<P: LeastSquaresProblem>(
  objective: &Objective<P>,
  start: Array1<f64>,
  (budget, cap): (usize, usize),
  mut solve: impl FnMut(Array1<f64>, usize, usize, bool) -> Run,
) -> Run {
  let mut total = solve(start, budget, cap, true);
  let Some(bounds) = &objective.bounds else {
    return total;
  };
  if !total.started && bounds.change.borrow().is_some() {
    // A start whose own set stopped the solve was evaluated last, so the trail opens with it.
    objective.model.borrow_mut().accept();
  }
  loop {
    let own = matches!(
      total.reason,
      TerminationReason::SolverConverged | TerminationReason::NumericalNoProgress
    );
    let spent = || objective.evaluations.get();
    if own && bounds.change.borrow().is_none() && spent() < budget {
      // The solver stops on a step, before the Jacobian that tells the set where it stopped.
      let _ = Residual::residual(&objective, &total.best)
        .and_then(|_| Jacobian::jacobian(&objective, &total.best));
    }
    let Some((set, cost)) = bounds.change.borrow_mut().take() else {
      return total;
    };
    let out = |best: Array1<f64>| Run {
      best,
      reason: match spent() >= budget {
        true => TerminationReason::MaxCostEvals,
        false => TerminationReason::MaxIter,
      },
      iterations: total.iterations,
      started: true,
    };
    if spent() >= budget || total.iterations >= cap {
      return out(total.best);
    }
    let from = bounds.seat(objective, total.best.clone(), &set, cost);
    if spent() >= budget {
      return out(from);
    }
    let next = solve(from, budget - spent(), cap - total.iterations, false);
    total = Run {
      iterations: total.iterations + next.iterations,
      started: true,
      ..next
    };
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_distributions::RealExt;

  use super::super::LmOptions;
  use super::super::LmReport;
  use super::super::minimize_in_box;
  use super::*;
  use crate::calibration::stencil::room;

  /// `r = [θ₁ − a, θ₂ + θ₁ − 3]` with `θ₁ = |x₁|`, a fold at 0, and `θ₂ = x₂`.
  struct Folded {
    x: Array1<f64>,
    a: f64,
  }

  impl LeastSquaresProblem for Folded {
    fn set_params(&mut self, x: ArrayView1<f64>) {
      self.x.assign(&x);
    }

    fn params(&self) -> Array1<f64> {
      self.x.clone()
    }

    fn residuals(&mut self) -> Option<Array1<f64>> {
      let (first, second) = (self.x[0].abs(), self.x[1]);
      Some(ndarray::array![first - self.a, second + first - 3.0])
    }

    fn jacobian(&self) -> Option<Array2<f64>> {
      let slope = if self.x[0] < 0.0 { -1.0 } else { 1.0 };
      Some(ndarray::array![[slope, 0.0], [slope, 1.0]])
    }
  }

  fn outward(x: ArrayView1<f64>) -> Room {
    let boxed = ndarray::array![x[0].abs(), x[1]];
    let bounds = [(0.0, f64::INFINITY), (f64::NEG_INFINITY, f64::INFINITY)];
    let slopes = ndarray::array![if x[0] < 0.0 { -1.0 } else { 1.0 }, 1.0];
    room(&boxed, &bounds, slopes.view())
  }

  fn folded(x: [f64; 2], a: f64) -> Folded {
    Folded {
      x: Array1::from_vec(x.to_vec()),
      a,
    }
  }

  /// The minimum over `θ₁ ≥ 0` is `(0, 3)`: `θ₁` is set on the fold and `θ₂` fitted with it there.
  #[test]
  fn a_coordinate_whose_cost_falls_across_its_bound_ends_on_it_with_the_rest_fitted() {
    let (fitted, report) = minimize_in_box(folded([1.0, 0.0], -0.5), LmOptions::default(), outward);
    assert!(report.converged, "{report:?}");
    assert!(
      fitted.x[0] == 0.0 && (fitted.x[1] - 3.0).abs() < 1e-9,
      "{}",
      fitted.x
    );
  }

  /// From the fold itself the cost falls into the box: the column stays and the fit leaves.
  #[test]
  fn a_coordinate_on_its_bound_whose_cost_falls_inward_is_released() {
    let (fitted, report) = minimize_in_box(folded([0.0, 0.0], 0.5), LmOptions::default(), outward);
    assert!(report.converged, "{report:?}");
    let gap = (fitted.x[0].abs() - 0.5)
      .abs()
      .max((fitted.x[1] - 2.5).abs());
    assert!(gap < 1e-9, "{}", fitted.x);
  }

  /// Residuals of `θ`, each `θⱼ = capⱼ − |xⱼ − capⱼ|` folded at an upper bound, `∞` for none.
  struct Capped {
    x: Array1<f64>,
    caps: Vec<f64>,
    rows: fn(&[f64]) -> Array1<f64>,
    slopes: fn(&[f64]) -> Array2<f64>,
    /// The values of the last `residuals` call, and those `accept` kept.
    staged: Array1<f64>,
    trail: Vec<Array1<f64>>,
  }

  impl Capped {
    fn values(x: ArrayView1<f64>, caps: &[f64]) -> Array1<f64> {
      let folded = x.iter().zip(caps).map(|(x, cap)| match cap.is_finite() {
        true => cap - (x - cap).abs(),
        false => *x,
      });
      folded.collect()
    }

    fn folds(x: ArrayView1<f64>, caps: &[f64]) -> Array1<f64> {
      let sides = x.iter().zip(caps);
      sides
        .map(|(x, cap)| if x > cap { -1.0 } else { 1.0 })
        .collect()
    }
  }

  impl LeastSquaresProblem for Capped {
    fn set_params(&mut self, x: ArrayView1<f64>) {
      self.x.assign(&x);
    }

    fn params(&self) -> Array1<f64> {
      self.x.clone()
    }

    fn residuals(&mut self) -> Option<Array1<f64>> {
      self.staged = Self::values(self.x.view(), &self.caps);
      Some((self.rows)(self.staged.as_slice()?))
    }

    fn jacobian(&self) -> Option<Array2<f64>> {
      let values = Self::values(self.x.view(), &self.caps);
      let slopes = (self.slopes)(values.as_slice()?);
      let folds = Self::folds(self.x.view(), &self.caps);
      Some(crate::calibration::scale_columns(slopes, folds.view()))
    }

    fn accept(&mut self) {
      self.trail.push(self.staged.clone());
    }
  }

  /// The fit of `rows` from `start` under `caps`, on a budget of `patience·(N + 1)` evaluations.
  fn fit<const N: usize>(
    start: [f64; N],
    caps: [f64; N],
    rows: fn(&[f64]) -> Array1<f64>,
    slopes: fn(&[f64]) -> Array2<f64>,
    patience: usize,
  ) -> (Capped, LmReport) {
    let problem = Capped {
      x: Array1::from_vec(start.to_vec()),
      caps: caps.to_vec(),
      rows,
      slopes,
      staged: Array1::zeros(N),
      trail: Vec::new(),
    };
    let rooms = move |x: ArrayView1<f64>| {
      let bounds = caps.map(|cap| (f64::NEG_INFINITY, cap));
      let folds = Capped::folds(x, &caps);
      room(&Capped::values(x, &caps), &bounds, folds.view())
    };
    let options = LmOptions {
      patience,
      ..LmOptions::default()
    };
    minimize_in_box(problem, options, rooms)
  }

  /// [`fit`] on the default budget: its values, its cost and whether it converged.
  fn capped<const N: usize>(
    start: [f64; N],
    caps: [f64; N],
    rows: fn(&[f64]) -> Array1<f64>,
    slopes: fn(&[f64]) -> Array2<f64>,
  ) -> (Array1<f64>, f64, bool) {
    let (mut fitted, report) = fit(start, caps, rows, slopes, LmOptions::default().patience);
    let values = Capped::values(fitted.x.view(), &fitted.caps);
    let cost = fitted.residuals().map_or(f64::NAN, |r| r.dot(&r));
    (values, cost, report.converged)
  }

  const FREE: f64 = f64::INFINITY;

  /// 999.1 lies within the stencil step 1 of the bound, 0.9 from it, and 0.4 short of its minimum.
  #[test]
  fn a_coordinate_whose_minimum_lies_short_of_its_bound_is_free() {
    let rows = |v: &[f64]| ndarray::array![v[0] - 999.5, v[1] - 2.0];
    for start in [999.1, 900.0] {
      let (fitted, _, converged) = capped([start, 2.0], [1000.0, FREE], rows, |_| Array2::eye(2));
      assert!(
        converged && (fitted[0] - 999.5).abs() < 1e-9,
        "{start}: {fitted}"
      );
    }
  }

  /// Both capped at 1000: the first's minimum, 1001, lies beyond its bound and the second's,
  /// 999.5, short of it, inside the strip.
  #[test]
  fn a_coordinate_short_of_its_bound_is_fitted_whatever_another_one_holds() {
    let rows = |v: &[f64]| ndarray::array![v[0] - 1001.0, v[1] - 999.5, v[2] - 2.0];
    let want = ndarray::array![1000.0, 999.5, 2.0];
    for start in [
      [999.05, 999.1, 2.0],
      [999.95, 999.1, 2.0],
      [1000.0, 999.1, 2.0],
      [900.0, 900.0, 0.0],
    ] {
      let caps = [1000.0, 1000.0, FREE];
      let (fitted, cost, converged) = capped(start, caps, rows, |_| Array2::eye(3));
      let gap = f64::max_or_nan_fold((&fitted - &want).iter().map(|d| d.abs()));
      assert!(converged && gap < 1e-9, "{start:?}: {fitted}, cost {cost}");
    }
  }

  /// Rosenbrock's valley under `θ₁ ≤ 0.8`: the minimum is `(0.8, 0.64)`, at cost 0.04.
  #[test]
  fn the_free_coordinates_are_fitted_with_the_held_one_on_its_bound() {
    let rows = |v: &[f64]| ndarray::array![10.0 * (v[1] - v[0] * v[0]), 1.0 - v[0]];
    let slopes = |v: &[f64]| ndarray::array![[-20.0 * v[0], 10.0], [-1.0, 0.0]];
    for start in [[-1.2, 1.0], [0.7998, 0.3]] {
      let (fitted, cost, converged) = capped(start, [0.8, FREE], rows, slopes);
      assert!(converged, "{start:?}: {fitted}, cost {cost}");
      assert!(
        fitted[0] == 0.8 && (fitted[1] - 0.64).abs() < 1e-9,
        "{start:?}: {fitted}"
      );
      assert!((cost - 0.04).abs() < 1e-12, "{start:?}: cost {cost}");
    }
  }

  /// The first start is cut where it stands and the second is the minimum, held on its bound.
  #[test]
  fn a_start_on_its_bound_opens_the_trail() {
    let rows = |v: &[f64]| ndarray::array![10.0 * (v[1] - v[0] * v[0]), 1.0 - v[0]];
    let slopes = |v: &[f64]| ndarray::array![[-20.0 * v[0], 10.0], [-1.0, 0.0]];
    for start in [[0.8, 0.9], [0.8, 0.64]] {
      let (fitted, _) = fit(start, [0.8, FREE], rows, slopes, 100);
      let opens = fitted.trail.first().map(|first| first.to_vec());
      assert_eq!(opens, Some(start.to_vec()), "{:?}", fitted.trail);
    }
  }

  /// Rosenbrock under `θ₁ ≤ 0.8` on 21 and 12 evaluations, each fit with a seat near its end.
  #[test]
  fn a_seat_is_evaluated_out_of_the_budget() {
    let rows = |v: &[f64]| ndarray::array![10.0 * (v[1] - v[0] * v[0]), 1.0 - v[0]];
    let slopes = |v: &[f64]| ndarray::array![[-20.0 * v[0], 10.0], [-1.0, 0.0]];
    for (start, patience) in [([-1.2, 1.0], 7), ([0.8, 0.3], 4)] {
      let (_, report) = fit(start, [0.8, FREE], rows, slopes, patience);
      assert_eq!(report.evaluations, 3 * patience, "{start:?}: {report:?}");
    }
  }

  /// On the bound itself the cost jumps by more than it falls towards it: the minimum is short
  /// of 1000, and the step onto the bound is refused.
  #[test]
  fn a_step_onto_a_bound_that_raises_the_cost_is_not_taken() {
    let rows = |v: &[f64]| {
      let cliff = if v[0] >= 1000.0 { 5.0 } else { 0.0 };
      ndarray::array![v[0] - 1001.0 + cliff, v[1] - 2.0]
    };
    let (fitted, cost, _) = capped([999.5, 0.0], [1000.0, FREE], rows, |_| Array2::eye(2));
    assert!(
      fitted[0] < 1000.0 && cost < 1.0 + 1e-9,
      "{fitted}, cost {cost}"
    );
  }

  /// One ulp short of a bound the cost falls across, where the cost on the bound reads `8ε`
  /// higher, 0.9 of the allowance `4‖r‖·ε‖Dθ‖ = 4√5·ε`: the seat is taken.
  #[test]
  fn a_seat_whose_cost_is_worse_by_its_rounding_alone_is_taken() {
    let rows = |v: &[f64]| {
      let bit = if v[0] >= 1.0 { 4.0 * f64::EPSILON } else { 0.0 };
      ndarray::array![2.0 - v[0] + bit, v[1] - 2.0]
    };
    let slopes = |_: &[f64]| ndarray::array![[-1.0, 0.0], [0.0, 1.0]];
    let (fitted, _, _) = capped([1.0 - f64::EPSILON / 2.0, 2.0], [1.0, FREE], rows, slopes);
    assert_eq!(fitted[0], 1.0, "{fitted}");
  }

  /// The same seat at a rise of `32ε`, 3.6 times the allowance, is refused.
  #[test]
  fn a_seat_whose_cost_is_worse_by_a_few_allowances_is_refused() {
    let rows = |v: &[f64]| {
      let bit = if v[0] >= 1.0 {
        16.0 * f64::EPSILON
      } else {
        0.0
      };
      ndarray::array![2.0 - v[0] + bit, v[1] - 2.0]
    };
    let slopes = |_: &[f64]| ndarray::array![[-1.0, 0.0], [0.0, 1.0]];
    let (fitted, _, _) = capped([1.0 - f64::EPSILON / 2.0, 2.0], [1.0, FREE], rows, slopes);
    assert!(fitted[0] < 1.0, "{fitted}");
  }

  /// Kou's `p`'s box: `θ₁` is reflected into it, as a Lévy fit folds a raw coordinate.
  const REFLECTED: (f64, f64) = (0.01, 0.99);

  /// `r = [θ₁ − 1000.97, θ₂ − 3]` with `θ₁` reflected into its box and `θ₂ = x₂`; `reach` is the
  /// largest raw `|x₁|` the fit has set.
  struct Reflected {
    theta: Array1<f64>,
    slope: f64,
    reach: f64,
  }

  impl LeastSquaresProblem for Reflected {
    fn set_params(&mut self, x: ArrayView1<f64>) {
      let (lo, hi) = REFLECTED;
      self.theta = ndarray::array![crate::calibration::periodic_map(x[0], lo, hi), x[1]];
      self.slope = crate::calibration::periodic_slope(x[0], lo, hi);
      self.reach = self.reach.max(x[0].abs());
    }

    fn params(&self) -> Array1<f64> {
      self.theta.clone()
    }

    fn residuals(&mut self) -> Option<Array1<f64>> {
      Some(ndarray::array![
        self.theta[0] - 1000.97,
        self.theta[1] - 3.0
      ])
    }

    fn jacobian(&self) -> Option<Array2<f64>> {
      Some(ndarray::array![[self.slope, 0.0], [0.0, 1.0]])
    }
  }

  /// The first step sends `x₁` a thousand periods out, where its fold reaches the bound only to
  /// its own spacing: the held `θ₁` is reported as 0.99 itself, on every libm.
  #[test]
  fn a_coordinate_held_on_its_bound_from_far_outside_its_box_is_reported_on_it() {
    let rooms = |x: ArrayView1<f64>| {
      let (lo, hi) = REFLECTED;
      let folded = ndarray::array![crate::calibration::periodic_map(x[0], lo, hi), x[1]];
      let slopes = ndarray::array![crate::calibration::periodic_slope(x[0], lo, hi), 1.0];
      room(
        &folded,
        &[REFLECTED, (f64::NEG_INFINITY, FREE)],
        slopes.view(),
      )
    };
    let problem = Reflected {
      theta: ndarray::array![0.5, 1e4],
      slope: 1.0,
      reach: 0.0,
    };
    let (fitted, report) = minimize_in_box(problem, LmOptions::default(), rooms);
    assert!(
      report.converged && fitted.reach > 1000.0,
      "{report:?}, {}",
      fitted.reach
    );
    assert_eq!(fitted.theta[0], 0.99, "{}", fitted.theta);
    assert!((fitted.theta[1] - 3.0).abs() < 1e-9, "{}", fitted.theta);
  }

  /// The first rests 1e-6 from a bound the cost falls across: its step is 1 and then 2e-7.
  #[test]
  fn a_coordinate_is_in_the_set_only_where_its_own_step_reaches_its_bound() {
    let jacobian = ndarray::array![[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]];
    let room = Room {
      down: ndarray::array![1e-6, 1.0, 1.0],
      up: ndarray::array![1.0, 1.0, 1.0],
      strip: ndarray::array![1e-5, 1e-5, 1e-5],
    };
    let far = cut(&jacobian, &ndarray::array![1.0, 0.0], &room);
    assert_eq!(far, [Some(-1e-6), None, None]);
    let near = cut(&jacobian, &ndarray::array![2e-7, 0.0], &room);
    assert_eq!(near, [None, None, None]);
  }
}
