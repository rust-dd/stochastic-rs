//! The L-BFGS searches the estimators share: the bounded one, and the unbounded polish.

use std::convert::Infallible;

use basin::BoxBounds;
use basin::ConstraintSet;
use basin::ConstraintSolveError;
use basin::Constraints;
use basin::CostFunction;
use basin::Executor;
use basin::FirstOrderState;
use basin::Gradient;
use basin::Lbfgsb;
use basin::LineSearch;
use basin::MoreThuente;
use basin::Solver;

/// Moré–Thuente line search with Nocedal–Wright's quasi-Newton constants `c1 = 1e-4`, `c2 = 0.9`.
pub(crate) fn more_thuente() -> MoreThuente<f64> {
  MoreThuente::new()
    .with_sufficient_decrease_coefficient(1e-4)
    .with_curvature_coefficient(0.9)
    .with_relative_bracket_tolerance(1e-10)
    .stpmin(f64::EPSILON.sqrt())
    .stpmax(f64::INFINITY)
}

/// Curvature pairs an L-BFGS search keeps.
const HISTORY: usize = 10;

/// L-BFGS-B stopping on a `√ε` projected gradient or an `ε` cost change alone: its relative
/// cost-reduction stop, the looser of the two cost tests, is off.
pub(crate) fn lbfgsb<P, S>(
  line_search: S,
) -> impl Solver<P, FirstOrderState<Vec<f64>>, Error = ConstraintSolveError<P::Error>>
where
  P: CostFunction<Param = Vec<f64>, Output = f64>
    + Gradient<Gradient = Vec<f64>>
    + Constraints<Set = ConstraintSet<BoxBounds<Vec<f64>>>>,
  S: LineSearch<P, Vec<f64>, Error = P::Error>,
{
  Lbfgsb::<Vec<f64>, f64, S>::with_line_search(line_search)
    .with_m_capacity(HISTORY)
    .with_absolute_projected_gradient_tolerance(f64::EPSILON.sqrt())
    .with_relative_cost_reduction_tolerance(None)
    .with_absolute_cost_change_tolerance(f64::EPSILON)
}

/// Unbounded L-BFGS from `theta`, 200 iterations at most, to a `√ε` gradient or an `ε` cost
/// change: the best point met and the iterations, `theta` itself unless that point costs less.
pub(crate) fn polish<P>(problem: P, theta: Vec<f64>) -> (Vec<f64>, usize)
where
  P: CostFunction<Param = Vec<f64>, Output = f64, Error = Infallible>
    + Gradient<Gradient = Vec<f64>>,
{
  let Ok(start_cost) = problem.cost(&theta);
  let solver = Lbfgsb::<Vec<f64>, f64, _>::with_line_search(more_thuente())
    .with_m_capacity(HISTORY)
    .unbounded()
    .with_absolute_gradient_tolerance(f64::EPSILON.sqrt())
    .with_absolute_cost_change_tolerance(f64::EPSILON);
  let state = FirstOrderState::new(theta.clone());
  let Ok(result) = Executor::new(problem, solver, state).max_iter(200).run();
  let iters = result.iter() as usize;
  match result.state.best() {
    Some((best, cost)) if cost < start_cost => (best.clone(), iters),
    _ => (theta, iters),
  }
}

#[cfg(test)]
mod tests {
  use basin::ConstraintDescription;
  use basin::ConvergenceTest;
  use basin::Termination;

  use super::*;

  /// `10⁶ + ½·Σ 3ⁱ(xᵢ − 1)²` over a box that holds `x₀` at `0.5`: near the minimum a step lowers
  /// the cost by less than `10ε·10⁶` while the gradient is still large.
  struct Offset {
    low: Vec<f64>,
    high: Vec<f64>,
  }

  impl Offset {
    fn weights() -> impl Iterator<Item = f64> {
      (0..6).map(|i| 3f64.powi(i))
    }
  }

  impl CostFunction for Offset {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
      let bowl = Self::weights().zip(x).map(|(w, x)| w * (x - 1.0).powi(2));
      Ok(1e6 + 0.5 * bowl.sum::<f64>())
    }
  }

  impl Gradient for Offset {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
      Ok(Self::weights().zip(x).map(|(w, x)| w * (x - 1.0)).collect())
    }
  }

  impl Constraints for Offset {
    type Set = ConstraintSet<BoxBounds<Vec<f64>>>;

    fn constraints(&self) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
      ConstraintSet::new().with_bounds(BoxBounds::new(Some(&self.low), Some(&self.high)))
    }
  }

  #[test]
  fn the_bounded_search_stops_on_its_gradient_or_cost_change_test_alone() {
    let problem = Offset {
      low: vec![0.0; 6],
      high: vec![0.5, 2.0, 2.0, 2.0, 2.0, 2.0],
    };
    let start = (0..6).map(|i| 0.1 + 0.3 * i as f64).collect::<Vec<_>>();
    let result = Executor::new(problem, lbfgsb(more_thuente()), FirstOrderState::new(start))
      .max_iter(200)
      .run()
      .unwrap();
    let Termination::Converged(met) = &result.report.termination else {
      panic!("{:?}", result.report.termination);
    };
    let ours = [
      ConvergenceTest::BoundClippedGradient,
      ConvergenceTest::AbsoluteCostChange,
    ];
    let mut tests = met.criteria().iter().map(|criterion| &criterion.test);
    assert!(tests.all(|test| ours.contains(test)), "{met:?}");
  }
}
