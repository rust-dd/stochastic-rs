//! Optimization helpers shared by the estimator modules.
//!
//! One Nelder–Mead simplex: the three-parameter entry wraps the any-dimension one GARCH uses.

use std::convert::Infallible;

use basin::BoxBounds;
use basin::ConstraintSet;
use basin::ConstraintSolveError;
use basin::Constraints;
use basin::CostFunction;
use basin::ExecutionLimit;
use basin::Executor;
use basin::FirstOrderState;
use basin::Gradient;
use basin::Lbfgsb;
use basin::LineSearch;
use basin::MoreThuente;
use basin::Solver;
use basin::Stall;
use basin::Termination;

use crate::traits::RealExt;

pub(crate) mod certificate;

/// The start with the lowest objective; a NaN objective loses to every number.
pub(crate) fn best_start<P: AsRef<[f64]>>(
  starts: impl IntoIterator<Item = P>,
  objective: impl Fn(&[f64]) -> f64,
) -> Option<P> {
  starts
    .into_iter()
    .map(|start| (objective(start.as_ref()), start))
    .min_by(|a, b| a.0.nan_last(&b.0))
    .map(|(_, start)| start)
}

/// Why a basin search stopped; `Converged` alone is one of the solver's own tolerances.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SolverStop {
  Converged,
  IterationCap,
  EvaluationBudget,
  NoProgress,
  Failed,
  Other,
}

impl From<&Termination> for SolverStop {
  fn from(termination: &Termination) -> Self {
    match termination {
      Termination::Converged(_) => Self::Converged,
      Termination::Limit(limits) => match limits.first() {
        Some(ExecutionLimit::Iterations { .. }) => Self::IterationCap,
        Some(
          ExecutionLimit::CostEvaluations { .. }
          | ExecutionLimit::GradientEvaluations { .. }
          | ExecutionLimit::Evaluations { .. },
        ) => Self::EvaluationBudget,
        _ => Self::Other,
      },
      Termination::Stalled(Stall::Numerical { .. }) => Self::NoProgress,
      Termination::Failed(_) => Self::Failed,
      _ => Self::Other,
    }
  }
}

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

/// Nelder–Mead minimiser of a three-parameter objective: `(argmin, iterations, converged)`.
/// `max_iter` bounds a Monte-Carlo objective, whose resampling kinks never reach the tolerance.
pub(crate) fn nelder_mead<F: Fn(&[f64; 3]) -> f64>(
  start: [f64; 3],
  max_iter: usize,
  f: F,
) -> ([f64; 3], usize, bool) {
  let (p, iters, converged) = nelder_mead_vec(&start, max_iter, |p| f(&[p[0], p[1], p[2]]));
  ([p[0], p[1], p[2]], iters, converged)
}

/// Nelder–Mead minimiser in any dimension, with [`nelder_mead`]'s coefficients and `1e-10` spread
/// tolerance; the initial simplex offsets each coordinate by `0.1`.
pub(crate) fn nelder_mead_vec<F: Fn(&[f64]) -> f64>(
  start: &[f64],
  max_iter: usize,
  f: F,
) -> (Vec<f64>, usize, bool) {
  const ALPHA: f64 = 1.0;
  const GAMMA: f64 = 2.0;
  const RHO: f64 = 0.5;
  const SHRINK: f64 = 0.5;
  const TOL: f64 = 1e-10;

  let n = start.len();
  let mut simplex: Vec<Vec<f64>> = vec![start.to_vec(); n + 1];
  for i in 0..n {
    simplex[i + 1][i] += 0.1;
  }
  let mut fvals: Vec<f64> = simplex.iter().map(|p| f(p)).collect();

  let mut iters = 0;
  while iters < max_iter {
    iters += 1;
    let mut order: Vec<usize> = (0..=n).collect();
    order.sort_by(|&a, &b| fvals[a].nan_last(&fvals[b]));
    let best = order[0];
    let worst = order[n];
    let second_worst = order[n - 1];

    if (fvals[worst] - fvals[best]).abs() < TOL {
      return (simplex[best].clone(), iters, true);
    }

    let mut centroid = vec![0.0; n];
    for &o in &order[..n] {
      for d in 0..n {
        centroid[d] += simplex[o][d] / n as f64;
      }
    }

    let reflect = combine(&centroid, &simplex[worst], ALPHA, true);
    let f_reflect = f(&reflect);

    if f_reflect < fvals[best] {
      let expand = combine(&centroid, &simplex[worst], ALPHA * GAMMA, true);
      let f_expand = f(&expand);
      if f_expand < f_reflect {
        simplex[worst] = expand;
        fvals[worst] = f_expand;
      } else {
        simplex[worst] = reflect;
        fvals[worst] = f_reflect;
      }
    } else if f_reflect < fvals[second_worst] {
      simplex[worst] = reflect;
      fvals[worst] = f_reflect;
    } else {
      let contract = combine(&centroid, &simplex[worst], RHO, false);
      let f_contract = f(&contract);
      if f_contract < fvals[worst] {
        simplex[worst] = contract;
        fvals[worst] = f_contract;
      } else {
        let anchor = simplex[best].clone();
        for &o in &order[1..] {
          for d in 0..n {
            simplex[o][d] = anchor[d] + SHRINK * (simplex[o][d] - anchor[d]);
          }
          fvals[o] = f(&simplex[o]);
        }
      }
    }
  }
  let best = (0..=n)
    .min_by(|&a, &b| fvals[a].nan_last(&fvals[b]))
    .expect("a simplex has n + 1 vertices");
  (simplex[best].clone(), iters, false)
}

/// Reflection (`reflect = true`) or contraction (`reflect = false`) of the
/// worst vertex through the centroid with the given coefficient.
fn combine(centroid: &[f64], worst: &[f64], coef: f64, reflect: bool) -> Vec<f64> {
  centroid
    .iter()
    .zip(worst)
    .map(|(c, w)| {
      if reflect {
        c + coef * (c - w)
      } else {
        c + coef * (w - c)
      }
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use basin::ApplicationStop;
  use basin::ConstraintDescription;
  use basin::ConvergenceCriterion;
  use basin::ConvergenceTest;
  use basin::EvaluationKind;

  use super::*;

  #[test]
  fn each_basin_termination_maps_to_its_stop() {
    let iterations = ExecutionLimit::Iterations {
      observed: 200,
      limit: 200,
    };
    let costs = ExecutionLimit::CostEvaluations {
      observed: 50,
      limit: 50,
    };
    let residuals = ExecutionLimit::Evaluations {
      kind: EvaluationKind::Residual,
      observed: 50,
      limit: 50,
    };
    let test = ConvergenceTest::ProjectedGradient;
    let met = ConvergenceCriterion::upper_bound(test, 0.0, 1e-8, 1e-8, None);
    let table = [
      (Termination::converged(met), SolverStop::Converged),
      (
        Termination::Limit(vec![iterations.clone()]),
        SolverStop::IterationCap,
      ),
      (
        Termination::Limit(vec![costs.clone()]),
        SolverStop::EvaluationBudget,
      ),
      (
        Termination::Limit(vec![residuals]),
        SolverStop::EvaluationBudget,
      ),
      (
        Termination::Limit(vec![iterations, costs]),
        SolverStop::IterationCap,
      ),
      (
        Termination::numerical_stall("no step"),
        SolverStop::NoProgress,
      ),
      (
        Termination::numerical_failure("no value"),
        SolverStop::Failed,
      ),
      (Termination::Cancelled, SolverStop::Other),
      (
        Termination::Application(ApplicationStop::new("asked")),
        SolverStop::Other,
      ),
    ];
    for (termination, stop) in table {
      assert_eq!(SolverStop::from(&termination), stop, "{termination:?}");
    }
  }

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

  /// Arithmetic leaves a NaN's sign unspecified (x86 sets it), so both signs must lose.
  #[test]
  fn a_nan_objective_never_wins_the_start_search() {
    for nan in [f64::NAN, -f64::NAN] {
      let objective = |p: &[f64]| if p[0] == 1.0 { 3.0 } else { nan };
      assert_eq!(best_start([[0.0], [1.0], [2.0]], objective), Some([1.0]));
      assert_eq!(
        nan.nan_last(&f64::NEG_INFINITY),
        std::cmp::Ordering::Greater
      );
    }
  }

  /// A NaN vertex ranks worst, so the simplex reflects away from it instead of shrinking onto it.
  #[test]
  fn the_simplex_leaves_a_nan_vertex() {
    for nan in [f64::NAN, -f64::NAN] {
      let objective = |p: &[f64]| {
        if p[0] < 0.05 {
          nan
        } else {
          (p[0] - 1.0).powi(2)
        }
      };
      let (argmin, _, converged) = nelder_mead_vec(&[0.0], 500, objective);
      assert!(converged && objective(&argmin) < 0.02, "{nan}: {argmin:?}");
    }
  }
}
