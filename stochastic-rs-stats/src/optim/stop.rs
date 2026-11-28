//! Why a search stopped, in the classes a fit's report tells apart, and the words for each.

use basin::ExecutionLimit;
use basin::Stall;
use basin::Termination;

/// Why a basin search stopped; `Converged` alone is one of the solver's own tolerances.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverStop {
  Converged,
  IterationCap,
  EvaluationBudget,
  NoProgress,
  Failed,
  Other,
}

impl From<&Termination> for SolverStop {
  /// Limits that trip together are classed by the first: the iteration cap ahead of a budget.
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

impl SolverStop {
  /// The stop in the crates' words: what a fit's `message` says where no check words it otherwise.
  pub fn words(self) -> &'static str {
    match self {
      Self::Converged => "converged",
      Self::IterationCap | Self::EvaluationBudget => "evaluation budget exhausted",
      Self::NoProgress => "no numerical progress",
      Self::Failed => "solver failed",
      Self::Other => "stopped without convergence",
    }
  }
}

#[cfg(test)]
mod tests {
  use basin::ApplicationStop;
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

  #[test]
  fn a_least_squares_termination_maps_to_its_stop() {
    let iterations = ExecutionLimit::Iterations {
      observed: 600,
      limit: 600,
    };
    let residuals = ExecutionLimit::Evaluations {
      kind: EvaluationKind::Residual,
      observed: 600,
      limit: 600,
    };
    let costs = ExecutionLimit::CostEvaluations {
      observed: 600,
      limit: 600,
    };
    let test = ConvergenceTest::GradientOrthogonality;
    let met = ConvergenceCriterion::upper_bound(test, 0.0, 1e-14, 1e-14, None);
    let table = [
      (Termination::converged(met), SolverStop::Converged),
      (
        Termination::Limit(vec![iterations.clone()]),
        SolverStop::IterationCap,
      ),
      (
        Termination::Limit(vec![residuals.clone()]),
        SolverStop::EvaluationBudget,
      ),
      (
        Termination::Limit(vec![costs]),
        SolverStop::EvaluationBudget,
      ),
      (
        Termination::Limit(vec![iterations, residuals]),
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

  #[test]
  fn each_stop_has_its_words() {
    let said = [
      (SolverStop::Converged, "converged"),
      (SolverStop::IterationCap, "evaluation budget exhausted"),
      (SolverStop::EvaluationBudget, "evaluation budget exhausted"),
      (SolverStop::NoProgress, "no numerical progress"),
      (SolverStop::Failed, "solver failed"),
      (SolverStop::Other, "stopped without convergence"),
    ];
    for (stop, words) in said {
      assert_eq!(stop.words(), words);
    }
  }
}
