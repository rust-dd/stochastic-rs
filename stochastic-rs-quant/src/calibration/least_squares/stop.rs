use basin::ExecutionLimit;
use basin::Stall;
use basin::Termination;

/// Why a solve stopped, in the classes a fit's report and its restarts tell apart.
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
}
