use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::LmOptions;
use crate::calibration::least_squares::LmReport;
use crate::calibration::least_squares::minimize;

/// Restarts of a fit that stopped on a saturated coordinate the cost pulls inward, each from
/// inside the box; the Nelder–Mead driver's count.
const MAX_RESEATS: usize = 3;

/// Within this of an end of its box a coordinate's logistic slope is under 1e-3 of the box; the
/// same bound on its orthogonality measure tells a stop on the box from one against it.
const SATURATED: f64 = 1e-3;

/// Where a saturated coordinate restarts: the slope there is 0.005 of the box.
const RESEAT: f64 = 0.005;

/// `units`, each coordinate's place in its box, with every saturated one whose descent direction
/// points into the box moved `RESEAT` inside; the first such coordinate, `None` when none is.
pub(crate) fn reseat(units: &mut [f64], cosines: &[f64]) -> Option<usize> {
  let mut first = None;
  for (j, unit) in units.iter_mut().enumerate() {
    let (low, high) = (*unit < SATURATED, *unit > 1.0 - SATURATED);
    // The cost falls along −Jⱼᵀr, which is inward for a negative cosine at the lower end.
    let inward = if low { -cosines[j] } else { cosines[j] };
    if (low || high) && inward > SATURATED {
      *unit = if low { RESEAT } else { 1.0 - RESEAT };
      first.get_or_insert(j);
    }
  }
  first
}

/// Why a re-seated fit stopped short: still on the bound of a coordinate, or out of iterations.
pub(crate) enum Unseated {
  Bound(usize),
  Budget,
}

impl Unseated {
  /// The fit's message, in the words of [`LmReport::message`] for an exhausted budget.
  pub(crate) fn message(&self, names: &[&str]) -> String {
    match self {
      Unseated::Bound(j) => format!("stopped at the bound of {}", names[*j]),
      Unseated::Budget => "evaluation budget exhausted".to_string(),
    }
  }
}

/// `minimize`, restarted from `reseated`'s start up to `MAX_RESEATS` times on one iteration budget;
/// the last pass, its report, every pass's iterations and why it stopped short, if it did.
pub(crate) fn minimize_reseated<P: LeastSquaresProblem, S>(
  problem: P,
  lm: &LmOptions,
  mut reseated: impl FnMut(&mut P) -> Option<(S, usize)>,
  mut restart: impl FnMut(P, S) -> P,
) -> (P, LmReport, usize, Option<Unseated>) {
  let cap = lm.iteration_cap(problem.params().len());
  let (mut fitted, mut report) = minimize(problem, lm.clone());
  let (mut iterations, mut restarts) = (report.iterations, 0);
  loop {
    let bound = report
      .solver_converged()
      .then(|| reseated(&mut fitted))
      .flatten();
    let Some((start, coordinate)) = bound else {
      return (fitted, report, iterations, None);
    };
    let left = cap.saturating_sub(iterations);
    if left == 0 {
      return (fitted, report, iterations, Some(Unseated::Budget));
    }
    if restarts == MAX_RESEATS {
      return (
        fitted,
        report,
        iterations,
        Some(Unseated::Bound(coordinate)),
      );
    }
    let rest = LmOptions {
      max_iterations: Some(left),
      ..lm.clone()
    };
    (fitted, report) = minimize(restart(fitted, start), rest);
    iterations += report.iterations;
    restarts += 1;
  }
}
