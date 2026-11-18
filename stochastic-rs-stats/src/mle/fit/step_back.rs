use basin::CostFunction;
use basin::Gradient;
use basin::LineSearch;
use basin::LineSearchBounds;
use basin::LineSearchEvaluation;
use basin::LineSearchOutcome;
use basin::LineSearchResult;
use basin::MoreThuente;
use basin::Problem;

/// An objective that reports whether a line search met a point where it has no value.
pub(super) trait Walled {
  /// Starts (`true`) or ends watching; while watching, every evaluation after such a point may
  /// return the penalty unevaluated.
  fn watch(&self, on: bool);

  fn walled(&self) -> bool;
}

/// Moré–Thuente that steps back: a search that met a point without a value runs again under a
/// step cap, halved until none is met, so no interpolation reads the penalty.
/// Precedent: basin 1.11's `HagerZhang`, whose non-finite trials "cause the search to contract".
pub(super) struct SteppingBack(pub(super) MoreThuente<f64>);

/// Halvings of the cap, which leave a step no rounding distinguishes from none.
const HALVINGS: i32 = 52;

impl<P> LineSearch<P, Vec<f64>, f64> for SteppingBack
where
  P: CostFunction<Param = Vec<f64>, Output = f64> + Gradient<Gradient = Vec<f64>> + Walled,
{
  type Error = P::Error;

  fn next(
    &mut self,
    problem: &mut Problem<P>,
    param: &Vec<f64>,
    cost: f64,
    gradient: &Vec<f64>,
    direction: &Vec<f64>,
  ) -> Result<f64, Self::Error> {
    self.0.next(problem, param, cost, gradient, direction)
  }

  fn next_with_bounds(
    &mut self,
    problem: &mut Problem<P>,
    param: &Vec<f64>,
    cost: f64,
    gradient: &Vec<f64>,
    direction: &Vec<f64>,
    bounds: LineSearchBounds<f64>,
  ) -> Result<LineSearchResult<Vec<f64>, f64>, Self::Error> {
    let mut run = |problem: &mut Problem<P>, bounds| {
      problem.inner().watch(true);
      let result = self
        .0
        .next_with_bounds(problem, param, cost, gradient, direction, bounds);
      let walled = problem.inner().walled();
      problem.inner().watch(false);
      result.map(|result| (!walled).then_some(result))
    };
    if let Some(result) = run(problem, bounds)? {
      return Ok(result);
    }
    for halving in 0..=HALVINGS {
      let cap = bounds.initial() * 0.5_f64.powi(halving);
      if let Some(result) = run(problem, LineSearchBounds::new(cap, cap))? {
        // A capped step that does not lower the cost would read as the solver's cost tolerance:
        // the wall stopped it, not a flat likelihood, so it is no step.
        let gained = |taken: &LineSearchEvaluation<Vec<f64>, f64>| cost - taken.cost > f64::EPSILON;
        if result.evaluation.as_ref().is_some_and(gained) {
          return Ok(result);
        }
        break;
      }
    }
    Ok(LineSearchResult::new(LineSearchOutcome::Failed))
  }
}

#[cfg(test)]
mod tests {
  use std::cell::Cell;
  use std::convert::Infallible;

  use super::*;
  use crate::optim::more_thuente;

  /// `(x − 3)²` up to `x = 1` and the penalty past it; `seen` holds each value handed out.
  struct Cliff {
    watching: Cell<bool>,
    met: Cell<bool>,
    seen: std::cell::RefCell<Vec<f64>>,
  }

  const PENALTY: f64 = 1e100;

  impl Cliff {
    fn value(&self, x: f64) -> f64 {
      let value = if x > 1.0 || self.met.get() {
        self.met.set(self.watching.get());
        PENALTY
      } else {
        (x - 3.0).powi(2)
      };
      self.seen.borrow_mut().push(value);
      value
    }
  }

  impl Walled for Cliff {
    fn watch(&self, on: bool) {
      self.watching.set(on);
      self.met.set(false);
    }

    fn walled(&self) -> bool {
      self.met.get()
    }
  }

  impl CostFunction for Cliff {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
      Ok(self.value(x[0]))
    }
  }

  impl Gradient for Cliff {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
      Ok(vec![
        (self.value(x[0] + 1e-6) - self.value(x[0] - 1e-6)) / 2e-6,
      ])
    }
  }

  fn cliff() -> Problem<Cliff> {
    Problem::new(Cliff {
      watching: Cell::new(false),
      met: Cell::new(false),
      seen: std::cell::RefCell::new(Vec::new()),
    })
  }

  /// From 0 along `+6` the unit step lands at 6, past the cliff at 1: the search halves its cap to
  /// `1/8`, where every probe has a value, and returns a step whose point and slope have one.
  #[test]
  fn a_search_past_a_point_without_a_value_steps_back_to_one_with_it() {
    let mut problem = cliff();
    let bounds = LineSearchBounds::new(1.0, 10.0);
    let (at, slope, along) = (vec![0.0], vec![-6.0], vec![6.0]);
    let result = SteppingBack(more_thuente())
      .next_with_bounds(&mut problem, &at, 9.0, &slope, &along, bounds)
      .unwrap();
    assert_eq!(result.outcome, LineSearchOutcome::Step(0.125));
    let taken = result.evaluation.expect("the accepted point's values");
    assert_eq!(taken.param, vec![0.75]);
    assert!((taken.cost - 5.0625).abs() < 1e-12 && taken.gradient[0] < 0.0);
    // The accepted run starts after the last penalty: nothing in it was read from the constant.
    let seen = problem.inner().seen.borrow();
    let last_penalty = seen.iter().rposition(|&v| v == PENALTY).unwrap();
    assert_eq!(seen.len() - last_penalty - 1, 3, "{seen:?}");
  }

  /// The plain search reads the penalty into its interpolation: its step is `1.5e-4` for `1/8`.
  #[test]
  fn the_plain_search_interpolates_through_the_penalty() {
    let mut problem = cliff();
    let bounds = LineSearchBounds::new(1.0, 10.0);
    let (at, slope, along) = (vec![0.0], vec![-6.0], vec![6.0]);
    let result = more_thuente()
      .next_with_bounds(&mut problem, &at, 9.0, &slope, &along, bounds)
      .unwrap();
    let LineSearchOutcome::Step(step) = result.outcome else {
      panic!("no step")
    };
    assert!(step < 1e-3, "step {step}");
  }

  /// With no value anywhere along the direction there is no step, and the solver ends unconverged.
  #[test]
  fn a_direction_without_any_value_is_a_failed_search() {
    let mut problem = cliff();
    let bounds = LineSearchBounds::new(1.0, 10.0);
    let (at, slope, along) = (vec![1.0], vec![-4.0], vec![4.0]);
    let result = SteppingBack(more_thuente())
      .next_with_bounds(&mut problem, &at, 4.0, &slope, &along, bounds)
      .unwrap();
    assert!(!matches!(result.outcome, LineSearchOutcome::Step(step) if step > 0.0));
  }
}
