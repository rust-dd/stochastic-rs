//! The estimators' one search: the simplex from the best start, a finisher on its point, the
//! first-order check, and the fit at each limit of the family: together they word the answer.

use std::convert::Infallible;

use basin::CostFunction;
use basin::Gradient;

use super::best_start;
use super::certificate::Axis;
use super::certificate::Stop;
use super::certificate::beside;
use super::polish;
use super::simplex::SimplexStop;
use super::simplex::Spread;
use super::simplex::run_nelder_mead_within;

/// What an objective answers where it has no value, above every sum it returns.
pub(crate) const NO_VALUE: f64 = 1e300;

/// Simplex iterations at most, as the estimators' searches have had.
const CAP: u64 = 5_000;

/// The simplex's first step along a coordinate, in its scale.
const STEP: f64 = 0.1;

/// The spread of the vertex costs a simplex stops at: with no finisher after it, an estimate
/// within `√(2e-11)`, `4.5e-6`, of a standard error of the minimum.
const SPREAD: f64 = 1e-11;

/// Searches from the point of a gain the check found, `fit_mle`'s count.
const RESEATS: usize = 3;

/// `e`-folds the logarithm of a scale may run from its centre: past `ln(1/ε)` the scale is under
/// the spacing of the doubles at the data's size.
const FOLDS: f64 = 36.0;

/// `e`-folds the logarithm of a shape may run: past `ln(1/√ε)` a law is its limit to the rounding
/// of a likelihood, as Student's is the normal's.
const SHAPE_FOLDS: f64 = 18.0;

/// How far above the estimate's cost a limit's fit may end and fit as well: `z²/2` at `z = 0.1`,
/// a drop that puts the limit a tenth of a standard error from the estimate.
const AT_LIMIT: f64 = 5e-3;

/// Doublings of a first step that carry a pin from an estimate to any bound.
const STAGES: i32 = 6;

/// What a fit as good at a limit says of the parameter it names.
#[derive(Clone, Copy)]
enum Word {
  Unidentified,
  Bound,
}

/// A limit the family declares along a coordinate: its place, its word and the name it words.
struct Limit {
  at: f64,
  word: Word,
  names: Option<usize>,
}

impl Limit {
  fn stop(&self, coordinate: usize) -> Stop {
    let names = self.names.unwrap_or(coordinate);
    match self.word {
      Word::Unidentified => Stop::Unidentified(names),
      Word::Bound => Stop::Bound(names),
    }
  }
}

/// One coordinate of a search: the size it varies on, the crate's own bounds on it, and the
/// limits of the family along it.
pub(crate) struct Coordinate {
  scale: f64,
  low: f64,
  high: f64,
  limits: Vec<Limit>,
  /// The others follow it down a valley to its limit, where a simplex stops short.
  valley: bool,
}

impl Coordinate {
  fn within(scale: f64, low: f64, high: f64) -> Self {
    Self {
      scale,
      low,
      high,
      limits: Vec::new(),
      valley: false,
    }
  }

  /// A location of a sample centred at `centre` with spread `spread`, held within `1/ε` spreads.
  pub(crate) fn location(centre: f64, spread: f64) -> Self {
    let reach = spread / f64::EPSILON;
    Self::within(spread, centre - reach, centre + reach)
  }

  /// The logarithm of a scale whose natural size is `e^centre`.
  pub(crate) fn log(centre: f64) -> Self {
    Self::within(1.0, centre - FOLDS, centre + FOLDS)
  }

  /// The logarithm of a shape: its far end is the law's limit, which leaves it unidentified.
  pub(crate) fn shape(centre: f64) -> Self {
    let high = centre + SHAPE_FOLDS;
    let limit = Limit {
      at: high,
      word: Word::Unidentified,
      names: None,
    };
    Self {
      limits: vec![limit],
      valley: true,
      ..Self::within(1.0, centre - SHAPE_FOLDS, high)
    }
  }

  /// A logit, or half of one, within `reach`: each end is the bound of the parameter it names.
  pub(crate) fn share(reach: f64, below: usize, above: usize) -> Self {
    let limit = |at: f64, names: usize| Limit {
      at,
      word: Word::Bound,
      names: Some(names),
    };
    Self {
      limits: vec![limit(-reach, below), limit(reach, above)],
      ..Self::within(1.0, -reach, reach)
    }
  }

  /// A coordinate of unit size that the objective bounds itself.
  pub(crate) fn free() -> Self {
    Self::within(1.0, f64::NEG_INFINITY, f64::INFINITY)
  }

  /// The limit the coordinate's bound at `value` is, if the family declares one there.
  fn limit(&self, value: f64) -> Option<&Limit> {
    self.limits.iter().find(|limit| limit.at == value)
  }
}

/// The reach of a logit: past it the share is `0` or `1` to the spacing of the doubles.
pub(crate) const LOGIT: f64 = FOLDS;

/// Where a search ended and what the first-order check says of the point.
pub(crate) struct Found {
  pub(crate) at: Vec<f64>,
  pub(crate) stop: Stop,
  pub(crate) iterations: usize,
}

struct Cost<'a>(&'a dyn Fn(&[f64]) -> f64);

impl CostFunction for Cost<'_> {
  type Param = Vec<f64>;
  type Output = f64;
  type Error = Infallible;

  fn cost(&self, p: &Vec<f64>) -> Result<f64, Infallible> {
    Ok((self.0)(p))
  }
}

impl Gradient for Cost<'_> {
  type Gradient = Vec<f64>;

  fn gradient(&self, p: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
    let slope = |i: usize| {
      let step = 1e-6 * (1.0 + p[i].abs());
      let (mut plus, mut minus) = (p.clone(), p.clone());
      plus[i] += step;
      minus[i] -= step;
      ((self.0)(&plus) - (self.0)(&minus)) / (2.0 * step)
    };
    Ok((0..p.len()).map(slope).collect())
  }
}

/// The L-BFGS polish of `objective` from `theta` by central differences: the finisher of a fit
/// that has no gradient of its own.
pub(crate) fn by_differences(
  objective: &dyn Fn(&[f64]) -> f64,
  theta: Vec<f64>,
) -> (Vec<f64>, usize) {
  polish(Cost(objective), theta)
}

/// A finisher: a point no higher than the one it is given, and the iterations it took.
pub(crate) trait Finish: Fn(&dyn Fn(&[f64]) -> f64, Vec<f64>) -> (Vec<f64>, usize) {}

impl<F: Fn(&dyn Fn(&[f64]) -> f64, Vec<f64>) -> (Vec<f64>, usize)> Finish for F {}

/// `objective` inside the coordinates' bounds, [`NO_VALUE`] outside.
fn boxed<'a>(
  objective: &'a dyn Fn(&[f64]) -> f64,
  coordinates: &'a [Coordinate],
) -> impl Fn(&[f64]) -> f64 + 'a {
  let inside =
    move |p: &[f64]| (p.iter().zip(coordinates)).all(|(v, c)| (c.low..=c.high).contains(v));
  move |p: &[f64]| if inside(p) { objective(p) } else { NO_VALUE }
}

/// A simplex about `from`, each coordinate `step` of its scale away.
fn simplex(from: &[f64], coordinates: &[Coordinate], step: f64) -> Vec<Vec<f64>> {
  let mut simplex = vec![from.to_vec(); from.len() + 1];
  for (i, coordinate) in coordinates.iter().enumerate() {
    simplex[i + 1][i] += step * coordinate.scale;
  }
  simplex
}

/// The simplex from the best of `starts` and the finisher on its point: the point, the
/// iterations, and whether the simplex ran into its cap.
fn descent(
  boxed: &dyn Fn(&[f64]) -> f64,
  finish: &impl Finish,
  starts: &[Vec<f64>],
  coordinates: &[Coordinate],
) -> (Vec<f64>, usize, bool) {
  let start = best_start(starts, boxed).expect("a non-empty starting grid");
  let first = simplex(start, coordinates, STEP);
  let run = run_nelder_mead_within(Cost(boxed), first, CAP, Spread::Absolute(SPREAD), false);
  let capped = run.stop == SimplexStop::IterationCap;
  // Off a valley a simplex that met its spread hands over a point the check reads as it is.
  let (at, polished) = if capped || coordinates.iter().any(|c| c.valley) {
    finish(boxed, run.best)
  } else {
    (run.best, 0)
  };
  (at, run.iterations as usize + polished, capped)
}

/// The search of [`minimise`] without its verdict: a stage whose point another stage starts from.
pub(crate) fn descend(
  objective: impl Fn(&[f64]) -> f64,
  finish: impl Finish,
  starts: &[Vec<f64>],
  coordinates: &[Coordinate],
) -> (Vec<f64>, usize) {
  let (at, iterations, _) = descent(
    &boxed(&objective, coordinates),
    &finish,
    starts,
    coordinates,
  );
  (at, iterations)
}

/// The minimum of `objective`, a sum of `terms` and [`NO_VALUE`] where it has none, from the best
/// of `starts`; `finish` polishes a point, never to a higher cost.
pub(crate) fn minimise(
  objective: impl Fn(&[f64]) -> f64,
  finish: impl Finish,
  starts: &[Vec<f64>],
  coordinates: &[Coordinate],
  terms: usize,
) -> Found {
  let boxed = boxed(&objective, coordinates);
  let valued = |p: &[f64]| Some(boxed(p)).filter(|cost| *cost < NO_VALUE);
  let (mut at, mut iterations, capped) = descent(&boxed, &finish, starts, coordinates);
  let axes = coordinates.iter().map(|c| Axis {
    low: c.low,
    high: c.high,
    scale: c.scale,
  });
  let axes = axes.collect::<Vec<_>>();
  let read = |at: &[f64]| {
    let found = beside(valued, at, &axes, terms);
    let i = found.along;
    let stop = found.stop(at[i] <= axes[i].low || at[i] >= axes[i].high);
    (found, stop)
  };
  let (mut found, mut stop) = read(&at);
  // A gain beside the point: the finisher runs from the point of it, and the lower stays.
  for round in 0..=RESEATS {
    let lower = match stop {
      Stop::NotStationary | Stop::Bound(_) => found.best.clone(),
      _ => break,
    };
    // The last round has no search left: its gain is kept and its verdict stands.
    let (next, steps) = if round < RESEATS {
      finish(&boxed, lower)
    } else {
      (lower, 0)
    };
    iterations += steps;
    if boxed(&next) >= boxed(&at) {
      break;
    }
    at = next;
    if round < RESEATS {
      (found, stop) = read(&at);
    }
  }
  // A stop that names a parameter comes first, then the cap, which is never a convergence: the
  // check passes stalls a capped search leaves.
  if matches!(stop, Stop::Converged | Stop::NotStationary | Stop::Edge) {
    let (limit, steps) = fitting(&boxed, &finish, &mut at, coordinates);
    iterations += steps;
    let named = limit.or_else(|| resting(&at, coordinates));
    stop = named.unwrap_or(if capped { Stop::Capped } else { stop });
  }
  Found {
    at,
    stop,
    iterations,
  }
}

/// The limit of the family that fits best among those the estimate reaches at no cost: the pin
/// goes there by doubled steps, the finisher after each, and stops where the fit has risen
/// [`AT_LIMIT`] above the cost at `at`. `at` moves to a lower fit at a limit.
fn fitting(
  boxed: &dyn Fn(&[f64]) -> f64,
  finish: &impl Finish,
  at: &mut Vec<f64>,
  coordinates: &[Coordinate],
) -> (Option<Stop>, usize) {
  let (here, mut iterations) = (boxed(at), 0);
  let mut best = (here + AT_LIMIT, None, at.clone());
  for (i, coordinate) in coordinates.iter().enumerate() {
    for limit in &coordinate.limits {
      let toward = (limit.at - at[i]).signum();
      let steps = (0..STAGES).map(|k| at[i] + toward * coordinate.scale * f64::from(1 << k));
      let short = steps.take_while(|value| (limit.at - value) * toward > 0.0);
      let (mut fit, mut cost) = (at.clone(), here);
      for value in short.chain([limit.at]) {
        let pin = |p: &[f64]| {
          let mut pinned = p.to_vec();
          pinned[i] = value;
          pinned
        };
        let (free, steps) = finish(&|p: &[f64]| boxed(&pin(p)), fit);
        iterations += steps;
        fit = pin(&free);
        cost = boxed(&fit);
        if cost > here + AT_LIMIT {
          break;
        }
      }
      if cost <= best.0 && (best.1.is_none() || cost < best.0) {
        best = (cost, Some(limit.stop(i)), fit);
      }
    }
  }
  if best.0 < here {
    *at = best.2;
  }
  (best.1, iterations)
}

/// The stop of a point within the check's first step of one of the crate's own bounds: no
/// estimate, a limit, in the family's word for it where it declares one.
fn resting(at: &[f64], coordinates: &[Coordinate]) -> Option<Stop> {
  let near = |c: &Coordinate| f64::EPSILON.cbrt() * c.scale;
  (at.iter().zip(coordinates).enumerate()).find_map(|(i, (v, c))| {
    let end = [c.low, c.high]
      .into_iter()
      .find(|end| (v - end).abs() <= near(c))?;
    Some(c.limit(end).map_or(Stop::Bound(i), |limit| limit.stop(i)))
  })
}

#[cfg(test)]
mod tests {
  use super::*;

  fn search(objective: impl Fn(&[f64]) -> f64, coordinates: &[Coordinate]) -> Found {
    let start = vec![vec![0.0; coordinates.len()]];
    minimise(objective, by_differences, &start, coordinates, 1000)
  }

  fn pair() -> [Coordinate; 2] {
    [Coordinate::location(0.0, 1.0), Coordinate::log(0.0)]
  }

  #[test]
  fn a_minimum_inside_the_bounds_converges() {
    let bowl = |p: &[f64]| 10.0 + 50.0 * (p[0] - 0.3).powi(2) + 50.0 * (p[1] - 0.2).powi(2);
    let found = search(bowl, &pair());
    assert!(
      found.stop.converged(),
      "{}",
      found.stop.message(&["a", "b"])
    );
    // The check's `1e-7` in the cost is `√(1e-7/50)` in a coordinate.
    assert!((found.at[0] - 0.3).abs() < 5e-5 && (found.at[1] - 0.2).abs() < 5e-5);
  }

  fn shaped() -> [Coordinate; 2] {
    [Coordinate::location(0.0, 1.0), Coordinate::shape(0.0)]
  }

  /// A valley along `p₀ = p₁` that falls `1e-3` in all to the shape's limit: a step along the
  /// shape alone climbs the valley's wall, and the fit pinned at the limit is as low.
  #[test]
  fn a_limit_that_fits_as_well_is_not_identified() {
    let valley = |p: &[f64]| 10.0 + 50.0 * (p[0] - p[1]).powi(2) + 1e-3 * (-p[1]).exp();
    let found = search(valley, &shaped());
    assert_eq!(found.stop.message(&["mu", "eta"]), "eta is not identified");
  }

  /// A shelf half a unit above the minimum is no fit as good: wide, and identified.
  #[test]
  fn a_limit_that_fits_worse_is_identified() {
    let shelf = |p: &[f64]| 10.0 + 50.0 * (p[0] - 0.3).powi(2) + 0.5 * (1.0 - (-p[1] * p[1]).exp());
    let found = search(shelf, &shaped());
    assert!(
      found.stop.converged(),
      "{}",
      found.stop.message(&["mu", "eta"])
    );
  }

  #[test]
  fn a_cost_falling_to_the_crate_bound_stops_at_it() {
    let slide = |p: &[f64]| 10.0 + 50.0 * (p[0] - 0.3).powi(2) - p[1];
    let found = search(slide, &pair());
    assert_eq!(
      found.stop.message(&["mu", "sigma"]),
      "stopped at the bound of sigma"
    );
  }

  /// Rosenbrock's valley in thirty coordinates outlasts the simplex's cap; wherever the finisher
  /// then takes the point, a capped search is no convergence.
  #[test]
  fn a_capped_search_is_not_converged() {
    let valley = |p: &[f64]| {
      let link = |w: &[f64]| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2);
      p.windows(2).map(link).sum::<f64>()
    };
    let coordinates = [(); 30].map(|()| Coordinate::free());
    let start = vec![vec![-1.2; 30]];
    let found = minimise(valley, by_differences, &start, &coordinates, 1000);
    assert_eq!(found.stop.message(&["p"; 30]), "iteration cap reached");
  }

  fn rosenbrock(p: &[f64]) -> f64 {
    let link = |w: &[f64]| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2);
    p.windows(2).map(link).sum::<f64>()
  }

  /// A capped search no finisher completes, in thirty coordinates.
  fn unfinished(objective: impl Fn(&[f64]) -> f64) -> Found {
    let coordinates = [(); 30].map(|()| Coordinate::free());
    let as_it_is = |_: &dyn Fn(&[f64]) -> f64, p: Vec<f64>| (p, 0);
    minimise(objective, as_it_is, &[vec![-1.2; 30]], &coordinates, 1000)
  }

  /// The cap is the search's stop, named before what the check finds beside its point.
  #[test]
  fn a_capped_search_the_check_fails_reads_the_cap() {
    let found = unfinished(rosenbrock);
    assert!(rosenbrock(&found.at) > 1.0, "{}", rosenbrock(&found.at));
    assert_eq!(found.stop.message(&["p"; 30]), "iteration cap reached");
  }

  /// The last round's gain is not left behind: the answer is the lowest point evaluated.
  #[test]
  fn the_lowest_point_seen_is_returned() {
    let lowest = std::cell::Cell::new(f64::INFINITY);
    let found = unfinished(|p| {
      let cost = rosenbrock(p);
      lowest.set(lowest.get().min(cost));
      cost
    });
    assert_eq!(rosenbrock(&found.at), lowest.get());
  }
}
