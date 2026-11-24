//! The estimators' one search: the simplex from the best start, the first-order check with a
//! finisher where it finds a gain, and the fit at each limit of the family that words the answer.

use std::convert::Infallible;

use basin::CostFunction;
use basin::Gradient;

use super::best_start;
use super::certificate::Axis;
use super::certificate::Stop;
use super::certificate::beside;
use super::curvature::standard_errors;
use super::polish;
use super::simplex::SimplexStop;
use super::simplex::Spread;
use super::simplex::run_nelder_mead_within;

/// What an objective answers where it has no value, above every sum it returns.
pub(crate) const NO_VALUE: f64 = 1e300;

/// Simplex iterations at most: the GARCH, EVT and skew-t searches' own; the QMLE and GMM had 2,000.
const CAP: u64 = 5_000;

/// The simplex's first step along a coordinate, in its scale.
const STEP: f64 = 0.1;

/// The spread of the vertex costs a simplex stops at where no finisher follows. Measured: the
/// point is up to `7e-11` above the minimum, `1.2e-5` model-based standard errors.
const SPREAD: f64 = 1e-11;

/// The spread where a finisher always follows, which takes the point the rest of the way.
const FINISHED_SPREAD: f64 = 1e-9;

/// Searches from the point of a gain the check found, `fit_mle`'s count.
const RESEATS: usize = 3;

/// `e`-folds the logarithm of a scale may run from its centre: past `ln(1/ε)` the scale is under
/// the spacing of the doubles at the data's size.
const FOLDS: f64 = 36.0;

/// `e`-folds the logarithm of a shape may run: past `ln(1/√ε)` a law is its limit to the rounding
/// of a likelihood, as Student's is the normal's.
const SHAPE_FOLDS: f64 = 18.0;

/// How far above the estimate's cost a limit's fit may end and fit as well. A convention:
/// `z²/2` at `z = 0.1` model-based standard errors, in summed log-likelihood.
const AT_LIMIT: f64 = 5e-3;

/// Model-based standard errors past which a limit is not read, a convention. It assumes the
/// profile to a limit is nowhere `K²/z² = 1.3e5` flatter than the quadratic model at the estimate.
const SCREEN: f64 = FOLDS;

/// The second differences' steps in a coordinate's scale: the simplex's own, then an eighth of
/// it where that gives no model. At `1e-3` a likelihood's roughness at a large shape is curvature.
const CURVATURE_STEPS: [f64; 2] = [STEP, 0.125 * STEP];

/// Doublings of a first step that carry a pin from an estimate to any bound.
const STAGES: i32 = 6;

/// What a fit as good at a limit says of the parameter it names.
#[derive(Clone, Copy)]
enum Word {
  Unidentified,
  Bound,
}

/// A limit the family declares along a coordinate: its place, its word, the name it words, and
/// a logarithm that rises as this one falls, their product held.
struct Limit {
  at: f64,
  word: Word,
  names: Option<usize>,
  held: Option<usize>,
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

  /// The logarithm of a shape: its far end is the law's limit, which leaves it unidentified,
  /// and its near end the shape's own bound.
  pub(crate) fn shape(centre: f64) -> Self {
    let (low, high) = (centre - SHAPE_FOLDS, centre + SHAPE_FOLDS);
    let limit = |at: f64, word: Word| Limit {
      at,
      word,
      names: None,
      held: None,
    };
    Self {
      limits: vec![limit(high, Word::Unidentified), limit(low, Word::Bound)],
      valley: true,
      ..Self::within(1.0, low, high)
    }
  }

  /// The logarithm with its low end a limit: the parameter vanishes there, and the logarithm
  /// `orphan` is not identified, the data telling their product alone.
  pub(crate) fn fading(mut self, orphan: usize) -> Self {
    self.limits.push(Limit {
      at: self.low,
      word: Word::Unidentified,
      names: Some(orphan),
      held: Some(orphan),
    });
    self
  }

  /// A logit, or half of one, within `reach`: each end is the bound of the parameter it names.
  pub(crate) fn share(reach: f64, below: usize, above: usize) -> Self {
    let limit = |at: f64, names: usize| Limit {
      at,
      word: Word::Bound,
      names: Some(names),
      held: None,
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

/// The simplex from the best of `starts` to `spread`, and the finisher where it capped or
/// `finished`: the point, the iterations, and whether the simplex ran into its cap.
fn descent(
  boxed: &dyn Fn(&[f64]) -> f64,
  finish: &impl Finish,
  starts: &[Vec<f64>],
  coordinates: &[Coordinate],
  (spread, finished): (Spread, bool),
) -> (Vec<f64>, usize, bool) {
  let start = best_start(starts, boxed).expect("a non-empty starting grid");
  let first = simplex(start, coordinates, STEP);
  let run = run_nelder_mead_within(Cost(boxed), first, CAP, spread, false);
  let capped = run.stop == SimplexStop::IterationCap;
  let (at, polished) = if capped || finished {
    finish(boxed, run.best)
  } else {
    (run.best, 0)
  };
  (at, run.iterations as usize + polished, capped)
}

/// A stage whose point another stage starts from: the simplex alone, and no verdict but whether
/// it capped.
pub(crate) fn descend(
  objective: impl Fn(&[f64]) -> f64,
  finish: impl Finish,
  starts: &[Vec<f64>],
  coordinates: &[Coordinate],
) -> (Vec<f64>, usize, bool) {
  let boxed = boxed(&objective, coordinates);
  let stop = (Spread::Absolute(SPREAD), false);
  descent(&boxed, &finish, starts, coordinates, stop)
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
  // On a valley a simplex stops short and the finisher follows; elsewhere the check reads the
  // simplex's own point, at the tighter spread.
  let stop = if coordinates.iter().any(|c| c.valley) {
    (Spread::Absolute(FINISHED_SPREAD), true)
  } else {
    (Spread::Absolute(SPREAD), false)
  };
  let (mut at, mut iterations, capped) = descent(&boxed, &finish, starts, coordinates, stop);
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

/// Whether each coordinate has a limit within [`SCREEN`] standard errors of `at` by the
/// quadratic model there; with no such model (no value, no positive curvature) every limit is.
fn screened(
  objective: &dyn Fn(&[f64]) -> f64,
  at: &[f64],
  coordinates: &[Coordinate],
) -> Vec<bool> {
  let limited = coordinates.iter().map(|c| !c.limits.is_empty());
  let limited = limited.collect::<Vec<_>>();
  if !limited.contains(&true) {
    return limited;
  }
  let model = CURVATURE_STEPS.iter().find_map(|step| {
    let steps = coordinates.iter().map(|c| step * c.scale);
    standard_errors(objective, at, &steps.collect::<Vec<_>>())
  });
  let Some(errors) = model else {
    return limited;
  };
  let near = |(i, c): (usize, &Coordinate)| {
    let reach = SCREEN * errors[i];
    c.limits
      .iter()
      .any(|limit| (limit.at - at[i]).abs() <= reach)
  };
  coordinates.iter().enumerate().map(near).collect()
}

/// The best-fitting limit whose fit is within [`AT_LIMIT`] of the cost at `at`: the pin goes to
/// a screened limit by doubled steps, the finisher after each. `at` moves to a lower fit.
fn fitting(
  boxed: &dyn Fn(&[f64]) -> f64,
  finish: &impl Finish,
  at: &mut Vec<f64>,
  coordinates: &[Coordinate],
) -> (Option<Stop>, usize) {
  let (here, mut iterations) = (boxed(at), 0);
  let mut best = (here + AT_LIMIT, None, at.clone());
  let read = screened(boxed, at, coordinates);
  for (i, coordinate) in coordinates.iter().enumerate().filter(|(i, _)| read[*i]) {
    for limit in &coordinate.limits {
      let toward = (limit.at - at[i]).signum();
      // A held logarithm is its limit `1/√ε` above its centre, as a shape is: the pin ends there.
      let room = limit.held.map_or(f64::INFINITY, |held| {
        let other = &coordinates[held];
        (0.5 * (other.low + other.high) + SHAPE_FOLDS - at[held]).max(0.0)
      });
      let end = at[i] + toward * (limit.at - at[i]).abs().min(room);
      let steps = (0..STAGES).map(|k| at[i] + toward * coordinate.scale * f64::from(1 << k));
      let short = steps.take_while(|value| (end - value) * toward > 0.0);
      let mut fit = at.clone();
      for value in short.chain([end]) {
        let pin = |p: &[f64]| {
          let mut pinned = p.to_vec();
          pinned[i] = value;
          pinned
        };
        let mut from = pin(&fit);
        if let Some(held) = limit.held {
          from[held] += fit[i] - value;
        }
        let (free, steps) = finish(&|p: &[f64]| boxed(&pin(p)), from);
        iterations += steps;
        fit = pin(&free);
      }
      let cost = boxed(&fit);
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
mod tests;
