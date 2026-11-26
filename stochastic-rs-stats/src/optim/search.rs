//! The estimators' one search: the simplex from the best start, the first-order check with a
//! finisher where it finds a gain, and the fit at each limit of the family that words the answer.

use std::convert::Infallible;

use basin::CostFunction;
use basin::Gradient;

use super::best_start;
use super::certificate::Axis;
use super::certificate::Stop;
use super::certificate::bar;
use super::certificate::beside;
use super::costed_start;
use super::curvature::standard_errors;
use super::polish;
use super::simplex::SimplexStop;
use super::simplex::Spread;
use super::simplex::run_nelder_mead_within;
use crate::traits::RealExt;

mod chart;

pub(crate) use chart::Coordinate;
use chart::FOLDS;
pub(crate) use chart::LOGIT;
use chart::Limit;
pub(crate) use chart::Parameter;
use chart::SHAPE_FOLDS;

/// What an objective answers where it has no value, above every sum it returns.
pub(crate) const NO_VALUE: f64 = 1e300;

/// Simplex iterations at most.
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

/// Searches from a fit on a limit's path that is lower than the answer; each lowers the cost.
const RESTARTS: usize = 3;

/// How far above the estimate's cost a limit's fit may end and fit as well. A convention:
/// `z²/2` at `z = 0.1` model-based standard errors, in summed log-likelihood.
const AT_LIMIT: f64 = 5e-3;

/// Model-based standard errors past which a limit is not read, a convention. It assumes a limit
/// the model puts `K²/2 = 648` above the estimate is not in fact within `Δ` of it (`1.3e5`).
const SCREEN: f64 = FOLDS;

/// The second differences' steps in a coordinate's scale: the simplex's own, then an eighth of
/// it where that gives no model. At `1e-3` a likelihood's roughness at a large shape is curvature.
const CURVATURE_STEPS: [f64; 2] = [STEP, 0.125 * STEP];

/// Doublings of a first step that carry a pin from an estimate to any bound.
const STAGES: i32 = 6;

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

/// The simplex from `start` to `spread`, and the finisher where it capped or `finished`: the point, the iterations, and whether the simplex ran into its cap.
fn descent(
  boxed: &dyn Fn(&[f64]) -> f64,
  finish: &impl Finish,
  start: &[f64],
  coordinates: &[Coordinate],
  (spread, finished): (Spread, bool),
) -> (Vec<f64>, usize, bool) {
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
  let start = best_start(starts, &boxed).expect("a non-empty starting grid");
  descent(&boxed, &finish, start, coordinates, stop)
}

/// The minimum of `objective`, a sum of `terms` and [`NO_VALUE`] where it has none: a descent
/// from the best start of each group, the best start's kept unless another is lower by the bar.
/// `weights` are the parameters of the limits that are no one coordinate's end.
pub(crate) fn minimise(
  objective: impl Fn(&[f64]) -> f64,
  finish: impl Finish,
  groups: &[&[Vec<f64>]],
  coordinates: &[Coordinate],
  weights: &[Parameter],
  terms: usize,
) -> Found {
  let boxed = boxed(&objective, coordinates);
  let own = chart::own(coordinates);
  let parameters = own.iter().chain(weights).collect::<Vec<_>>();
  let chart = Chart {
    coordinates,
    parameters: &parameters,
    terms,
  };
  let picks = groups.iter().map(|group| costed_start(*group, &boxed));
  let picks = picks
    .collect::<Option<Vec<_>>>()
    .expect("a start in every group");
  let first = (0..picks.len())
    .min_by(|a, b| picks[*a].0.nan_last(&picks[*b].0))
    .expect("a group of starts");
  let mut kept = restarting(&boxed, &finish, picks[first].1, &chart);
  let mut iterations = kept.iterations;
  for group in (0..groups.len()).filter(|group| *group != first) {
    let next = restarting(&boxed, &finish, picks[group].1, &chart);
    iterations += next.iterations;
    let cost = boxed(&kept.at);
    if boxed(&next.at) < cost - bar(cost, terms) {
      kept = next;
    }
  }
  Found { iterations, ..kept }
}

/// What a search reads of its family: the coordinates, the parameters of their limits, and the
/// objective's count of terms.
struct Chart<'a> {
  coordinates: &'a [Coordinate],
  parameters: &'a [&'a Parameter<'a>],
  terms: usize,
}

/// One descent from `start`, begun again from a lower point a limit's path or parameter shows.
fn restarting(
  boxed: &dyn Fn(&[f64]) -> f64,
  finish: &impl Finish,
  start: &[f64],
  chart: &Chart,
) -> Found {
  let (mut from, mut iterations) = (start.to_vec(), 0);
  for restart in 0..=RESTARTS {
    let (found, lower) = search(boxed, finish, &from, chart);
    iterations += found.iterations;
    match lower {
      Some(point) if restart < RESTARTS => from = point,
      // Out of restarts with a lower point still in sight: that point, and no convergence.
      Some(at) => {
        let stop = Stop::NotStationary;
        return Found {
          at,
          stop,
          iterations,
        };
      }
      None => {
        return Found {
          iterations,
          ..found
        };
      }
    }
  }
  unreachable!("the last restart returns")
}

/// One search from `start`: its answer, and the lowest point below it that a limit's path or a
/// step along a limit's parameter shows.
fn search(
  boxed: &dyn Fn(&[f64]) -> f64,
  finish: &impl Finish,
  start: &[f64],
  chart: &Chart,
) -> (Found, Option<Vec<f64>>) {
  let (coordinates, terms) = (chart.coordinates, chart.terms);
  let valued = |p: &[f64]| Some(boxed(p)).filter(|cost| *cost < NO_VALUE);
  // On a valley a simplex stops short and the finisher follows; elsewhere the check reads the
  // simplex's own point, at the tighter spread.
  let stop = if coordinates.iter().any(|c| c.valley) {
    (Spread::Absolute(FINISHED_SPREAD), true)
  } else {
    (Spread::Absolute(SPREAD), false)
  };
  let (mut at, mut iterations, capped) = descent(boxed, finish, start, coordinates, stop);
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
      finish(boxed, lower)
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
  let mut lower = None;
  // A stop that names a parameter comes first, then the cap, which is never a convergence: the
  // check passes stalls a capped search leaves.
  if matches!(stop, Stop::Converged | Stop::NotStationary | Stop::Edge) {
    let (limit, steps, inside) = fitting(boxed, finish, &mut at, coordinates, found.bar());
    iterations += steps;
    lower = inside;
    let named = limit.or_else(|| resting(&at, coordinates));
    stop = named.unwrap_or(if capped { Stop::Capped } else { stop });
  }
  let released = released(boxed, &at, chart);
  let lower = [lower, released].into_iter().flatten();
  let lower = lower.min_by(|a, b| boxed(a).nan_last(&boxed(b)));
  let found = Found {
    at,
    stop,
    iterations,
  };
  (found, lower)
}

/// The point of the largest gain past the bar that a step along one limit's parameter realises
/// beside `at`, the walks of [`beside`] on that axis alone.
fn released(boxed: &dyn Fn(&[f64]) -> f64, at: &[f64], chart: &Chart) -> Option<Vec<f64>> {
  let here = boxed(at);
  let mut best = (bar(here, chart.terms), None);
  for parameter in chart.parameters {
    let cost = |value: &[f64]| {
      let moved = (parameter.moved)(at, value[0])?;
      Some(boxed(&moved)).filter(|cost| *cost < NO_VALUE)
    };
    let axis = std::slice::from_ref(&parameter.axis);
    let read = beside(cost, &[(parameter.of)(at)], axis, chart.terms);
    let Some(moved) = (parameter.moved)(at, read.best[0]) else {
      continue;
    };
    let gain = here - boxed(&moved);
    if read.gain > read.bar() && gain > best.0 {
      best = (gain, Some(moved));
    }
  }
  best.1
}

/// Where a pin toward `limit` along coordinate `i` ends: the limit, or short of it where a held
/// logarithm is `1/√ε` above its centre, its own limit as a shape's is.
fn pin_end(limit: &Limit, i: usize, at: &[f64], coordinates: &[Coordinate]) -> f64 {
  let room = limit.held.map_or(f64::INFINITY, |held| {
    let other = &coordinates[held];
    (0.5 * (other.low + other.high) + SHAPE_FOLDS - at[held]).max(0.0)
  });
  let toward = (limit.at - at[i]).signum();
  at[i] + toward * (limit.at - at[i]).abs().min(room)
}

/// Whether each coordinate has a limit whose pin ends within [`SCREEN`] standard errors of `at`
/// by the quadratic model there; with no such model (no value, no positive curvature) each is.
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
    let ends = c
      .limits
      .iter()
      .map(|limit| pin_end(limit, i, at, coordinates));
    ends.into_iter().any(|end| (end - at[i]).abs() <= reach)
  };
  coordinates.iter().enumerate().map(near).collect()
}

/// The best-fitting limit whose fit is within [`AT_LIMIT`] of the cost at `at`, which moves to a
/// lower one; and a fit on a pin's way that is more than `bar` below that answer.
fn fitting(
  boxed: &dyn Fn(&[f64]) -> f64,
  finish: &impl Finish,
  at: &mut Vec<f64>,
  coordinates: &[Coordinate],
  bar: f64,
) -> (Option<Stop>, usize, Option<Vec<f64>>) {
  let (here, mut iterations) = (boxed(at), 0);
  let mut best = (here + AT_LIMIT, None, at.clone());
  let mut inside = (f64::INFINITY, at.clone());
  let read = screened(boxed, at, coordinates);
  for (i, coordinate) in coordinates.iter().enumerate().filter(|(i, _)| read[*i]) {
    for limit in &coordinate.limits {
      let end = pin_end(limit, i, at, coordinates);
      let toward = (end - at[i]).signum();
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
        let cost = boxed(&fit);
        if value != end && cost < inside.0 {
          inside = (cost, fit.clone());
        }
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
  let answer = best.0.min(here);
  let lower = (inside.0 < answer - bar).then_some(inside.1);
  (best.1, iterations, lower)
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
