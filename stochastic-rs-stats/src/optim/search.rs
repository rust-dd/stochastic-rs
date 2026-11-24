//! The estimators' one search: the simplex from the best start, a finisher on its point, and the
//! first-order check that words the answer.

use std::convert::Infallible;

use basin::CostFunction;
use basin::Gradient;

use super::best_start;
use super::certificate::Axis;
use super::certificate::IDENTIFIED;
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

/// The probing simplex's, and its cap: a minimum ends it in a step, a valley never does.
const PROBE: f64 = 1e-3;
const PROBE_CAP: u64 = 400;

/// Probes in a row, each from the last one's best: a simplex that shrank onto a valley's floor
/// is rebuilt at full size (Kelley 1999, the restart).
const PROBES: usize = 2;

/// The spread of the vertex costs a simplex hands the finisher at: under the check's `1e-7`.
const SPREAD: f64 = 1e-9;

/// Searches from the point of a gain the check found, `fit_mle`'s count.
const RESEATS: usize = 3;

/// `e`-folds the logarithm of a scale may run from its centre: past `ln(1/ε)` the scale is under
/// the spacing of the doubles at the data's size.
const FOLDS: f64 = 36.0;

/// `e`-folds the logarithm of a shape may run: past `ln(1/√ε)` a law is its limit to the rounding
/// of a likelihood, as Student's is the normal's.
const SHAPE_FOLDS: f64 = 18.0;

/// One coordinate of a search: the size it varies on, the crate's own bounds on it, and whether
/// it is a logarithm, whose bounds are where the data stop telling its value.
pub(crate) struct Coordinate {
  scale: f64,
  low: f64,
  high: f64,
  log: bool,
}

impl Coordinate {
  /// A location of a sample centred at `centre` with spread `spread`, held within `1/ε` spreads.
  pub(crate) fn location(centre: f64, spread: f64) -> Self {
    let reach = spread / f64::EPSILON;
    Self {
      scale: spread,
      low: centre - reach,
      high: centre + reach,
      log: false,
    }
  }

  /// The logarithm of a scale or a shape whose natural size is `e^centre`.
  pub(crate) fn log(centre: f64) -> Self {
    Self {
      scale: 1.0,
      low: centre - FOLDS,
      high: centre + FOLDS,
      log: true,
    }
  }

  /// The logarithm of a shape, whose far end is the law's limit.
  pub(crate) fn shape(centre: f64) -> Self {
    Self {
      scale: 1.0,
      low: centre - SHAPE_FOLDS,
      high: centre + SHAPE_FOLDS,
      log: true,
    }
  }

  /// A coordinate of unit size that the objective bounds itself.
  pub(crate) fn free() -> Self {
    Self {
      scale: 1.0,
      low: f64::NEG_INFINITY,
      high: f64::INFINITY,
      log: false,
    }
  }
}

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

/// The minimum of `objective`, a sum of `terms` and [`NO_VALUE`] where it has none, from the best
/// of `starts`; `finish` polishes a point, never to a higher cost.
pub(crate) fn minimise(
  objective: impl Fn(&[f64]) -> f64,
  finish: impl Fn(&dyn Fn(&[f64]) -> f64, Vec<f64>) -> (Vec<f64>, usize),
  starts: &[Vec<f64>],
  coordinates: &[Coordinate],
  terms: usize,
) -> Found {
  let inside = |p: &[f64]| (p.iter().zip(coordinates)).all(|(v, c)| (c.low..=c.high).contains(v));
  let boxed = |p: &[f64]| if inside(p) { objective(p) } else { NO_VALUE };
  let valued = |p: &[f64]| Some(boxed(p)).filter(|cost| *cost < NO_VALUE);
  let start = best_start(starts, boxed).expect("a non-empty starting grid");
  let mut simplex = vec![start.clone(); start.len() + 1];
  for (i, coordinate) in coordinates.iter().enumerate() {
    simplex[i + 1][i] += STEP * coordinate.scale;
  }
  let run = run_nelder_mead_within(Cost(&boxed), simplex, CAP, Spread::Absolute(SPREAD), false);
  let (mut at, polished) = finish(&boxed, run.best);
  let mut iterations = run.iterations as usize + polished;
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
  // A simplex a thousandth of each scale wide from `at`: it ends at once on a minimum and crawls
  // along a valley no axis runs down, the gain the check's walks cannot read.
  let probe = |at: &[f64]| {
    let mut lower = at.to_vec();
    let mut steps = 0;
    for _ in 0..PROBES {
      let mut simplex = vec![lower.clone(); lower.len() + 1];
      for (i, coordinate) in coordinates.iter().enumerate() {
        simplex[i + 1][i] += PROBE * coordinate.scale;
      }
      let spread = Spread::Absolute(SPREAD);
      let run = run_nelder_mead_within(Cost(&boxed), simplex, PROBE_CAP, spread, false);
      steps += run.iterations as usize;
      lower = run.best;
    }
    (lower, steps)
  };
  let (mut found, mut stop) = read(&at);
  // A gain beside the point, by the check or by the probe: the finisher runs from the point of
  // it, and the lower stays. A point both pass is never searched again.
  for round in 0..=RESEATS {
    let from = match stop {
      Stop::NotStationary | Stop::Bound(_) => found.best.clone(),
      Stop::Converged => {
        let (lower, steps) = probe(&at);
        iterations += steps;
        if boxed(&at) - boxed(&lower) <= found.bar() {
          break;
        }
        stop = Stop::NotStationary;
        lower
      }
      _ => break,
    };
    if round == RESEATS {
      break;
    }
    let (next, steps) = finish(&boxed, from);
    iterations += steps;
    if boxed(&next) >= boxed(&at) {
      break;
    }
    at = next;
    (found, stop) = read(&at);
  }
  let stop = match stop {
    Stop::Converged => resting(&at, coordinates)
      .or_else(|| unidentified(valued, &at, coordinates))
      .unwrap_or(match run.stop {
        // The check passes stalls a capped search leaves, so a cap is never a convergence.
        SimplexStop::IterationCap => Stop::Simplex(run.stop),
        _ => Stop::Converged,
      }),
    stop => stop,
  };
  Found {
    at,
    stop,
    iterations,
  }
}

/// The stop of a point within the check's first step of one of the crate's own bounds: no
/// estimate, a limit.
fn resting(at: &[f64], coordinates: &[Coordinate]) -> Option<Stop> {
  let near = |c: &Coordinate| f64::EPSILON.cbrt() * c.scale;
  let on = |(v, c): (&f64, &Coordinate)| *v <= c.low + near(c) || *v >= c.high - near(c);
  at.iter().zip(coordinates).position(on).map(Stop::Bound)
}

/// The stop of a point whose 95 % likelihood interval for a logarithmic coordinate has no upper
/// end: no rise of [`IDENTIFIED`] at 1, 2, 4, … `e`-folds up, as far as the objective has a value.
fn unidentified(
  valued: impl Fn(&[f64]) -> Option<f64>,
  at: &[f64],
  coordinates: &[Coordinate],
) -> Option<Stop> {
  let here = valued(at)?;
  let open = |(i, c): (usize, &Coordinate)| {
    let rise = |folds: f64| {
      let mut moved = at.to_vec();
      moved[i] = (at[i] + folds).min(c.high);
      valued(&moved).map(|cost| cost - here)
    };
    let mut rises = (0..6)
      .map_while(|doubling| rise(f64::from(1 << doubling)))
      .peekable();
    c.log && rises.peek().is_some() && rises.all(|rise| rise < IDENTIFIED)
  };
  (coordinates.iter().enumerate())
    .position(open)
    .map(Stop::Unidentified)
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

  /// The cost rises `0.5` at most above its minimum along the logarithm, under [`IDENTIFIED`].
  #[test]
  fn a_logarithm_the_cost_is_flat_above_is_not_identified() {
    let shelf = |p: &[f64]| 10.0 + 50.0 * (p[0] - 0.3).powi(2) + 0.5 * (1.0 - (-p[1] * p[1]).exp());
    let found = search(shelf, &pair());
    assert_eq!(
      found.stop.message(&["mu", "sigma"]),
      "sigma is not identified"
    );
  }

  /// The same shelf `4` high is a rise the interval closes on.
  #[test]
  fn a_logarithm_the_cost_rises_above_is_identified() {
    let shelf = |p: &[f64]| 10.0 + 50.0 * (p[0] - 0.3).powi(2) + 4.0 * (1.0 - (-p[1] * p[1]).exp());
    assert!(search(shelf, &pair()).stop.converged());
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
}
