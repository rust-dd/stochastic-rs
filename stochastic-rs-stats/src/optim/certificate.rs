//! The first-order check a fit's verdict rests on: what a step along one parameter gains beside the
//! estimate, in the objective's own units, and the stop that reading gives.

use super::SolverStop;
use super::simplex::SimplexStop;
use crate::traits::RealExt;

/// The most a step along one parameter gains in log-likelihood beside a converged estimate: on
/// 117,768 fits the stationary ones read `2.3e-9` or less, and a stall anything up to 94.
const STATIONARY: f64 = 1e-7;

/// Roundings of the sum a gain must clear: on 366 fits of `1e4` to `3e6` points the walks beside
/// an optimum gain `0.30` of one or less, and beside the stalls they flag `1.2` or more.
const ROUNDINGS: f64 = 1.0;

/// The least count of roundings a sum is given: below 256 terms a walk needs this allowance to
/// cross a flat stretch, as one census stall of 100 points shows.
const SHORT: f64 = 16.0;

/// Quarterings of a first step that has no value, down to `4⁻⁸` of it.
const SHORTENINGS: usize = 8;

/// Steps of a walk beside an estimate, the last `4⁶³` times the first: past any box met.
const FURTHEST: usize = 64;

/// One coordinate the walks step along: its box and the size it varies on, the first step's unit.
pub(crate) struct Axis {
  pub(crate) low: f64,
  pub(crate) high: f64,
  pub(crate) scale: f64,
}

/// What the walks of [`beside`] found.
pub(crate) struct Beside {
  /// The most one of them gains in the objective.
  pub(crate) gain: f64,
  /// The parameter that walk steps along.
  pub(crate) along: usize,
  /// The rounding of the summed objective at the estimate, `ε max(√N, 16) (1 + |f|)`.
  pub(crate) rounding: f64,
  /// The first step of a walk met a point without a value.
  pub(crate) walled: bool,
  /// The point of that gain, the estimate itself where nothing gains.
  pub(crate) best: Vec<f64>,
}

/// Why a fit stopped, `Converged` alone being `converged = true`.
pub(crate) enum Stop {
  Converged,
  Search(SolverStop),
  NotStationary,
  Bound(usize),
  Edge,
  Outside,
  Unidentified(usize),
  Capped,
}

impl Stop {
  pub(crate) fn converged(&self) -> bool {
    matches!(self, Self::Converged)
  }

  /// The stop in the words of quant's calibrators, `names` the fit's parameters.
  pub(crate) fn message(&self, names: &[&str]) -> String {
    let words = match self {
      Self::Converged => "converged",
      Self::Search(SolverStop::Converged) | Self::Outside => SolverStop::Other.words(),
      Self::Search(stop) => stop.words(),
      Self::NotStationary => "not stationary",
      Self::Bound(held) => return format!("stopped at the bound of {}", names[*held]),
      Self::Unidentified(open) => return format!("{} is not identified", names[*open]),
      Self::Capped => SimplexStop::IterationCap.message(),
      Self::Edge => "stopped at the edge of the law's support",
    };
    words.into()
  }
}

impl Beside {
  /// The reading of a point no walk was taken beside.
  pub(crate) fn nothing(at: &[f64]) -> Self {
    Self {
      gain: 0.0,
      along: 0,
      rounding: 0.0,
      walled: false,
      best: at.to_vec(),
    }
  }

  /// What a gain must clear: the crate's bound, or the sum's own rounding where that is larger.
  pub(crate) fn bar(&self) -> f64 {
    STATIONARY.max(ROUNDINGS * self.rounding)
  }

  /// The stop the walks read, `held` saying that the parameter of the largest gain is on a bound.
  pub(crate) fn stop(&self, held: bool) -> Stop {
    match () {
      _ if self.gain <= self.bar() => Stop::Converged,
      _ if held => Stop::Bound(self.along),
      _ if self.walled => Stop::Edge,
      _ => Stop::NotStationary,
    }
  }
}

/// The largest gain a step along one axis realises beside `at`: `ε^{1/3}` of its scale either way,
/// four times further while `objective` (`None`: no value) does not rise, or its parabola's low.
pub(crate) fn beside(
  objective: impl Fn(&[f64]) -> Option<f64>,
  at: &[f64],
  axes: &[Axis],
  additions: usize,
) -> Beside {
  let Some(here) = objective(at) else {
    return Beside {
      gain: f64::INFINITY,
      along: 0,
      rounding: 0.0,
      walled: true,
      best: at.to_vec(),
    };
  };
  // `N` additions each round a partial sum near `|f|` by `ε/2`, and the roundings add as a walk.
  let mut beside = Beside {
    gain: 0.0,
    along: 0,
    rounding: f64::EPSILON * (additions as f64).sqrt().max(SHORT) * (1.0 + here.abs()),
    walled: false,
    best: at.to_vec(),
  };
  let bar = beside.bar();
  // A sum's rounding is no rise: below it the walk goes on, across a flat stretch.
  let rise = ROUNDINGS * beside.rounding;
  // Half the two first steps' difference along each axis they are whole on: the slope there.
  let mut slopes = vec![0.0; at.len()];
  for (i, axis) in axes.iter().enumerate() {
    let first = f64::EPSILON.cbrt() * axis.scale;
    // What each first step costs where it is a whole step onto a point with a value.
    let mut whole = [None, None];
    for (side, whole) in [1.0, -1.0].into_iter().zip(&mut whole) {
      let mut step = side * first;
      let mut moved = at.to_vec();
      for nth in 0..FURTHEST {
        let to = (at[i] + step).clamp(axis.low, axis.high);
        if to == moved[i] || !to.is_finite() {
          break;
        }
        moved[i] = to;
        let mut cost = objective(&moved);
        beside.walled |= nth == 0 && cost.is_none();
        if nth == 0 && to == at[i] + step {
          *whole = cost;
        }
        // A first step past the edge of the objective's domain is shortened until it has a
        // value: a slope into the edge is a gain no whole step can read.
        for _ in 0..SHORTENINGS {
          if nth > 0 || cost.is_some() {
            break;
          }
          step *= 0.25;
          moved[i] = at[i] + step;
          cost = objective(&moved);
        }
        let Some(cost) = cost.filter(|cost| *cost <= here + rise) else {
          break;
        };
        if here - cost > beside.gain {
          (beside.gain, beside.along) = (here - cost, i);
          beside.best.clone_from(&moved);
        }
        step *= 4.0;
      }
    }
    // Both steps rise: a Newton step under half the first is past the walks, and the three
    // points' parabola predicts its gain `g²/2H`, which counts only as its minimum realises it.
    let [Some(up), Some(down)] = whole else {
      continue;
    };
    slopes[i] = 0.5 * (up - down);
    let curvature = up + down - 2.0 * here;
    if !(up > here + rise && down > here + rise) || (up - down).powi(2) <= 8.0 * curvature * bar {
      continue;
    }
    let mut moved = at.to_vec();
    moved[i] = at[i] - first * (up - down) / (2.0 * curvature);
    let realised = objective(&moved).map_or(0.0, |cost| here - cost);
    if realised > beside.gain {
      (beside.gain, beside.along) = (realised, i);
      beside.best = moved;
    }
  }
  // A valley no axis runs along: one walk down the slopes, the steepest axis by its first step.
  let steepest = f64::max_or_nan_fold(slopes.iter().map(|g| g.abs()));
  if steepest > 0.0 {
    let along = (0..at.len())
      .find(|i| slopes[*i].abs() == steepest)
      .unwrap_or(0);
    let down = |i: usize| -slopes[i] / steepest * f64::EPSILON.cbrt() * axes[i].scale;
    let mut reach = 1.0;
    for _ in 0..FURTHEST {
      let step = |i: usize| (at[i] + reach * down(i)).clamp(axes[i].low, axes[i].high);
      let moved = (0..at.len()).map(step).collect::<Vec<_>>();
      let Some(cost) = objective(&moved).filter(|cost| *cost <= here + rise) else {
        break;
      };
      if here - cost > beside.gain {
        (beside.gain, beside.along) = (here - cost, along);
        beside.best = moved;
      }
      reach *= 4.0;
    }
  }
  beside
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The first step on an axis of scale 1.
  const ROOT: f64 = 6.055454452393343e-6;

  fn axis(low: f64, high: f64) -> [Axis; 1] {
    [Axis {
      low,
      high,
      scale: 1.0,
    }]
  }

  /// A Newton step of a quarter of the first step: both first steps rise, so no walk gains, and
  /// the three points give `g²/2H = ½·H·(h/4)²` to the rounding of their differences.
  #[test]
  fn a_newton_step_shorter_than_half_the_first_step_is_read() {
    let (curvature, level, terms) = (2e9, 1e4, 1_000_000);
    let optimum = 0.3 - 0.25 * ROOT;
    let objective = |z: &[f64]| Some(level + 0.5 * curvature * (z[0] - optimum).powi(2));
    let read = beside(objective, &[0.3], &axis(-1.0, 1.0), terms);
    let decrement = 0.5 * curvature * (0.25 * ROOT).powi(2);
    assert!(decrement > 1e-3 && read.rounding < 1e-8);
    assert!(
      (read.gain - decrement).abs() <= 0.5 * read.rounding,
      "{} against {decrement}",
      read.gain
    );
    assert!(matches!(read.stop(false), Stop::NotStationary));
  }

  /// Two slopes at a true minimum predict a gain and realise none.
  #[test]
  fn a_kink_at_a_minimum_reads_no_gain() {
    let objective = |z: &[f64]| {
      let d = z[0] - 0.3;
      Some(1e4 + if d > 0.0 { 5e3 * d } else { -1e3 * d })
    };
    let read = beside(objective, &[0.3], &axis(-1.0, 1.0), 1000);
    assert_eq!(read.gain, 0.0);
    assert!(matches!(read.stop(false), Stop::Converged));
  }

  /// A first step onto a point without a value is no rise to fit a parabola through.
  #[test]
  fn a_wall_beside_the_estimate_is_not_read_as_curvature() {
    let objective =
      |z: &[f64]| (z[0] <= 0.3 + 0.5 * ROOT).then(|| 1e4 + 2e9 * (z[0] - 0.3).powi(2));
    let read = beside(objective, &[0.3], &axis(-1.0, 1.0), 1000);
    assert!(read.walled && read.gain == 0.0, "{}", read.gain);
  }

  /// The objective falls into the edge of its domain an eighth of a first step away: the
  /// shortened step reads the fall, and the stop is the edge's.
  #[test]
  fn a_fall_into_the_edge_of_the_domain_is_read() {
    let objective = |z: &[f64]| (z[0] <= 0.3 + 0.125 * ROOT).then(|| 1e4 - 1e3 * (z[0] - 0.3));
    let read = beside(objective, &[0.3], &axis(-1.0, 1.0), 1000);
    assert!(read.walled && read.gain > read.bar(), "{}", read.gain);
    assert!(matches!(read.stop(false), Stop::Edge));
  }

  /// A valley along `x = y` falling by `2e-3` a unit: a step along either axis climbs its wall,
  /// and the walk down the two slopes gains what neither can.
  #[test]
  fn a_valley_no_axis_runs_along_is_walked_down_its_slopes() {
    let objective = |z: &[f64]| Some(1e6 * (z[0] - z[1]).powi(2) - 1e-3 * (z[0] + z[1]));
    let axes = [axis(-1.0, 1.0), axis(-1.0, 1.0)].map(|[axis]| axis);
    let read = beside(objective, &[0.0, 0.0], &axes, 1000);
    assert!(read.gain > read.bar(), "{}", read.gain);
    assert_eq!(read.best[0], read.best[1]);
    assert!(matches!(read.stop(false), Stop::NotStationary));
  }

  #[test]
  fn each_stop_of_a_search_has_its_words() {
    let names = ["mu", "sigma"];
    assert_eq!(Stop::Converged.message(&names), "converged");
    assert_eq!(Stop::NotStationary.message(&names), "not stationary");
    assert_eq!(
      Stop::Bound(1).message(&names),
      "stopped at the bound of sigma"
    );
    assert_eq!(
      Stop::Unidentified(1).message(&names),
      "sigma is not identified"
    );
    let capped = Stop::Capped;
    assert_eq!(capped.message(&names), "iteration cap reached");
    assert!(!capped.converged() && Stop::Converged.converged());
  }

  /// Within a first step of a bound the two steps are unequal, and the three points no parabola's.
  #[test]
  fn a_step_cut_at_a_bound_is_not_read_as_curvature() {
    let optimum = 0.3 - 0.25 * ROOT;
    let objective = |z: &[f64]| Some(1e4 + 1e9 * (z[0] - optimum).powi(2));
    let read = beside(objective, &[0.3], &axis(-1.0, 0.3 + 0.5 * ROOT), 1000);
    assert_eq!(read.gain, 0.0);
  }
}
