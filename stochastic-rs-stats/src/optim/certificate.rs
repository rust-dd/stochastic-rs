//! The first-order check a fit's verdict rests on: what a step along one parameter gains beside the
//! estimate, in the objective's own units, and the stop that reading gives.

use super::SolverStop;

/// The most a step along one parameter gains in log-likelihood beside a converged estimate: on
/// 117,768 fits the stationary ones read `2.3e-9` or less, and a stall anything up to 94.
const STATIONARY: f64 = 1e-7;

/// Roundings of the sum a gain must clear: on 366 fits of `1e4` to `3e6` points the walks beside
/// an optimum gain `0.30` of one or less, and beside the stalls they flag `1.2` or more.
const ROUNDINGS: f64 = 1.0;

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
  /// The rounding of the summed objective at the estimate, `ε √N (1 + |f|)`.
  pub(crate) rounding: f64,
  /// The first step of a walk met a point without a value.
  pub(crate) walled: bool,
}

/// Why a fit stopped, `Converged` alone being `converged = true`.
pub(crate) enum Stop {
  Converged,
  Search(SolverStop),
  NotStationary,
  Bound(usize),
  Edge,
  Outside,
}

impl Stop {
  /// The stop in the words of quant's calibrators, `names` the fit's parameters.
  pub(crate) fn message(&self, names: &[&str]) -> String {
    let words = match self {
      Self::Converged => "converged",
      Self::Search(SolverStop::IterationCap | SolverStop::EvaluationBudget) => {
        "evaluation budget exhausted"
      }
      Self::Search(SolverStop::NoProgress) => "no numerical progress",
      Self::Search(SolverStop::Failed) => "solver failed",
      Self::Search(_) | Self::Outside => "stopped without convergence",
      Self::NotStationary => "not stationary",
      Self::Bound(held) => return format!("stopped at the bound of {}", names[*held]),
      Self::Edge => "stopped at the edge of the law's support",
    };
    words.into()
  }
}

impl Beside {
  /// The stop the walks read, `held` saying that the parameter of the largest gain is on a bound.
  pub(crate) fn stop(&self, held: bool) -> Stop {
    match () {
      _ if self.gain <= STATIONARY.max(ROUNDINGS * self.rounding) => Stop::Converged,
      _ if held => Stop::Bound(self.along),
      _ if self.walled => Stop::Edge,
      _ => Stop::NotStationary,
    }
  }
}

/// What steps along one axis find beside `at`: from `ε^{1/3}` of its scale, either way, four times
/// further while `objective`, a sum of `additions` terms and `None` without a value, is not above.
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
    };
  };
  // A sum's rounding is no rise: below it the walk goes on, across a flat stretch.
  let rise = 16.0 * f64::EPSILON * (1.0 + here.abs());
  // `N` additions each round a partial sum near `|f|` by `ε/2`, and the roundings add as a walk.
  let mut beside = Beside {
    gain: 0.0,
    along: 0,
    rounding: f64::EPSILON * (additions as f64).sqrt() * (1.0 + here.abs()),
    walled: false,
  };
  for (i, axis) in axes.iter().enumerate() {
    for side in [1.0, -1.0] {
      let mut step = side * f64::EPSILON.cbrt() * axis.scale;
      let mut moved = at.to_vec();
      for nth in 0..FURTHEST {
        let to = (at[i] + step).clamp(axis.low, axis.high);
        if to == moved[i] || !to.is_finite() {
          break;
        }
        moved[i] = to;
        let cost = objective(&moved);
        beside.walled |= nth == 0 && cost.is_none();
        let Some(cost) = cost.filter(|cost| *cost <= here + rise) else {
          break;
        };
        if here - cost > beside.gain {
          (beside.gain, beside.along) = (here - cost, i);
        }
        step *= 4.0;
      }
    }
  }
  beside
}
