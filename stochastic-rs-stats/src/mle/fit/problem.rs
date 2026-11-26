use std::convert::Infallible;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

use basin::BoxBounds;
use basin::ConstraintDescription;
use basin::ConstraintSet;
use basin::Constraints;
use basin::CostFunction;
use basin::Gradient;
use ndarray::ArrayView1;
use parking_lot::Mutex;

use super::step_back::Walled;
use crate::mle::DiffusionModel;
use crate::mle::density::DensityApprox;
use crate::optim::certificate::Axis;
use crate::optim::certificate::Beside;
use crate::optim::certificate::beside;

/// What a move of one parameter alone to a box end may cost and still count as loose: half of
/// `χ²₁(0.95)`. It picks a restart's point and words no fit.
const LOOSE: f64 = 1.92;

/// The objective at a point without a likelihood, above every sum [`ranked`] returns.
pub(super) const NO_LIKELIHOOD: f64 = 1e100;

/// A line search is not running, has met only points with a likelihood, or has met one without.
const IDLE: u8 = 0;
const CLEAN: u8 = 1;
const WALLED: u8 = 2;

/// `nll` up to `1e97`, far past any realistic path's sum, and its logarithmic continuation past it,
/// at most `4.9e99`: increasing and C¹, so a finite sum ranks below the penalty and none is `∞`.
pub(super) fn ranked(nll: f64) -> f64 {
  const KNEE: f64 = 1e97;
  if nll <= KNEE {
    nll
  } else {
    KNEE * (1.0 + (nll / KNEE).ln())
  }
}

/// The solver's coordinates: each parameter over its unit, with the caller's box in them.
#[derive(Clone)]
pub(super) struct Units {
  box_low: Vec<f64>,
  box_high: Vec<f64>,
  scale: Vec<f64>,
  low: Vec<f64>,
  high: Vec<f64>,
}

/// The ratio to its unit, either way, past which a parameter's estimate asks for new units.
const FAR: f64 = 8.0;

/// The unit the path gives a parameter that brings none, never an `Err`: its standard deviation,
/// its largest size where the deviations' squares underflow, 1 for a path of zeros.
fn path_unit(sample: ArrayView1<f64>) -> f64 {
  let size = sample.iter().fold(0.0_f64, |size, v| size.max(v.abs()));
  if size == 0.0 {
    return 1.0;
  }
  let n = sample.len() as f64;
  let mean = sample.iter().map(|v| v / size).sum::<f64>() / n;
  let variance = sample
    .iter()
    .map(|v| (v / size - mean).powi(2))
    .sum::<f64>()
    / n;
  if variance > 0.0 {
    size * variance.sqrt()
  } else {
    size
  }
}

impl Units {
  /// The units of `start`: `|start|` for each parameter, and for one that starts at zero half
  /// its box's width, or `path_unit`, where the box has no finite width.
  fn new(lower: &[f64], upper: &[f64], start: &[f64], path_unit: f64) -> Self {
    let scale = (0..lower.len())
      .map(|i| {
        let (low, high) = (lower[i], upper[i]);
        let from = start[i].clamp(low, high);
        if from != 0.0 {
          from.abs()
        } else if low.is_finite() && high.is_finite() && high > low {
          0.5 * (high - low)
        } else {
          path_unit
        }
      })
      .collect::<Vec<_>>();
    Self {
      box_low: lower.to_vec(),
      box_high: upper.to_vec(),
      low: lower.iter().zip(&scale).map(|(l, u)| l / u).collect(),
      high: upper.iter().zip(&scale).map(|(h, u)| h / u).collect(),
      scale,
    }
  }

  /// Whether some parameter of `params` is more than [`FAR`] times from its unit, either way.
  pub(super) fn far(&self, params: &[f64]) -> bool {
    let near = 1.0 / FAR..=FAR;
    (params.iter().zip(&self.scale)).any(|(p, unit)| *p != 0.0 && !near.contains(&(p.abs() / unit)))
  }

  /// The solver's coordinates of `params`, inside its box.
  pub(super) fn of(&self, params: &[f64]) -> Vec<f64> {
    (0..params.len())
      .map(|i| {
        let inside = params[i].clamp(self.box_low[i], self.box_high[i]);
        (inside / self.scale[i]).clamp(self.low[i], self.high[i])
      })
      .collect()
  }

  /// The parameters at the solver's point `z`: inside the caller's box when `z` is inside the
  /// solver's, and as they fall otherwise, which the fit then reports unconverged.
  pub(super) fn params(&self, z: &[f64]) -> Vec<f64> {
    let inside = (0..z.len()).all(|i| (self.low[i]..=self.high[i]).contains(&z[i]));
    let params = self.scaled(z);
    if !inside {
      return params;
    }
    (0..z.len())
      .map(|i| params[i].clamp(self.box_low[i], self.box_high[i]))
      .collect()
  }

  /// The parameters at `z`; an end of the solver's box is the caller's bound to the bit, which
  /// the product with the unit may miss by a rounding.
  fn scaled(&self, z: &[f64]) -> Vec<f64> {
    (0..z.len())
      .map(|i| {
        if z[i] == self.low[i] {
          self.box_low[i]
        } else if z[i] == self.high[i] {
          self.box_high[i]
        } else {
          self.scale[i] * z[i]
        }
      })
      .collect()
  }
}

/// The negative log-likelihood over the box, which the solver reads in the units of a start.
pub(super) struct MleProblem<'a> {
  model: Mutex<&'a mut dyn DiffusionModel>,
  sample: ArrayView1<'a, f64>,
  dt: f64,
  density: DensityApprox,
  units: Units,
  probe: AtomicU8,
}

impl<'a> MleProblem<'a> {
  pub(super) fn new(
    model: &'a mut dyn DiffusionModel,
    sample: ArrayView1<'a, f64>,
    dt: f64,
    density: DensityApprox,
    lower: &[f64],
    upper: &[f64],
    start: &[f64],
  ) -> Self {
    Self {
      model: Mutex::new(model),
      sample,
      dt,
      density,
      units: Units::new(lower, upper, start, path_unit(sample)),
      probe: AtomicU8::new(IDLE),
    }
  }

  pub(super) fn units(&self) -> Units {
    self.units.clone()
  }

  /// With two or more loose moves, `at` with each such parameter back at `start`'s value: a move
  /// alone to a box end, or to a `start` left far behind for an infinite one, under [`LOOSE`].
  pub(super) fn loosened(&self, at: &[f64], start: &[f64]) -> Vec<f64> {
    let here = self.eval_nll(at);
    let mut from = at.to_vec();
    if here >= NO_LIKELIHOOD {
      return from;
    }
    let mut loose_ends = 0;
    for i in 0..at.len() {
      let ends = [self.units.box_low[i], self.units.box_high[i]];
      if ends[0] >= ends[1] {
        continue;
      }
      // An end at infinity has no likelihood to read: the start's value stands in, once, unless
      // the estimate is still by it, on its side of zero, where the move back says nothing.
      let left = !(1.0 / FAR..=FAR).contains(&(at[i] / start[i]));
      let open = (left && ends.iter().any(|end| end.is_infinite())).then_some(start[i]);
      let finite = ends.into_iter().filter(|end| end.is_finite());
      let loose = finite
        .chain(open)
        .filter(|&to| {
          let mut moved = at.to_vec();
          moved[i] = to;
          self.eval_nll(&moved) - here < LOOSE
        })
        .count();
      if loose > 0 {
        from[i] = start[i];
      }
      loose_ends += loose;
    }
    // One loose end is an optimum at or beside a bound; a stall leaves two, as a speed and a level.
    if loose_ends < 2 { at.to_vec() } else { from }
  }

  /// What steps along one parameter find beside `at`, in the problem's units: each axis varies on
  /// `1 + |z|` there, the gradient's own step. The point of the gain is in the parameters' units.
  pub(super) fn beside(&self, at: &[f64]) -> Beside {
    let z = self.units.of(at);
    let axis = |i: usize| Axis {
      low: self.units.low[i],
      high: self.units.high[i],
      scale: 1.0 + z[i].abs(),
    };
    let axes = (0..z.len()).map(axis).collect::<Vec<_>>();
    let valued =
      |z: &[f64]| Some(self.eval_nll(&self.units.scaled(z))).filter(|cost| *cost < NO_LIKELIHOOD);
    let mut read = beside(valued, &z, &axes, self.sample.len() - 1);
    read.best = self.units.params(&read.best);
    read
  }

  pub(super) fn eval_nll(&self, params: &[f64]) -> f64 {
    if self.probe.load(Ordering::Relaxed) == WALLED {
      return NO_LIKELIHOOD;
    }
    // Line-search probes can leave the box, even with a bounded solver.
    let params = (0..params.len())
      .map(|i| params[i].clamp(self.units.box_low[i], self.units.box_high[i]))
      .collect::<Vec<_>>();
    let mut model = self.model.lock();
    model.set_params(&params);
    let mut sum = 0.0;
    for i in 1..self.sample.len() {
      let t0 = (i - 1) as f64 * self.dt;
      let ln_d = self
        .density
        .ln_density(&**model, self.sample[i - 1], self.sample[i], t0, self.dt);
      // A density of zero, an infinite one and none at all are no likelihood.
      if !ln_d.is_finite() {
        return self.no_likelihood();
      }
      sum -= ln_d;
    }
    if sum.is_finite() {
      ranked(sum)
    } else {
      self.no_likelihood()
    }
  }

  fn no_likelihood(&self) -> f64 {
    let _ = self
      .probe
      .compare_exchange(CLEAN, WALLED, Ordering::Relaxed, Ordering::Relaxed);
    NO_LIKELIHOOD
  }
}

impl Walled for MleProblem<'_> {
  fn watch(&self, on: bool) {
    self
      .probe
      .store(if on { CLEAN } else { IDLE }, Ordering::Relaxed);
  }

  fn walled(&self) -> bool {
    self.probe.load(Ordering::Relaxed) == WALLED
  }
}

impl CostFunction for MleProblem<'_> {
  type Param = Vec<f64>;
  type Output = f64;
  type Error = Infallible;

  fn cost(&self, z: &Self::Param) -> Result<Self::Output, Self::Error> {
    Ok(self.eval_nll(&self.units.scaled(z)))
  }
}

impl Gradient for MleProblem<'_> {
  type Gradient = Vec<f64>;

  fn gradient(&self, z: &Self::Param) -> Result<Self::Gradient, Self::Error> {
    let (low, high) = (&self.units.low, &self.units.high);
    let z = (0..z.len())
      .map(|i| z[i].clamp(low[i], high[i]))
      .collect::<Vec<_>>();
    // The central difference's own optimum, `ε^{1/3}`, relative to the coordinate.
    let step = f64::EPSILON.cbrt();
    let mut grad = vec![0.0; z.len()];
    for i in 0..z.len() {
      let h = step * (1.0 + z[i].abs());
      let mut z_plus = z.clone();
      let mut z_minus = z.clone();
      z_plus[i] = (z[i] + h).min(high[i]);
      z_minus[i] = (z[i] - h).max(low[i]);
      let actual_2h = z_plus[i] - z_minus[i];
      if actual_2h > 0.0 {
        let fp = self.eval_nll(&self.units.scaled(&z_plus));
        let fm = self.eval_nll(&self.units.scaled(&z_minus));
        grad[i] = (fp - fm) / actual_2h;
      }
    }
    Ok(grad)
  }
}

impl Constraints for MleProblem<'_> {
  type Set = ConstraintSet<BoxBounds<Vec<f64>>>;

  fn constraints(&self) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
    ConstraintSet::new().with_bounds(BoxBounds::new(
      Some(&self.units.low),
      Some(&self.units.high),
    ))
  }
}
