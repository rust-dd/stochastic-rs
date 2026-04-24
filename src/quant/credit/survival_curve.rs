//! Survival probability curves and piecewise-constant hazard rates.
//!
//! Reference: Brigo & Mercurio, "Interest Rate Models — Theory and Practice",
//! Springer, 2nd ed. (2006), Chapter 22.
//!
//! Reference: O'Kane & Turnbull, "Valuation of Credit Default Swaps",
//! Lehman Brothers Quantitative Credit Research Quarterly (2003).
//!
//! The risk-neutral survival probability is related to the hazard rate by
//! $$
//! Q(t)=\mathbb{P}(\tau>t)=\exp\!\left(-\int_0^t h(s)\,ds\right),
//! $$
//! and the instantaneous forward hazard between two grid times $t_1<t_2$ is
//! $$
//! h(t_1,t_2)=-\frac{\ln Q(t_2)-\ln Q(t_1)}{t_2-t_1}.
//! $$

use std::fmt::Display;

use ndarray::Array1;

use crate::traits::FloatExt;

/// Interpolation rule for survival probabilities between calibrated nodes.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HazardInterpolation {
  /// Linear interpolation on $\ln Q(t)$. Equivalent to piecewise-constant
  /// forward hazard between grid points and is the ISDA standard choice.
  #[default]
  PiecewiseConstantHazard,
  /// Linear interpolation on the survival probability $Q(t)$ itself.
  LinearSurvival,
}

impl Display for HazardInterpolation {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::PiecewiseConstantHazard => write!(f, "Piecewise-constant hazard"),
      Self::LinearSurvival => write!(f, "Linear survival"),
    }
  }
}

/// Single calibrated node of a survival curve.
#[derive(Debug, Clone, Copy)]
pub struct SurvivalPoint<T: FloatExt> {
  /// Year fraction from the valuation date.
  pub time: T,
  /// Risk-neutral survival probability $Q(t)$.
  pub survival_probability: T,
}

/// Calibrated survival curve $Q(t)=\mathbb{P}(\tau>t)$.
#[derive(Debug, Clone)]
pub struct SurvivalCurve<T: FloatExt> {
  points: Vec<SurvivalPoint<T>>,
  method: HazardInterpolation,
}

impl<T: FloatExt> SurvivalCurve<T> {
  /// Build a survival curve from sorted `(time, survival)` nodes.
  ///
  /// Nodes are automatically sorted; a `(0, 1)` anchor is inserted if absent.
  pub fn new(points: Vec<SurvivalPoint<T>>, method: HazardInterpolation) -> Self {
    let mut pts = points;
    pts.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    if pts.is_empty() || pts[0].time > T::zero() {
      pts.insert(
        0,
        SurvivalPoint {
          time: T::zero(),
          survival_probability: T::one(),
        },
      );
    }
    Self {
      points: pts,
      method,
    }
  }

  /// Build from parallel arrays of times and survival probabilities.
  pub fn from_survival_probs(
    times: &Array1<T>,
    survivals: &Array1<T>,
    method: HazardInterpolation,
  ) -> Self {
    let points = times
      .iter()
      .zip(survivals.iter())
      .map(|(&t, &q)| SurvivalPoint {
        time: t,
        survival_probability: q,
      })
      .collect();
    Self::new(points, method)
  }

  /// Build from parallel arrays of times and cumulative default probabilities.
  pub fn from_default_probs(
    times: &Array1<T>,
    defaults: &Array1<T>,
    method: HazardInterpolation,
  ) -> Self {
    let survivals: Array1<T> = defaults.mapv(|p| T::one() - p);
    Self::from_survival_probs(times, &survivals, method)
  }

  /// Build from parallel arrays of times and piecewise-constant forward hazard
  /// rates, where `hazards[i]` applies on $[t_{i-1}, t_i)$ (the first entry is
  /// used on $[0, t_0)$).
  pub fn from_hazard_rates(
    times: &Array1<T>,
    hazards: &Array1<T>,
    method: HazardInterpolation,
  ) -> Self {
    assert_eq!(
      times.len(),
      hazards.len(),
      "times and hazards must have matching lengths"
    );
    let mut survivals = Array1::zeros(times.len());
    let mut cum_log = T::zero();
    let mut prev_t = T::zero();
    for i in 0..times.len() {
      let dt = times[i] - prev_t;
      cum_log -= hazards[i] * dt;
      survivals[i] = cum_log.exp();
      prev_t = times[i];
    }
    Self::from_survival_probs(times, &survivals, method)
  }

  /// Number of calibrated points (including the anchor at `t=0`).
  pub fn len(&self) -> usize {
    self.points.len()
  }

  /// Whether the curve has no user-supplied nodes.
  pub fn is_empty(&self) -> bool {
    self.points.len() <= 1
  }

  /// Reference to the calibrated nodes.
  pub fn points(&self) -> &[SurvivalPoint<T>] {
    &self.points
  }

  /// Interpolation rule.
  pub fn method(&self) -> HazardInterpolation {
    self.method
  }

  /// Interpolated survival probability $Q(t)$.
  pub fn survival_probability(&self, t: T) -> T {
    if t <= T::zero() {
      return T::one();
    }

    let last = self.points.last().expect("survival curve has an anchor");
    if t >= last.time {
      return extrapolate_flat_hazard(last.time, last.survival_probability, t);
    }

    let idx = self
      .points
      .partition_point(|p| p.time < t)
      .saturating_sub(1);
    let p0 = &self.points[idx];
    let p1 = &self.points[idx + 1];

    match self.method {
      HazardInterpolation::PiecewiseConstantHazard => {
        let ln0 = p0.survival_probability.ln();
        let ln1 = p1.survival_probability.ln();
        let w = (t - p0.time) / (p1.time - p0.time);
        (ln0 * (T::one() - w) + ln1 * w).exp()
      }
      HazardInterpolation::LinearSurvival => {
        let w = (t - p0.time) / (p1.time - p0.time);
        p0.survival_probability * (T::one() - w) + p1.survival_probability * w
      }
    }
  }

  /// Cumulative default probability $1-Q(t)$.
  pub fn default_probability(&self, t: T) -> T {
    T::one() - self.survival_probability(t)
  }

  /// Conditional default probability over $(t_1, t_2]$.
  ///
  /// $$\mathbb{P}(t_1<\tau\le t_2\mid\tau>t_1)=1-\frac{Q(t_2)}{Q(t_1)}.$$
  pub fn conditional_default_probability(&self, t1: T, t2: T) -> T {
    if t2 <= t1 {
      return T::zero();
    }
    let q1 = self.survival_probability(t1);
    if q1 <= T::min_positive_val() {
      return T::zero();
    }
    T::one() - self.survival_probability(t2) / q1
  }

  /// Piecewise-constant forward hazard rate between `t1` and `t2`.
  pub fn forward_hazard(&self, t1: T, t2: T) -> T {
    if t2 <= t1 {
      return T::zero();
    }
    let q1 = self.survival_probability(t1);
    let q2 = self.survival_probability(t2);
    if q1 <= T::min_positive_val() || q2 <= T::min_positive_val() {
      return T::zero();
    }
    -(q2.ln() - q1.ln()) / (t2 - t1)
  }

  /// Flat-equivalent continuous hazard $-\ln Q(t)/t$.
  pub fn average_hazard(&self, t: T) -> T {
    if t <= T::zero() {
      return T::zero();
    }
    let q = self.survival_probability(t);
    if q <= T::min_positive_val() {
      return T::zero();
    }
    -q.ln() / t
  }

  /// Survival probabilities at the requested maturities.
  pub fn survival_probabilities(&self, maturities: &Array1<T>) -> Array1<T> {
    Array1::from_iter(maturities.iter().map(|&t| self.survival_probability(t)))
  }

  /// Default probabilities at the requested maturities.
  pub fn default_probabilities(&self, maturities: &Array1<T>) -> Array1<T> {
    Array1::from_iter(maturities.iter().map(|&t| self.default_probability(t)))
  }
}

/// Hazard rate curve is a thin view on a [`SurvivalCurve`] that exposes only
/// the forward/average hazard surface.
#[derive(Debug, Clone)]
pub struct HazardRateCurve<T: FloatExt> {
  inner: SurvivalCurve<T>,
}

impl<T: FloatExt> HazardRateCurve<T> {
  /// Wrap an existing survival curve.
  pub fn new(survival: SurvivalCurve<T>) -> Self {
    Self { inner: survival }
  }

  /// Build from parallel arrays of times and piecewise-constant hazards.
  pub fn from_hazard_rates(
    times: &Array1<T>,
    hazards: &Array1<T>,
    method: HazardInterpolation,
  ) -> Self {
    Self::new(SurvivalCurve::from_hazard_rates(times, hazards, method))
  }

  /// Underlying survival curve.
  pub fn survival_curve(&self) -> &SurvivalCurve<T> {
    &self.inner
  }

  /// Piecewise-constant forward hazard between `t1` and `t2`.
  pub fn forward_hazard(&self, t1: T, t2: T) -> T {
    self.inner.forward_hazard(t1, t2)
  }

  /// Flat-equivalent hazard $-\ln Q(t)/t$.
  pub fn average_hazard(&self, t: T) -> T {
    self.inner.average_hazard(t)
  }

  /// Survival probability at time `t`.
  pub fn survival_probability(&self, t: T) -> T {
    self.inner.survival_probability(t)
  }
}

/// Flat-hazard extrapolation beyond the last calibrated node.
fn extrapolate_flat_hazard<T: FloatExt>(last_time: T, last_survival: T, t: T) -> T {
  if last_time <= T::zero() {
    return last_survival;
  }
  let h = -last_survival.ln() / last_time;
  (-h * t).exp()
}
