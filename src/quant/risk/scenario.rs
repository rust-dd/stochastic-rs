//! Scenario analysis and stress-testing.
//!
//! Reference: Board of Governors of the Federal Reserve System, "Dodd-Frank
//! Act Stress Test (DFAST) Methodology" (2019).
//!
//! Reference: EBA, "Final Draft Regulatory Technical Standards on the IMA
//! under Article 325bh(3) of Regulation (EU) No 575/2013" (2020).
//!
//! A *shock* is an elementary perturbation of a risk factor (additive bump,
//! multiplicative scaling, or absolute-level override).  A *scenario* is a
//! named collection of shocks plus an arbitrary closure that applies them to
//! a user-supplied portfolio valuer.  A [`StressTest`] runs a list of
//! scenarios against the same valuer and returns a per-scenario PnL.

use std::fmt::Display;

use ndarray::Array1;

use crate::quant::curves::DiscountCurve;
use crate::traits::FloatExt;

/// Elementary risk-factor perturbation.
#[derive(Debug, Clone, Copy)]
pub enum Shock<T: FloatExt> {
  /// Add `value` to the factor (e.g. +100 bps to every zero rate).
  Additive(T),
  /// Multiply the factor by `value` (e.g. 1.10 = +10%).
  Multiplicative(T),
  /// Override the factor to `value`.
  Level(T),
}

impl<T: FloatExt> Shock<T> {
  /// Apply the shock to `x`.
  pub fn apply(&self, x: T) -> T {
    match *self {
      Self::Additive(v) => x + v,
      Self::Multiplicative(v) => x * v,
      Self::Level(v) => v,
    }
  }
}

impl<T: FloatExt> Display for Shock<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Additive(v) => write!(f, "+{v:?}"),
      Self::Multiplicative(v) => write!(f, "×{v:?}"),
      Self::Level(v) => write!(f, "={v:?}"),
    }
  }
}

/// Deterministic parallel / key-rate / user-supplied yield curve shift.
#[derive(Debug, Clone)]
pub enum CurveShift<T: FloatExt> {
  /// Parallel shift: add `amount` to every zero rate.
  Parallel(T),
  /// Twist: linear interpolation between `short_shift` (at `t=0`) and
  /// `long_shift` (at the longest node).
  Twist { short_shift: T, long_shift: T },
  /// Key-rate shift — additive bump of the zero rate at the single tenor
  /// `pillar` only (with no contagion to neighbours).
  KeyRate { pillar: T, amount: T },
  /// Arbitrary user-supplied shift function `r -> r_shocked` as an additive
  /// delta at every pillar, expressed as an `ndarray::Array1` aligned to the
  /// curve pillars.
  AtPillars(Array1<T>),
}

impl<T: FloatExt> CurveShift<T> {
  /// Apply the shift to the provided discount curve, returning a new
  /// [`DiscountCurve`] that preserves the original interpolation method.
  pub fn apply(&self, base: &DiscountCurve<T>) -> DiscountCurve<T> {
    let points = base.points();
    let n = points.len();
    let times: Array1<T> = Array1::from_iter(points.iter().map(|p| p.time));
    let base_rates: Array1<T> = Array1::from_iter(points.iter().map(|p| {
      if p.time > T::zero() {
        -p.discount_factor.ln() / p.time
      } else {
        T::zero()
      }
    }));

    let shifted_rates = match self {
      Self::Parallel(amount) => base_rates.mapv(|r| r + *amount),
      Self::Twist {
        short_shift,
        long_shift,
      } => {
        let t_max = times[n - 1].max(T::min_positive_val());
        base_rates
          .iter()
          .zip(times.iter())
          .map(|(&r, &t)| {
            let w = t / t_max;
            let shift = *short_shift * (T::one() - w) + *long_shift * w;
            r + shift
          })
          .collect::<Array1<T>>()
      }
      Self::KeyRate { pillar, amount } => {
        let mut out = base_rates.clone();
        let idx = nearest_pillar_index(&times, *pillar);
        out[idx] += *amount;
        out
      }
      Self::AtPillars(deltas) => {
        assert_eq!(
          deltas.len(),
          n,
          "AtPillars shift length {} must match curve pillar count {n}",
          deltas.len()
        );
        &base_rates + deltas
      }
    };

    DiscountCurve::from_zero_rates(&times, &shifted_rates, base.method())
  }
}

fn nearest_pillar_index<T: FloatExt>(times: &Array1<T>, target: T) -> usize {
  let mut best = 0usize;
  let mut best_diff = (times[0] - target).abs();
  for (i, &t) in times.iter().enumerate().skip(1) {
    let d = (t - target).abs();
    if d < best_diff {
      best_diff = d;
      best = i;
    }
  }
  best
}

/// Single stress scenario.
#[derive(Debug, Clone)]
pub struct Scenario<T: FloatExt> {
  /// Human-readable scenario name.
  pub name: String,
  /// Optional free-form tags (e.g. regulator, region).
  pub tags: Vec<String>,
  /// Named shocks keyed by factor label.
  pub shocks: Vec<(String, Shock<T>)>,
  /// Curve shifts applied by curve label.
  pub curve_shifts: Vec<(String, CurveShift<T>)>,
}

impl<T: FloatExt> Scenario<T> {
  /// Construct an empty scenario.
  pub fn new(name: impl Into<String>) -> Self {
    Self {
      name: name.into(),
      tags: Vec::new(),
      shocks: Vec::new(),
      curve_shifts: Vec::new(),
    }
  }

  /// Attach a scalar shock keyed by factor.
  pub fn with_shock(mut self, factor: impl Into<String>, shock: Shock<T>) -> Self {
    self.shocks.push((factor.into(), shock));
    self
  }

  /// Attach a curve shift keyed by curve name.
  pub fn with_curve_shift(
    mut self,
    curve_key: impl Into<String>,
    shift: CurveShift<T>,
  ) -> Self {
    self.curve_shifts.push((curve_key.into(), shift));
    self
  }

  /// Add a free-form tag.
  pub fn tagged(mut self, tag: impl Into<String>) -> Self {
    self.tags.push(tag.into());
    self
  }

  /// Resolve a scalar factor value by applying every matching shock in order.
  pub fn resolve_scalar(&self, factor: &str, base: T) -> T {
    let mut x = base;
    for (name, shock) in &self.shocks {
      if name == factor {
        x = shock.apply(x);
      }
    }
    x
  }

  /// Resolve a curve by applying every matching curve shift in order.
  pub fn resolve_curve(&self, curve_key: &str, base: &DiscountCurve<T>) -> DiscountCurve<T> {
    let mut curve = base.clone();
    for (key, shift) in &self.curve_shifts {
      if key == curve_key {
        curve = shift.apply(&curve);
      }
    }
    curve
  }
}

/// Outcome of a single scenario evaluation.
#[derive(Debug, Clone)]
pub struct ScenarioResult<T: FloatExt> {
  /// Scenario name.
  pub name: String,
  /// Portfolio value under the base state.
  pub base_value: T,
  /// Portfolio value under the shocked state.
  pub shocked_value: T,
  /// PnL = shocked − base.
  pub pnl: T,
}

/// Stress-test engine — runs a set of scenarios against a `Fn(&Scenario<T>) -> T`
/// valuer supplied by the caller.  The valuer closure is responsible for
/// consuming any relevant shocks / curve shifts from the scenario and
/// returning the portfolio value under the shocked state.
#[derive(Debug, Clone)]
pub struct StressTest<T: FloatExt> {
  scenarios: Vec<Scenario<T>>,
}

impl<T: FloatExt> StressTest<T> {
  /// Build a stress test from an explicit scenario list.
  pub fn new(scenarios: Vec<Scenario<T>>) -> Self {
    Self { scenarios }
  }

  /// Add one more scenario.
  pub fn push(&mut self, scenario: Scenario<T>) {
    self.scenarios.push(scenario);
  }

  /// Borrow the scenarios.
  pub fn scenarios(&self) -> &[Scenario<T>] {
    &self.scenarios
  }

  /// Run every scenario and collect results.  `base_value` is evaluated once,
  /// `scenario_value(s)` is called for every scenario.
  pub fn run<F, G>(&self, base_value: F, mut scenario_value: G) -> Vec<ScenarioResult<T>>
  where
    F: Fn() -> T,
    G: FnMut(&Scenario<T>) -> T,
  {
    let base = base_value();
    self
      .scenarios
      .iter()
      .map(|s| {
        let shocked = scenario_value(s);
        ScenarioResult {
          name: s.name.clone(),
          base_value: base,
          shocked_value: shocked,
          pnl: shocked - base,
        }
      })
      .collect()
  }
}
