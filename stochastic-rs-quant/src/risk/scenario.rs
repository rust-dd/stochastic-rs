//! Scenario analysis: a shock perturbs one risk factor, a scenario is a named set of shocks,
//! and a [`StressTest`] runs scenarios against one valuer for a per-scenario PnL.
//!
//! Reference: Federal Reserve Board (2019), "Dodd-Frank Act Stress Test (DFAST) Methodology"
//! Reference: EBA (2020), Final Draft RTS on the IMA, Article 325bh(3) of Regulation (EU) 575/2013

use std::fmt::Display;

use ndarray::Array1;

use crate::curves::DiscountCurve;
use crate::traits::RealExt;

/// Elementary risk-factor perturbation.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Shock<T: RealExt> {
  /// Add `value` to the factor (e.g. +100 bps to every zero rate).
  Additive(T),
  /// Multiply the factor by `value` (e.g. 1.10 = +10%).
  Multiplicative(T),
  /// Override the factor to `value`.
  Level(T),
}

impl<T: RealExt> Shock<T> {
  /// Apply the shock to `x`.
  pub fn apply(&self, x: T) -> T {
    match *self {
      Self::Additive(v) => x + v,
      Self::Multiplicative(v) => x * v,
      Self::Level(v) => v,
    }
  }
}

impl<T: RealExt> Display for Shock<T> {
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
#[non_exhaustive]
pub enum CurveShift<T: RealExt> {
  /// Parallel shift: add `amount` to every zero rate.
  Parallel(T),
  /// Twist: linear interpolation between `short_shift` (at `t=0`) and
  /// `long_shift` (at the longest node).
  Twist { short_shift: T, long_shift: T },
  /// Key-rate shift — additive bump of the zero rate at the single tenor
  /// `pillar` only (with no contagion to neighbours).
  KeyRate { pillar: T, amount: T },
  /// An additive shift per curve pillar, aligned to the pillars.
  AtPillars(Array1<T>),
}

impl<T: RealExt> CurveShift<T> {
  /// Apply the shift to the provided discount curve, returning a new
  /// [`DiscountCurve`] that preserves the original interpolation method.
  pub fn apply(&self, base: &DiscountCurve<T>) -> DiscountCurve<T> {
    let points = base.points();
    let n = points.len();
    let times: Array1<T> = Array1::from_iter(points.iter().map(|p| p.time));
    let base_rates: Array1<T> =
      Array1::from_iter(points.iter().map(|p| -p.discount_factor.ln() / p.time));

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

fn nearest_pillar_index<T: RealExt>(times: &Array1<T>, target: T) -> usize {
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
pub struct Scenario<T: RealExt> {
  /// Human-readable scenario name.
  pub name: String,
  /// Optional free-form tags (e.g. regulator, region).
  pub tags: Vec<String>,
  /// Named shocks keyed by factor label.
  pub shocks: Vec<(String, Shock<T>)>,
  /// Curve shifts applied by curve label.
  pub curve_shifts: Vec<(String, CurveShift<T>)>,
}

impl<T: RealExt> Scenario<T> {
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
  pub fn with_curve_shift(mut self, curve_name: impl Into<String>, shift: CurveShift<T>) -> Self {
    self.curve_shifts.push((curve_name.into(), shift));
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
  pub fn resolve_curve(&self, curve_name: &str, base: &DiscountCurve<T>) -> DiscountCurve<T> {
    let mut curve = base.clone();
    for (key, shift) in &self.curve_shifts {
      if key == curve_name {
        curve = shift.apply(&curve);
      }
    }
    curve
  }
}

/// Outcome of a single scenario evaluation.
#[derive(Debug, Clone)]
pub struct ScenarioResult<T: RealExt> {
  /// Scenario name.
  pub name: String,
  /// Portfolio value under the base state.
  pub base_value: T,
  /// Portfolio value under the shocked state.
  pub shocked_value: T,
  /// PnL = shocked − base.
  pub pnl: T,
}

/// Runs scenarios against the caller's `Fn(&Scenario<T>) -> T` valuer, which applies the
/// scenario's shocks and curve shifts and returns the portfolio value under them.
#[derive(Debug, Clone)]
pub struct StressTest<T: RealExt> {
  scenarios: Vec<Scenario<T>>,
}

impl<T: RealExt> StressTest<T> {
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
