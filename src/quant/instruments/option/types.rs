//! Common types shared by cap, floor, collar and swaption instruments.

use std::fmt::Display;

use chrono::NaiveDate;

use crate::traits::FloatExt;

/// Rate option payoff family.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterestRateOptionKind {
  /// Cap — call on the forward rate: payoff $(L-K)^+$ per caplet.
  #[default]
  Cap,
  /// Floor — put on the forward rate: payoff $(K-L)^+$ per floorlet.
  Floor,
}

impl Display for InterestRateOptionKind {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Cap => write!(f, "Cap"),
      Self::Floor => write!(f, "Floor"),
    }
  }
}

/// Swaption payoff direction.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SwaptionDirection {
  /// Pay fixed, receive floating — call on the forward swap rate.
  #[default]
  Payer,
  /// Receive fixed, pay floating — put on the forward swap rate.
  Receiver,
}

impl Display for SwaptionDirection {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Payer => write!(f, "Payer"),
      Self::Receiver => write!(f, "Receiver"),
    }
  }
}

/// Family a volatility quote belongs to.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VolatilityQuoteKind {
  /// Lognormal (Black-76) volatility.
  #[default]
  Lognormal,
  /// Normal (Bachelier) volatility.
  Normal,
}

impl Display for VolatilityQuoteKind {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Lognormal => write!(f, "Lognormal"),
      Self::Normal => write!(f, "Normal"),
    }
  }
}

/// Cap / Floor valuation summary.
#[derive(Debug, Clone)]
pub struct CapFloorValuation<T: FloatExt> {
  /// Total dirty present value.
  pub npv: T,
  /// Discounted cap/floor annuity $\sum_i P(t,t_i)\,\alpha_i\,N_i$.
  pub annuity: T,
  /// Per-caplet / per-floorlet prices in chronological order.
  pub caplet_prices: Vec<T>,
  /// Per-caplet forward rates in chronological order.
  pub forward_rates: Vec<T>,
  /// Per-caplet accrual factors in chronological order.
  pub accrual_factors: Vec<T>,
}

/// Collar valuation summary.
#[derive(Debug, Clone)]
pub struct CollarValuation<T: FloatExt> {
  /// Total net present value (long cap minus long floor, or vice versa).
  pub npv: T,
  /// Cap leg valuation.
  pub cap: CapFloorValuation<T>,
  /// Floor leg valuation.
  pub floor: CapFloorValuation<T>,
}

/// European swaption valuation summary.
#[derive(Debug, Clone)]
pub struct SwaptionValuation<T: FloatExt> {
  /// Present value at the valuation date.
  pub npv: T,
  /// Forward swap rate implied by the current curves.
  pub forward_swap_rate: T,
  /// Fixed-leg annuity $\sum_j D(t_j)\,\delta_j\,N_j$.
  pub annuity: T,
  /// Year fraction from valuation date to exercise.
  pub tau: T,
  /// Effective volatility used by the pricer.
  pub volatility: T,
  /// Quote family of the volatility above.
  pub volatility_quote: VolatilityQuoteKind,
}

/// Bermudan swaption valuation summary.
#[derive(Debug, Clone)]
pub struct BermudanSwaptionValuation<T: FloatExt> {
  /// Present value at the tree root.
  pub npv: T,
  /// Number of exercise dates considered.
  pub exercise_count: usize,
}

/// Exercise schedule for a Bermudan swaption expressed as tree levels.
#[derive(Debug, Clone)]
pub struct ExerciseSchedule {
  /// Sorted, unique tree levels at which the holder may exercise.
  pub levels: Vec<usize>,
}

impl ExerciseSchedule {
  /// Build an exercise schedule from tree levels.
  pub fn new(mut levels: Vec<usize>) -> Self {
    levels.sort_unstable();
    levels.dedup();
    Self { levels }
  }

  /// True when the schedule contains the given level.
  pub fn contains(&self, level: usize) -> bool {
    self.levels.binary_search(&level).is_ok()
  }
}

/// Fixed coupon schedule for a tree-priced swaption expressed as tree levels.
#[derive(Debug, Clone)]
pub struct TreeCouponSchedule<T: FloatExt> {
  /// Tree levels at which fixed-leg coupons are paid.
  pub levels: Vec<usize>,
  /// Accrual factors $\delta_j$ in the same order as the levels.
  pub accrual_factors: Vec<T>,
}

impl<T: FloatExt> TreeCouponSchedule<T> {
  /// Build a coupon schedule; levels are kept in input order.
  pub fn new(levels: Vec<usize>, accrual_factors: Vec<T>) -> Self {
    assert_eq!(
      levels.len(),
      accrual_factors.len(),
      "coupon levels and accrual factors must have the same length"
    );
    Self {
      levels,
      accrual_factors,
    }
  }
}

/// Discrete exercise date paired with a tree level index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExerciseDate {
  /// Calendar exercise date.
  pub date: NaiveDate,
  /// Tree level at which the holder may exercise.
  pub level: usize,
}
