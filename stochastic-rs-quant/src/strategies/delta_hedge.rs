//! # Delta Hedge
//!
//! $$
//! \Delta_t=\partial_S V(t,S_t),\quad d\Pi_t=dV_t-\Delta_t dS_t
//! $$
//!
use crate::Moneyness;
use crate::OptionType;

/// The option being hedged and the moneyness thresholds that classify it.
#[derive(Debug, Clone, Copy)]
pub struct DeltaHedgeSpec {
  /// Current option value used by strategy/PnL routines.
  pub c: f64,
  /// The option's premium
  pub c_premium: f64,
  /// The option's delta
  pub c_delta: f64,
  /// Strike price
  pub k: f64,
  /// Spot price.
  pub s: f64,
  /// Initial spot price.
  pub s0: f64,
  /// The size of the option contract
  pub contract_size: f64,
  /// Call or put being hedged.
  pub option_type: OptionType,
  /// Deep out-of-the-money threshold
  pub dotm_threshold: f64,
  /// Deep in-the-money threshold
  pub ditm_threshold: f64,
  /// At-the-money threshold
  pub atm_threshold: f64,
  /// In-the-money threshold
  pub itm_threshold: f64,
}

/// Delta hedge of one option position: `c_delta · contract_size` shares against the premium.
#[derive(Debug, Clone, Copy)]
pub struct DeltaHedge {
  /// The hedged option and its thresholds.
  pub spec: DeltaHedgeSpec,
}

impl DeltaHedge {
  pub fn new(spec: DeltaHedgeSpec) -> Self {
    Self { spec }
  }

  pub fn hedge_cost(&self) -> f64 {
    self.hedge_shares() * self.spec.s - (self.spec.c_premium * self.spec.contract_size)
  }

  pub fn current_hedge_profit(&self) -> f64 {
    let option_profit = (self.spec.c_premium - self.spec.c) * self.spec.contract_size;
    let stock_profit = (self.spec.s - self.spec.s0) * self.hedge_shares();
    option_profit + stock_profit
  }

  pub fn hedge_shares(&self) -> f64 {
    self.spec.c_delta * self.spec.contract_size
  }

  /// Classify the option as DOTM / OTM / ATM / ITM / DITM using
  /// asymmetric thresholds.
  ///
  /// **Threshold semantics:** `moneyness_ratio = S/K` for a call,
  /// `K/S` for a put — so the side that "goes ITM" always sees a ratio
  /// above 1.
  /// - `ditm_threshold` is a one-sided **upper** cut: `ratio >
  ///   ditm_threshold` ⇒ Deep-ITM. Typical: 1.10 (10% past strike).
  /// - `dotm_threshold` is a one-sided **lower** cut: `ratio <
  ///   dotm_threshold` ⇒ Deep-OTM. Typical: 0.90.
  /// - `atm_threshold` and `itm_threshold` together form a symmetric
  ///   ATM band around 1.0 (`atm_threshold` close to 1, e.g. 0.99,
  ///   makes the ATM band wide).
  ///
  /// These thresholds are **deliberately one-sided**: a 1.05 ratio is
  /// "in the money" for a call but the implementation does not test
  /// `1.0 - x` symmetrically below 1, since the OTM bucket already
  /// covers that side.
  pub fn moneyness(&self) -> Moneyness {
    let spec = &self.spec;
    let moneyness_ratio = match spec.option_type {
      OptionType::Call => spec.s / spec.k,
      OptionType::Put => spec.k / spec.s,
    };

    if moneyness_ratio > spec.ditm_threshold {
      Moneyness::DeepInTheMoney
    } else if moneyness_ratio > 1.0 {
      Moneyness::InTheMoney
    } else if (moneyness_ratio - spec.itm_threshold).abs() <= (1.0 - spec.atm_threshold) {
      Moneyness::AtTheMoney
    } else if moneyness_ratio < spec.dotm_threshold {
      Moneyness::DeepOutOfTheMoney
    } else {
      Moneyness::OutOfTheMoney
    }
  }
}
