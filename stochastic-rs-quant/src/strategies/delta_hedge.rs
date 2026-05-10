//! # Delta Hedge
//!
//! $$
//! \Delta_t=\partial_S V(t,S_t),\quad d\Pi_t=dV_t-\Delta_t dS_t
//! $$
//!
use crate::Moneyness;
use crate::OptionType;

#[derive(Debug, Clone)]
pub struct DeltaHedge {
  /// Current option value used by strategy/PnL routines.
  pub c: f64,
  /// The option's premium
  pub c_premium: f64,
  /// The option's delta
  pub c_delta: f64,
  /// Strike price
  pub k: f64,
  /// Stock price
  pub s: f64,
  /// Initial stock price
  pub s0: f64,
  /// The size of the option contract
  pub contract_size: f64,
  /// Hedge size
  pub hedge_size: f64,
  /// Option type
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

impl DeltaHedge {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    c: f64,
    c_premium: f64,
    c_delta: f64,
    k: f64,
    s: f64,
    s0: f64,
    contract_size: f64,
    hedge_size: f64,
    option_type: OptionType,
    dotm_threshold: f64,
    ditm_threshold: f64,
    atm_threshold: f64,
    itm_threshold: f64,
  ) -> Self {
    Self {
      c,
      c_premium,
      c_delta,
      k,
      s,
      s0,
      contract_size,
      hedge_size,
      option_type,
      dotm_threshold,
      ditm_threshold,
      atm_threshold,
      itm_threshold,
    }
  }
}

impl DeltaHedge {
  pub fn hedge_cost(&self) -> f64 {
    self.hedge_shares() * self.s - (self.c_premium * self.contract_size)
  }

  pub fn current_hedge_profit(&self) -> f64 {
    let option_profit = (self.c_premium - self.c) * self.contract_size;
    let stock_profit = (self.s - self.s0) * self.hedge_shares();
    option_profit + stock_profit
  }

  pub fn hedge_shares(&self) -> f64 {
    self.c_delta * self.contract_size
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
    let moneyness_ratio = match self.option_type {
      OptionType::Call => self.s / self.k,
      OptionType::Put => self.k / self.s,
    };

    if moneyness_ratio > self.ditm_threshold {
      Moneyness::DeepInTheMoney
    } else if moneyness_ratio > 1.0 {
      Moneyness::InTheMoney
    } else if (moneyness_ratio - self.itm_threshold).abs() <= (1.0 - self.atm_threshold) {
      Moneyness::AtTheMoney
    } else if moneyness_ratio < self.dotm_threshold {
      Moneyness::DeepOutOfTheMoney
    } else {
      Moneyness::OutOfTheMoney
    }
  }
}
