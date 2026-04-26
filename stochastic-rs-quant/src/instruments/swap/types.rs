use std::fmt::Display;

use crate::traits::FloatExt;

/// Swap direction.
///
/// `Payer` means pay fixed / receive floating.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SwapDirection {
  #[default]
  Payer,
  Receiver,
}

impl Display for SwapDirection {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Payer => write!(f, "Payer"),
      Self::Receiver => write!(f, "Receiver"),
    }
  }
}

/// Direction for cross-currency swaps quoted in the domestic currency.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CrossCurrencySwapDirection {
  #[default]
  PayDomesticReceiveForeign,
  ReceiveDomesticPayForeign,
}

impl Display for CrossCurrencySwapDirection {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::PayDomesticReceiveForeign => write!(f, "Pay domestic / receive foreign"),
      Self::ReceiveDomesticPayForeign => write!(f, "Receive domestic / pay foreign"),
    }
  }
}

/// Standard vanilla IRS valuation summary.
#[derive(Debug, Clone)]
pub struct SwapValuation<T: FloatExt> {
  /// Present value of the fixed leg.
  pub fixed_leg_npv: T,
  /// Present value of the floating leg.
  pub floating_leg_npv: T,
  /// Net swap present value under the swap direction.
  pub net_npv: T,
  /// Fair fixed rate equating both legs.
  pub fair_rate: T,
  /// Discounted fixed-leg annuity.
  pub annuity: T,
  /// Absolute basis-point value of the fixed rate.
  pub bpv: T,
  /// Signed DV01 with respect to a 1 bp fixed-rate bump.
  pub dv01: T,
}

/// Basis-swap valuation summary.
#[derive(Debug, Clone)]
pub struct BasisSwapValuation<T: FloatExt> {
  /// Present value of the pay leg.
  pub pay_leg_npv: T,
  /// Present value of the receive leg.
  pub receive_leg_npv: T,
  /// Net present value of the basis swap.
  pub net_npv: T,
  /// Absolute fair spread on the pay leg keeping the receive leg fixed.
  pub fair_spread_on_pay_leg: T,
  /// Absolute fair spread on the receive leg keeping the pay leg fixed.
  pub fair_spread_on_receive_leg: T,
  /// Absolute basis-point value of the pay-leg spread.
  pub pay_leg_bpv: T,
  /// Absolute basis-point value of the receive-leg spread.
  pub receive_leg_bpv: T,
}

/// Cross-currency basis-swap valuation summary in domestic currency terms.
#[derive(Debug, Clone)]
pub struct CrossCurrencyBasisSwapValuation<T: FloatExt> {
  /// Present value of the domestic leg in domestic currency.
  pub domestic_leg_npv: T,
  /// Present value of the foreign leg in foreign currency.
  pub foreign_leg_npv_foreign: T,
  /// Present value of the foreign leg converted to domestic currency.
  pub foreign_leg_npv_domestic: T,
  /// Net present value in domestic currency under the swap direction.
  pub net_npv: T,
  /// Absolute basis-point value of the domestic-leg spread.
  pub domestic_leg_bpv: T,
  /// Absolute basis-point value of the foreign-leg spread, converted to domestic currency.
  pub foreign_leg_bpv_domestic: T,
  /// Absolute fair spread on the domestic leg.
  pub fair_domestic_spread: T,
  /// Absolute fair spread on the foreign leg.
  pub fair_foreign_spread: T,
}
