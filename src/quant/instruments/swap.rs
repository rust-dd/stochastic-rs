//! Vanilla and basis interest-rate swap valuation.
//!
//! $$
//! \mathrm{PV}_{\mathrm{swap}}=
//! \mathrm{PV}_{\mathrm{receive}}-\mathrm{PV}_{\mathrm{pay}},\qquad
//! S^\star=\frac{\sum_i D(t_i)\,\alpha_i\,L_i\,N_i}{\sum_j D(T_j)\,\delta_j\,N_j}
//! $$
//!
//! Reference: Pallavicini & Tarenghi, "Interest-Rate Modeling with Multiple
//! Yield Curves", arXiv:1006.4767 (2010).
//!
//! Reference: Bianchetti & Carlicchi, "Interest Rates After The Credit Crunch:
//! Multiple-Curve Vanilla Derivatives and SABR", arXiv:1103.2567 (2011).
//!
//! Reference: Moreni & Pallavicini, "FX Modelling in Collateralized Markets:
//! foreign measures, basis curves, and pricing formulae", arXiv:1508.04321 (2015).

use std::fmt::Display;

use chrono::NaiveDate;

use super::super::calendar::DayCountConvention;
use super::super::calendar::Schedule;
use super::super::cashflows::Cashflow;
use super::super::cashflows::CashflowPricer;
use super::super::cashflows::CurveProvider;
use super::super::cashflows::FloatingIndex;
use super::super::cashflows::Leg;
use super::super::cashflows::NotionalSchedule;
use super::super::cashflows::OvernightIndex;
use super::super::fx::Currency;
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

/// Vanilla interest-rate swap backed by fixed and floating legs.
#[derive(Debug, Clone)]
pub struct VanillaInterestRateSwap<T: FloatExt> {
  /// Pay/receive direction.
  pub direction: SwapDirection,
  /// Contract notional.
  pub notional: T,
  /// Fixed coupon rate.
  pub fixed_rate: T,
  /// Fixed leg.
  pub fixed_leg: Leg<T>,
  /// Floating leg.
  pub floating_leg: Leg<T>,
}

impl<T: FloatExt> VanillaInterestRateSwap<T> {
  /// Build a vanilla fixed-versus-floating swap.
  pub fn new(
    direction: SwapDirection,
    fixed_schedule: &Schedule,
    float_schedule: &Schedule,
    notional: T,
    fixed_rate: T,
    fixed_day_count: DayCountConvention,
    float_index: FloatingIndex<T>,
    float_spread: T,
    float_day_count: DayCountConvention,
  ) -> Self {
    assert!(
      fixed_schedule.adjusted_dates.len() >= 2,
      "fixed schedule must contain at least two dates"
    );
    assert!(
      float_schedule.adjusted_dates.len() >= 2,
      "floating schedule must contain at least two dates"
    );

    let fixed_leg = Leg::fixed_rate(
      fixed_schedule,
      NotionalSchedule::bullet(fixed_schedule.adjusted_dates.len() - 1, notional),
      fixed_rate,
      fixed_day_count,
    );
    let floating_leg = Leg::floating_rate(
      float_schedule,
      NotionalSchedule::bullet(float_schedule.adjusted_dates.len() - 1, notional),
      float_index,
      float_spread,
      float_day_count,
    );

    Self {
      direction,
      notional,
      fixed_rate,
      fixed_leg,
      floating_leg,
    }
  }

  /// Build an overnight-indexed swap using the built-in overnight index type.
  pub fn overnight_indexed(
    direction: SwapDirection,
    fixed_schedule: &Schedule,
    float_schedule: &Schedule,
    notional: T,
    fixed_rate: T,
    fixed_day_count: DayCountConvention,
    overnight_index: OvernightIndex<T>,
    overnight_spread: T,
    float_day_count: DayCountConvention,
  ) -> Self {
    Self::new(
      direction,
      fixed_schedule,
      float_schedule,
      notional,
      fixed_rate,
      fixed_day_count,
      FloatingIndex::Overnight(overnight_index),
      overnight_spread,
      float_day_count,
    )
  }

  /// Swap valuation summary.
  pub fn valuation(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> SwapValuation<T> {
    let pricer = CashflowPricer::new(valuation_date, discount_day_count);
    let fixed_leg_npv = pricer.leg_npv(&self.fixed_leg, curves);
    let floating_leg_npv = pricer.leg_npv(&self.floating_leg, curves);
    let annuity =
      fixed_leg_bpv_annuity(&self.fixed_leg, valuation_date, discount_day_count, curves);
    let fair_rate = if annuity > T::zero() {
      floating_leg_npv / annuity
    } else {
      T::zero()
    };
    let bpv = annuity * T::from_f64_fast(1e-4);
    let dv01 = match self.direction {
      SwapDirection::Payer => -bpv,
      SwapDirection::Receiver => bpv,
    };
    let net_npv = match self.direction {
      SwapDirection::Payer => floating_leg_npv - fixed_leg_npv,
      SwapDirection::Receiver => fixed_leg_npv - floating_leg_npv,
    };

    SwapValuation {
      fixed_leg_npv,
      floating_leg_npv,
      net_npv,
      fair_rate,
      annuity,
      bpv,
      dv01,
    }
  }

  /// Net present value.
  pub fn npv(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    self
      .valuation(valuation_date, discount_day_count, curves)
      .net_npv
  }

  /// Fair fixed rate implied by the current curves.
  pub fn fair_rate(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    self
      .valuation(valuation_date, discount_day_count, curves)
      .fair_rate
  }

  /// Absolute basis-point value of the fixed rate.
  pub fn bpv(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    self
      .valuation(valuation_date, discount_day_count, curves)
      .bpv
  }

  /// Signed DV01 with respect to a 1 bp bump in the fixed rate.
  pub fn dv01(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    self
      .valuation(valuation_date, discount_day_count, curves)
      .dv01
  }
}

/// Same-currency floating-versus-floating basis swap.
#[derive(Debug, Clone)]
pub struct BasisSwap<T: FloatExt> {
  /// Pay-leg spread.
  pub pay_spread: T,
  /// Receive-leg spread.
  pub receive_spread: T,
  /// Pay leg.
  pub pay_leg: Leg<T>,
  /// Receive leg.
  pub receive_leg: Leg<T>,
}

impl<T: FloatExt> BasisSwap<T> {
  /// Build a basis swap from two floating legs.
  pub fn new(
    pay_schedule: &Schedule,
    receive_schedule: &Schedule,
    pay_notional: T,
    pay_index: FloatingIndex<T>,
    pay_spread: T,
    pay_day_count: DayCountConvention,
    receive_notional: T,
    receive_index: FloatingIndex<T>,
    receive_spread: T,
    receive_day_count: DayCountConvention,
  ) -> Self {
    assert!(
      pay_schedule.adjusted_dates.len() >= 2,
      "pay schedule must contain at least two dates"
    );
    assert!(
      receive_schedule.adjusted_dates.len() >= 2,
      "receive schedule must contain at least two dates"
    );

    let pay_leg = Leg::floating_rate(
      pay_schedule,
      NotionalSchedule::bullet(pay_schedule.adjusted_dates.len() - 1, pay_notional),
      pay_index,
      pay_spread,
      pay_day_count,
    );
    let receive_leg = Leg::floating_rate(
      receive_schedule,
      NotionalSchedule::bullet(receive_schedule.adjusted_dates.len() - 1, receive_notional),
      receive_index,
      receive_spread,
      receive_day_count,
    );

    Self {
      pay_spread,
      receive_spread,
      pay_leg,
      receive_leg,
    }
  }

  /// Basis-swap valuation summary.
  pub fn valuation(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> BasisSwapValuation<T> {
    let pricer = CashflowPricer::new(valuation_date, discount_day_count);
    let pay_leg_npv = pricer.leg_npv(&self.pay_leg, curves);
    let receive_leg_npv = pricer.leg_npv(&self.receive_leg, curves);
    let net_npv = receive_leg_npv - pay_leg_npv;
    let pay_leg_annuity =
      floating_leg_spread_annuity(&self.pay_leg, valuation_date, discount_day_count, curves);
    let receive_leg_annuity = floating_leg_spread_annuity(
      &self.receive_leg,
      valuation_date,
      discount_day_count,
      curves,
    );
    let fair_spread_on_pay_leg = if pay_leg_annuity > T::zero() {
      self.pay_spread + net_npv / pay_leg_annuity
    } else {
      self.pay_spread
    };
    let fair_spread_on_receive_leg = if receive_leg_annuity > T::zero() {
      self.receive_spread - net_npv / receive_leg_annuity
    } else {
      self.receive_spread
    };

    BasisSwapValuation {
      pay_leg_npv,
      receive_leg_npv,
      net_npv,
      fair_spread_on_pay_leg,
      fair_spread_on_receive_leg,
      pay_leg_bpv: pay_leg_annuity * T::from_f64_fast(1e-4),
      receive_leg_bpv: receive_leg_annuity * T::from_f64_fast(1e-4),
    }
  }

  /// Net present value.
  pub fn npv(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    self
      .valuation(valuation_date, discount_day_count, curves)
      .net_npv
  }
}

/// Constant-notional cross-currency basis swap with domestic reporting currency.
#[derive(Debug, Clone)]
pub struct CrossCurrencyBasisSwap<T: FloatExt> {
  /// Receive/pay orientation in domestic currency terms.
  pub direction: CrossCurrencySwapDirection,
  /// Domestic currency.
  pub domestic_currency: Currency,
  /// Foreign currency.
  pub foreign_currency: Currency,
  /// Spot FX rate expressed as domestic-currency units per one foreign-currency unit.
  pub fx_spot_domestic_per_foreign: T,
  /// Domestic-leg spread.
  pub domestic_spread: T,
  /// Foreign-leg spread.
  pub foreign_spread: T,
  /// Domestic floating leg.
  pub domestic_leg: Leg<T>,
  /// Foreign floating leg.
  pub foreign_leg: Leg<T>,
}

impl<T: FloatExt> CrossCurrencyBasisSwap<T> {
  /// Build a cross-currency basis swap. The initial notional exchange is assumed
  /// to have already taken place or to be neutral at current spot.
  pub fn new(
    direction: CrossCurrencySwapDirection,
    domestic_currency: Currency,
    foreign_currency: Currency,
    fx_spot_domestic_per_foreign: T,
    domestic_schedule: &Schedule,
    foreign_schedule: &Schedule,
    domestic_notional: T,
    domestic_index: FloatingIndex<T>,
    domestic_spread: T,
    domestic_day_count: DayCountConvention,
    foreign_notional: T,
    foreign_index: FloatingIndex<T>,
    foreign_spread: T,
    foreign_day_count: DayCountConvention,
    exchange_final_notionals: bool,
  ) -> Self {
    assert!(
      domestic_schedule.adjusted_dates.len() >= 2,
      "domestic schedule must contain at least two dates"
    );
    assert!(
      foreign_schedule.adjusted_dates.len() >= 2,
      "foreign schedule must contain at least two dates"
    );

    let domestic_maturity = *domestic_schedule.adjusted_dates.last().unwrap();
    let foreign_maturity = *foreign_schedule.adjusted_dates.last().unwrap();

    let mut domestic_leg = Leg::floating_rate(
      domestic_schedule,
      NotionalSchedule::bullet(
        domestic_schedule.adjusted_dates.len() - 1,
        domestic_notional,
      ),
      domestic_index,
      domestic_spread,
      domestic_day_count,
    );
    let mut foreign_leg = Leg::floating_rate(
      foreign_schedule,
      NotionalSchedule::bullet(foreign_schedule.adjusted_dates.len() - 1, foreign_notional),
      foreign_index,
      foreign_spread,
      foreign_day_count,
    );

    if exchange_final_notionals {
      domestic_leg = domestic_leg.with_redemption(domestic_maturity, domestic_notional);
      foreign_leg = foreign_leg.with_redemption(foreign_maturity, foreign_notional);
    }

    Self {
      direction,
      domestic_currency,
      foreign_currency,
      fx_spot_domestic_per_foreign,
      domestic_spread,
      foreign_spread,
      domestic_leg,
      foreign_leg,
    }
  }

  /// Cross-currency valuation summary in domestic currency terms.
  pub fn valuation(
    &self,
    valuation_date: NaiveDate,
    domestic_discount_day_count: DayCountConvention,
    domestic_curves: &(impl CurveProvider<T> + ?Sized),
    foreign_discount_day_count: DayCountConvention,
    foreign_curves: &(impl CurveProvider<T> + ?Sized),
  ) -> CrossCurrencyBasisSwapValuation<T> {
    let domestic_pricer = CashflowPricer::new(valuation_date, domestic_discount_day_count);
    let foreign_pricer = CashflowPricer::new(valuation_date, foreign_discount_day_count);
    let domestic_leg_npv = domestic_pricer.leg_npv(&self.domestic_leg, domestic_curves);
    let foreign_leg_npv_foreign = foreign_pricer.leg_npv(&self.foreign_leg, foreign_curves);
    let foreign_leg_npv_domestic = self.fx_spot_domestic_per_foreign * foreign_leg_npv_foreign;
    let domestic_leg_annuity = floating_leg_spread_annuity(
      &self.domestic_leg,
      valuation_date,
      domestic_discount_day_count,
      domestic_curves,
    );
    let foreign_leg_annuity_foreign = floating_leg_spread_annuity(
      &self.foreign_leg,
      valuation_date,
      foreign_discount_day_count,
      foreign_curves,
    );
    let foreign_leg_annuity_domestic =
      self.fx_spot_domestic_per_foreign * foreign_leg_annuity_foreign;
    let net_npv = match self.direction {
      CrossCurrencySwapDirection::PayDomesticReceiveForeign => {
        foreign_leg_npv_domestic - domestic_leg_npv
      }
      CrossCurrencySwapDirection::ReceiveDomesticPayForeign => {
        domestic_leg_npv - foreign_leg_npv_domestic
      }
    };

    let fair_domestic_spread = if domestic_leg_annuity > T::zero() {
      match self.direction {
        CrossCurrencySwapDirection::PayDomesticReceiveForeign => {
          self.domestic_spread + net_npv / domestic_leg_annuity
        }
        CrossCurrencySwapDirection::ReceiveDomesticPayForeign => {
          self.domestic_spread - net_npv / domestic_leg_annuity
        }
      }
    } else {
      self.domestic_spread
    };
    let fair_foreign_spread = if foreign_leg_annuity_domestic > T::zero() {
      match self.direction {
        CrossCurrencySwapDirection::PayDomesticReceiveForeign => {
          self.foreign_spread - net_npv / foreign_leg_annuity_domestic
        }
        CrossCurrencySwapDirection::ReceiveDomesticPayForeign => {
          self.foreign_spread + net_npv / foreign_leg_annuity_domestic
        }
      }
    } else {
      self.foreign_spread
    };

    CrossCurrencyBasisSwapValuation {
      domestic_leg_npv,
      foreign_leg_npv_foreign,
      foreign_leg_npv_domestic,
      net_npv,
      domestic_leg_bpv: domestic_leg_annuity * T::from_f64_fast(1e-4),
      foreign_leg_bpv_domestic: foreign_leg_annuity_domestic * T::from_f64_fast(1e-4),
      fair_domestic_spread,
      fair_foreign_spread,
    }
  }

  /// Net present value in domestic currency.
  pub fn npv(
    &self,
    valuation_date: NaiveDate,
    domestic_discount_day_count: DayCountConvention,
    domestic_curves: &(impl CurveProvider<T> + ?Sized),
    foreign_discount_day_count: DayCountConvention,
    foreign_curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    self
      .valuation(
        valuation_date,
        domestic_discount_day_count,
        domestic_curves,
        foreign_discount_day_count,
        foreign_curves,
      )
      .net_npv
  }
}

fn fixed_leg_bpv_annuity<T: FloatExt>(
  leg: &Leg<T>,
  valuation_date: NaiveDate,
  discount_day_count: DayCountConvention,
  curves: &(impl CurveProvider<T> + ?Sized),
) -> T {
  leg
    .cashflows()
    .iter()
    .map(|cashflow| match cashflow {
      Cashflow::Fixed(coupon) if coupon.period.payment_date >= valuation_date => {
        let tau = discount_day_count.year_fraction(valuation_date, coupon.period.payment_date);
        let df = curves.discount_curve().discount_factor(tau);
        df * coupon.notional * coupon.period.accrual_factor
      }
      _ => T::zero(),
    })
    .fold(T::zero(), |acc, value| acc + value)
}

fn floating_leg_spread_annuity<T: FloatExt>(
  leg: &Leg<T>,
  valuation_date: NaiveDate,
  discount_day_count: DayCountConvention,
  curves: &(impl CurveProvider<T> + ?Sized),
) -> T {
  leg
    .cashflows()
    .iter()
    .map(|cashflow| match cashflow {
      Cashflow::Floating(coupon) if coupon.period.payment_date >= valuation_date => {
        let tau = discount_day_count.year_fraction(valuation_date, coupon.period.payment_date);
        let df = curves.discount_curve().discount_factor(tau);
        df * coupon.notional * coupon.period.accrual_factor
      }
      _ => T::zero(),
    })
    .fold(T::zero(), |acc, value| acc + value)
}
