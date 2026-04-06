use chrono::NaiveDate;

use super::shared::floating_leg_spread_annuity;
use super::types::CrossCurrencyBasisSwapValuation;
use super::types::CrossCurrencySwapDirection;
use crate::quant::calendar::DayCountConvention;
use crate::quant::calendar::Schedule;
use crate::quant::cashflows::CashflowPricer;
use crate::quant::cashflows::CurveProvider;
use crate::quant::cashflows::FloatingIndex;
use crate::quant::cashflows::Leg;
use crate::quant::cashflows::NotionalSchedule;
use crate::quant::fx::Currency;
use crate::traits::FloatExt;

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
