use chrono::NaiveDate;

use super::shared::floating_leg_spread_annuity;
use super::types::CrossCurrencyBasisSwapValuation;
use super::types::CrossCurrencySwapDirection;
use crate::calendar::DayCountConvention;
use crate::calendar::Schedule;
use crate::cashflows::CashflowPricer;
use crate::cashflows::CurveProvider;
use crate::cashflows::FloatingIndex;
use crate::cashflows::Leg;
use crate::cashflows::NotionalSchedule;
use crate::fx::Currency;
use crate::traits::RealExt;

/// One floating leg of a cross-currency swap, in its own currency.
#[derive(Debug, Clone)]
pub struct CurrencyLeg<T: RealExt> {
  /// Currency the leg pays in.
  pub currency: Currency,
  /// Constant notional in `currency`.
  pub notional: T,
  /// Floating index the coupons fix on.
  pub index: FloatingIndex<T>,
  /// Spread over the index.
  pub spread: T,
  /// Accrual day count.
  pub day_count: DayCountConvention,
  /// Accrual and payment schedule.
  pub schedule: Schedule,
}

/// Constant-notional cross-currency basis swap with domestic reporting currency.
#[derive(Debug, Clone)]
pub struct CrossCurrencyBasisSwap<T: RealExt> {
  /// Receive/pay orientation in domestic currency terms.
  pub direction: CrossCurrencySwapDirection,
  /// Currency the swap's value is reported in.
  pub domestic_currency: Currency,
  /// Currency of the foreign leg, converted at `fx_spot_domestic_per_foreign`.
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

impl<T: RealExt> CrossCurrencyBasisSwap<T> {
  /// A swap whose initial notional exchange is settled or neutral at spot; panics unless both
  /// schedules hold at least two dates.
  pub fn new(
    direction: CrossCurrencySwapDirection,
    fx_spot_domestic_per_foreign: T,
    domestic: CurrencyLeg<T>,
    foreign: CurrencyLeg<T>,
    exchange_final_notionals: bool,
  ) -> Self {
    let dates = domestic.schedule.adjusted_dates.len();
    assert!(
      dates >= 2,
      "domestic.schedule must satisfy `adjusted_dates.len() >= 2`, got adjusted_dates.len() = {dates}"
    );
    let dates = foreign.schedule.adjusted_dates.len();
    assert!(
      dates >= 2,
      "foreign.schedule must satisfy `adjusted_dates.len() >= 2`, got adjusted_dates.len() = {dates}"
    );
    Self {
      direction,
      domestic_currency: domestic.currency,
      foreign_currency: foreign.currency,
      fx_spot_domestic_per_foreign,
      domestic_spread: domestic.spread,
      foreign_spread: foreign.spread,
      domestic_leg: floating_leg(domestic, exchange_final_notionals),
      foreign_leg: floating_leg(foreign, exchange_final_notionals),
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

fn floating_leg<T: RealExt>(leg: CurrencyLeg<T>, exchange_final_notional: bool) -> Leg<T> {
  let periods = leg.schedule.adjusted_dates.len() - 1;
  let floating = Leg::floating_rate(
    &leg.schedule,
    NotionalSchedule::bullet(periods, leg.notional),
    leg.index,
    leg.spread,
    leg.day_count,
  );
  match leg.schedule.adjusted_dates.last() {
    Some(&maturity) if exchange_final_notional => floating.with_redemption(maturity, leg.notional),
    _ => floating,
  }
}
