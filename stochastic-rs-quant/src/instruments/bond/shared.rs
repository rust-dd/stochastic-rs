use chrono::NaiveDate;
use roots::SimpleConvergency;
use roots::find_root_brent;

use super::types::BondAnalytics;
use super::types::BondPrice;
use crate::calendar::DayCountConvention;
use crate::cashflows::Cashflow;
use crate::cashflows::CashflowPricer;
use crate::cashflows::CurveProvider;
use crate::cashflows::Leg;
use crate::curves::Compounding;
use crate::traits::FloatExt;

pub(crate) fn bond_analytics_from_dirty_price<T: FloatExt>(
  leg: &Leg<T>,
  settlement_date: NaiveDate,
  dirty_price: T,
  accrued_interest: T,
  yield_day_count: DayCountConvention,
  compounding: Compounding,
) -> BondAnalytics<T> {
  let yield_to_maturity = yield_to_maturity_from_dirty_price_for_leg(
    leg,
    settlement_date,
    dirty_price,
    yield_day_count,
    compounding,
  );
  BondAnalytics {
    dirty_price,
    clean_price: dirty_price - accrued_interest,
    accrued_interest,
    yield_to_maturity,
    macaulay_duration: macaulay_duration_for_leg(
      leg,
      settlement_date,
      yield_to_maturity,
      yield_day_count,
      compounding,
    ),
    modified_duration: modified_duration_for_leg(
      leg,
      settlement_date,
      yield_to_maturity,
      yield_day_count,
      compounding,
    ),
    convexity: convexity_for_leg(
      leg,
      settlement_date,
      yield_to_maturity,
      yield_day_count,
      compounding,
    ),
  }
}

pub(crate) fn price_deterministic_leg_from_curve<T: FloatExt>(
  leg: &Leg<T>,
  valuation_date: NaiveDate,
  discount_day_count: DayCountConvention,
  curves: &(impl CurveProvider<T> + ?Sized),
) -> BondPrice<T> {
  let summary = CashflowPricer::new(valuation_date, discount_day_count).summarize_leg(leg, curves);
  BondPrice {
    dirty_price: summary.dirty_npv,
    accrued_interest: summary.accrued_interest,
    clean_price: summary.clean_npv,
  }
}

pub(crate) fn accrued_interest_for_deterministic_leg<T: FloatExt>(
  leg: &Leg<T>,
  settlement_date: NaiveDate,
) -> T {
  leg
    .cashflows()
    .iter()
    .map(|cashflow| match cashflow {
      Cashflow::Fixed(coupon) => coupon.accrued_interest(settlement_date),
      Cashflow::Simple(_) => T::zero(),
      Cashflow::Floating(_) | Cashflow::Cms(_) => {
        unreachable!("deterministic bond legs must not contain stochastic coupons")
      }
    })
    .fold(T::zero(), |acc, value| acc + value)
}

pub(crate) fn dirty_price_from_yield_for_leg<T: FloatExt>(
  leg: &Leg<T>,
  settlement_date: NaiveDate,
  yield_to_maturity: T,
  yield_day_count: DayCountConvention,
  compounding: Compounding,
) -> T {
  leg
    .cashflows()
    .iter()
    .filter(|cashflow| cashflow.payment_date() >= settlement_date)
    .map(|cashflow| {
      let tau = yield_day_count.year_fraction(settlement_date, cashflow.payment_date());
      let amount = deterministic_cashflow_amount(cashflow);
      amount * compounding.discount_factor(yield_to_maturity, tau)
    })
    .fold(T::zero(), |acc, value| acc + value)
}

pub(crate) fn yield_to_maturity_from_dirty_price_for_leg<T: FloatExt>(
  leg: &Leg<T>,
  settlement_date: NaiveDate,
  dirty_price: T,
  yield_day_count: DayCountConvention,
  compounding: Compounding,
) -> T {
  if dirty_price <= T::zero() {
    return T::zero();
  }

  let target = dirty_price.to_f64().unwrap();
  let min_yield = min_yield_for_leg(leg, settlement_date, yield_day_count, compounding);
  let mut low = min_yield + 1e-8;
  let mut high = 0.10f64;
  let f = |y: f64| {
    dirty_price_from_yield_for_leg(
      leg,
      settlement_date,
      T::from_f64_fast(y),
      yield_day_count,
      compounding,
    )
    .to_f64()
    .unwrap()
      - target
  };

  let mut f_low = f(low);
  let mut expand_low = 0;
  while f_low < 0.0 && matches!(compounding, Compounding::Continuous) && expand_low < 16 {
    low *= 2.0;
    f_low = f(low);
    expand_low += 1;
  }

  let mut f_high = f(high);
  let mut expand_high = 0;
  while f_high > 0.0 && expand_high < 32 {
    high = high * 2.0 + 0.05;
    f_high = f(high);
    expand_high += 1;
  }

  if f_low.abs() < 1e-12 {
    return T::from_f64_fast(low);
  }
  if f_high.abs() < 1e-12 {
    return T::from_f64_fast(high);
  }
  if f_low * f_high > 0.0 {
    panic!(
      "failed to bracket yield-to-maturity for dirty price {} between {low} and {high}",
      dirty_price.to_f64().unwrap()
    );
  }

  let mut convergency = SimpleConvergency {
    eps: 1e-12,
    max_iter: 100,
  };
  let root = find_root_brent(low, high, f, &mut convergency)
    .expect("failed to solve yield-to-maturity with Brent root finder");
  T::from_f64_fast(root)
}

pub(crate) fn macaulay_duration_for_leg<T: FloatExt>(
  leg: &Leg<T>,
  settlement_date: NaiveDate,
  yield_to_maturity: T,
  yield_day_count: DayCountConvention,
  compounding: Compounding,
) -> T {
  let dirty_price = dirty_price_from_yield_for_leg(
    leg,
    settlement_date,
    yield_to_maturity,
    yield_day_count,
    compounding,
  );
  if dirty_price <= T::zero() {
    return T::zero();
  }

  let weighted = leg
    .cashflows()
    .iter()
    .filter(|cashflow| cashflow.payment_date() >= settlement_date)
    .map(|cashflow| {
      let tau = yield_day_count.year_fraction(settlement_date, cashflow.payment_date());
      let amount = deterministic_cashflow_amount(cashflow);
      let pv = amount * compounding.discount_factor(yield_to_maturity, tau);
      tau * pv
    })
    .fold(T::zero(), |acc, value| acc + value);

  weighted / dirty_price
}

pub(crate) fn modified_duration_for_leg<T: FloatExt>(
  leg: &Leg<T>,
  settlement_date: NaiveDate,
  yield_to_maturity: T,
  yield_day_count: DayCountConvention,
  compounding: Compounding,
) -> T {
  let price = dirty_price_from_yield_for_leg(
    leg,
    settlement_date,
    yield_to_maturity,
    yield_day_count,
    compounding,
  );
  if price <= T::zero() {
    return T::zero();
  }

  let y = yield_to_maturity.to_f64().unwrap();
  let h = (1e-5f64).max(1e-5 * (1.0 + y.abs()));
  let down =
    (y - h).max(min_yield_for_leg(leg, settlement_date, yield_day_count, compounding) + 1e-8);
  let up = y + h;
  let p_down = dirty_price_from_yield_for_leg(
    leg,
    settlement_date,
    T::from_f64_fast(down),
    yield_day_count,
    compounding,
  )
  .to_f64()
  .unwrap();
  let p_up = dirty_price_from_yield_for_leg(
    leg,
    settlement_date,
    T::from_f64_fast(up),
    yield_day_count,
    compounding,
  )
  .to_f64()
  .unwrap();
  let p = price.to_f64().unwrap();
  T::from_f64_fast(-(p_up - p_down) / ((up - down) * p))
}

pub(crate) fn convexity_for_leg<T: FloatExt>(
  leg: &Leg<T>,
  settlement_date: NaiveDate,
  yield_to_maturity: T,
  yield_day_count: DayCountConvention,
  compounding: Compounding,
) -> T {
  let price = dirty_price_from_yield_for_leg(
    leg,
    settlement_date,
    yield_to_maturity,
    yield_day_count,
    compounding,
  );
  if price <= T::zero() {
    return T::zero();
  }

  let y = yield_to_maturity.to_f64().unwrap();
  let h = (1e-4f64).max(1e-4 * (1.0 + y.abs()));
  let down =
    (y - h).max(min_yield_for_leg(leg, settlement_date, yield_day_count, compounding) + 1e-8);
  let up = y + h;
  let mid = dirty_price_from_yield_for_leg(
    leg,
    settlement_date,
    T::from_f64_fast(y),
    yield_day_count,
    compounding,
  )
  .to_f64()
  .unwrap();
  let p_down = dirty_price_from_yield_for_leg(
    leg,
    settlement_date,
    T::from_f64_fast(down),
    yield_day_count,
    compounding,
  )
  .to_f64()
  .unwrap();
  let p_up = dirty_price_from_yield_for_leg(
    leg,
    settlement_date,
    T::from_f64_fast(up),
    yield_day_count,
    compounding,
  )
  .to_f64()
  .unwrap();
  let p = price.to_f64().unwrap();
  let h_eff = (up - down) / 2.0;
  T::from_f64_fast((p_up - 2.0 * mid + p_down) / (h_eff * h_eff * p))
}

pub(crate) fn price_with_constant_spread_for_leg<T: FloatExt>(
  leg: &Leg<T>,
  valuation_date: NaiveDate,
  discount_day_count: DayCountConvention,
  curves: &(impl CurveProvider<T> + ?Sized),
  spread: T,
) -> T {
  leg
    .cashflows()
    .iter()
    .map(|cashflow| {
      let payment_date = cashflow.payment_date();
      if payment_date < valuation_date {
        return T::zero();
      }
      let tau = discount_day_count.year_fraction(valuation_date, payment_date);
      let df = curves.discount_curve().discount_factor(tau) * (-spread * tau).exp();
      df * deterministic_cashflow_amount(cashflow)
    })
    .fold(T::zero(), |acc, value| acc + value)
}

pub(crate) fn solve_constant_spread_for_leg<T: FloatExt>(
  leg: &Leg<T>,
  valuation_date: NaiveDate,
  market_dirty_price: T,
  discount_day_count: DayCountConvention,
  curves: &(impl CurveProvider<T> + ?Sized),
  option_cost: T,
) -> T {
  if market_dirty_price <= T::zero() {
    return T::zero();
  }

  let target = market_dirty_price.to_f64().unwrap();
  let f = |spread: f64| {
    let dirty = price_with_constant_spread_for_leg(
      leg,
      valuation_date,
      discount_day_count,
      curves,
      T::from_f64_fast(spread),
    )
    .to_f64()
    .unwrap();
    dirty - option_cost.to_f64().unwrap() - target
  };

  if f(0.0).abs() < 1e-12 {
    return T::zero();
  }

  let mut low = -0.05f64;
  let mut high = 0.05f64;
  let mut f_low = f(low);
  let mut f_high = f(high);
  let mut expand = 0;
  while f_low * f_high > 0.0 && expand < 40 {
    if f_low < 0.0 {
      low = low * 2.0 - 0.01;
      f_low = f(low);
    } else {
      high = high * 2.0 + 0.01;
      f_high = f(high);
    }
    expand += 1;
  }

  if f_low * f_high > 0.0 {
    panic!(
      "failed to bracket constant spread for dirty price {} between {low} and {high}",
      market_dirty_price.to_f64().unwrap()
    );
  }

  let mut convergency = SimpleConvergency {
    eps: 1e-12,
    max_iter: 100,
  };
  let root =
    find_root_brent(low, high, f, &mut convergency).expect("failed to solve constant spread");
  T::from_f64_fast(root)
}

pub(crate) fn fixed_leg_spread_annuity<T: FloatExt>(
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

fn min_yield_for_leg<T: FloatExt>(
  leg: &Leg<T>,
  settlement_date: NaiveDate,
  yield_day_count: DayCountConvention,
  compounding: Compounding,
) -> f64 {
  match compounding {
    Compounding::Continuous => -1.0,
    Compounding::Periodic(periods_per_year) => -(periods_per_year as f64) + 1e-6,
    Compounding::Simple => {
      let max_tau = leg
        .cashflows()
        .iter()
        .filter(|cashflow| cashflow.payment_date() >= settlement_date)
        .map(|cashflow| {
          yield_day_count
            .year_fraction::<T>(settlement_date, cashflow.payment_date())
            .to_f64()
            .unwrap()
        })
        .fold(0.0f64, f64::max);
      if max_tau <= 0.0 {
        -1.0
      } else {
        -1.0 / max_tau + 1e-6
      }
    }
  }
}

fn deterministic_cashflow_amount<T: FloatExt>(cashflow: &Cashflow<T>) -> T {
  match cashflow {
    Cashflow::Fixed(coupon) => coupon.amount(),
    Cashflow::Simple(cashflow) => cashflow.amount,
    Cashflow::Floating(_) | Cashflow::Cms(_) => {
      unreachable!("deterministic bond legs must not contain stochastic coupons")
    }
  }
}
