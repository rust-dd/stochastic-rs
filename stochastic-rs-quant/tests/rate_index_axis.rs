//! A rate index reads its curve on the pricer's time axis and accrues on its own day count: an
//! ACT/360 leg on one curve telescopes on an ACT/365F axis, as QuantLib's `forecastFixing` reads.

use chrono::NaiveDate;
use ndarray::array;
use stochastic_rs_quant::calendar::DayCountConvention;
use stochastic_rs_quant::calendar::Frequency;
use stochastic_rs_quant::calendar::ScheduleBuilder;
use stochastic_rs_quant::cashflows::CashflowPricer;
use stochastic_rs_quant::cashflows::CmsIndex;
use stochastic_rs_quant::cashflows::FloatingIndex;
use stochastic_rs_quant::cashflows::IborIndex;
use stochastic_rs_quant::cashflows::Leg;
use stochastic_rs_quant::cashflows::NotionalSchedule;
use stochastic_rs_quant::cashflows::OvernightIndex;
use stochastic_rs_quant::cashflows::RateTenor;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;

const ACT360: DayCountConvention = DayCountConvention::Actual360;
const AXIS: DayCountConvention = DayCountConvention::Actual365Fixed;

fn d(year: i32, month: u32, day: u32) -> NaiveDate {
  NaiveDate::from_ymd_opt(year, month, day).unwrap()
}

fn curve() -> DiscountCurve<f64> {
  DiscountCurve::from_zero_rates(
    array![0.5, 2.0, 5.0, 10.0].view(),
    array![0.02, 0.025, 0.03, 0.033].view(),
    InterpolationMethod::LogLinearOnDiscountFactors,
  )
}

/// The 2y×5y floating leg paid at its accrual ends is worth `D(start) − D(end)` on one curve.
#[test]
fn an_act360_floating_leg_telescopes_on_the_pricers_axis() {
  let (valuation, curve) = (d(2025, 1, 2), curve());
  let schedule = ScheduleBuilder::new(d(2027, 1, 5), d(2032, 1, 5))
    .frequency(Frequency::Quarterly)
    .build();
  let periods = schedule.adjusted_dates.len() - 1;
  let read = |date: NaiveDate| curve.discount_factor(AXIS.year_fraction(valuation, date));
  let telescoped = read(schedule.adjusted_dates[0]) - read(schedule.adjusted_dates[periods]);
  for index in [
    FloatingIndex::Ibor(IborIndex::new("EUR-3M", RateTenor::ThreeMonths, ACT360)),
    FloatingIndex::Overnight(OvernightIndex::new("ESTR", ACT360)),
  ] {
    let name = index.to_string();
    let leg = Leg::floating_rate(
      &schedule,
      NotionalSchedule::bullet(periods, 1.0),
      index,
      0.0,
      ACT360,
    );
    let pv = CashflowPricer::new(valuation, AXIS).leg_npv(&leg, &curve);
    assert!(
      (pv - telescoped).abs() < 1e-14,
      "{name}: {pv} vs {telescoped}"
    );
  }
}

/// A CMS coupon fixes the forward swap rate with its annuity read on the axis and accrued ACT/360.
#[test]
fn an_act360_cms_coupon_fixes_the_swap_rate_on_the_pricers_axis() {
  let (valuation, curve) = (d(2025, 1, 2), curve());
  let schedule = ScheduleBuilder::new(d(2026, 1, 5), d(2026, 7, 5))
    .frequency(Frequency::SemiAnnual)
    .build();
  let index = CmsIndex::new(
    "EUR-CMS-2Y",
    24,
    Frequency::SemiAnnual,
    ACT360,
    RateTenor::SixMonths,
  );
  let leg = Leg::cms(
    &schedule,
    NotionalSchedule::bullet(1, 1.0),
    index,
    0.0,
    ACT360,
  );
  let pv = CashflowPricer::new(valuation, AXIS).leg_npv(&leg, &curve);

  let read = |date: NaiveDate| curve.discount_factor(AXIS.year_fraction(valuation, date));
  let swap_dates = [
    d(2026, 1, 5),
    d(2026, 7, 5),
    d(2027, 1, 5),
    d(2027, 7, 5),
    d(2028, 1, 5),
  ];
  let annuity: f64 = swap_dates
    .windows(2)
    .map(|w| ACT360.year_fraction::<f64>(w[0], w[1]) * read(w[1]))
    .sum();
  let swap_rate = (read(swap_dates[0]) - read(swap_dates[4])) / annuity;
  let (start, end) = (schedule.adjusted_dates[0], schedule.adjusted_dates[1]);
  let expected = read(end) * ACT360.year_fraction::<f64>(start, end) * swap_rate;
  assert!((pv - expected).abs() < 1e-15, "{pv} vs {expected}");
}
