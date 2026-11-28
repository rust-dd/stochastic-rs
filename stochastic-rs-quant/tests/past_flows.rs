//! A flow wholly in the past is worth zero and reads no curve: a finished coupon accrues nothing,
//! a settled FRA is worth nothing, and a coupon whose fixing is past prices from that fixing.

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
use stochastic_rs_quant::cashflows::RateTenor;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::instruments::FloatingRateBond;
use stochastic_rs_quant::instruments::ForwardRateAgreement;
use stochastic_rs_quant::instruments::FraPosition;
use stochastic_rs_quant::market::FixingHistory;

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

/// Valued on a roll date, one coupon is paid that day and the next starts: neither accrues, so the
/// accrued interest is zero without a fixing, though the coupon paid today needs one to be valued.
#[test]
fn a_leg_valued_on_a_roll_date_accrues_nothing() {
  let schedule = ScheduleBuilder::new(d(2024, 1, 5), d(2027, 1, 5))
    .frequency(Frequency::Quarterly)
    .build();
  let periods = schedule.adjusted_dates.len() - 1;
  let notionals = || NotionalSchedule::bullet(periods, 1.0);
  let ibor = FloatingIndex::Ibor(IborIndex::new("EUR-3M", RateTenor::ThreeMonths, ACT360));
  let cms = CmsIndex::new(
    "EUR-CMS-2Y",
    24,
    Frequency::Annual,
    ACT360,
    RateTenor::ThreeMonths,
  );
  let legs = [
    (
      "floating",
      Leg::floating_rate(&schedule, notionals(), ibor, 0.0, ACT360),
    ),
    ("cms", Leg::cms(&schedule, notionals(), cms, 0.0, ACT360)),
  ];
  let (pricer, curve) = (CashflowPricer::new(d(2025, 1, 5), AXIS), curve());
  for (name, leg) in &legs {
    assert_eq!(pricer.leg_accrued_interest(leg, &curve), 0.0, "{name}");
  }
}

/// A FRA settled at its start and one paid at its end, each valued after its payment.
#[test]
fn a_settled_fra_is_worth_nothing() {
  let index = IborIndex::new("EUR-6M", RateTenor::SixMonths, ACT360);
  let (start, end) = (d(2025, 1, 6), d(2025, 7, 6));
  let curve = curve();
  let at_start = ForwardRateAgreement::with_fra_discounting(
    FraPosition::Long,
    1.0,
    0.03,
    start,
    end,
    ACT360,
    index.clone(),
  );
  assert_eq!(at_start.valuation(d(2025, 2, 6), AXIS, &curve).npv, 0.0);
  let at_end = ForwardRateAgreement::new(FraPosition::Long, 1.0, 0.03, start, end, ACT360, index);
  assert_eq!(at_end.valuation(d(2025, 8, 6), AXIS, &curve).npv, 0.0);
}

/// A note a month into its first coupon is NaN until that coupon has its fixing; with it the
/// accrued interest is the fixing over the month, and the later coupons project off the curve.
#[test]
fn a_seasoned_note_prices_once_its_running_coupon_has_its_fixing() {
  let schedule = ScheduleBuilder::new(d(2024, 1, 5), d(2027, 1, 5))
    .frequency(Frequency::Quarterly)
    .build();
  let index = FloatingIndex::Ibor(IborIndex::new("EUR-3M", RateTenor::ThreeMonths, ACT360));
  let note = FloatingRateBond::new(&schedule, 100.0, index, 0.0, ACT360);
  let (valuation, curve) = (d(2024, 2, 5), curve());
  assert!(
    note
      .price_from_curve(valuation, AXIS, &curve)
      .dirty_price
      .is_nan()
  );
  let fixings = FixingHistory::new();
  fixings.add_fixing(d(2024, 1, 5), 0.039);
  let price = note
    .with_fixings(&fixings)
    .price_from_curve(valuation, AXIS, &curve);
  let accrued = 100.0 * 0.039 * ACT360.year_fraction::<f64>(d(2024, 1, 5), valuation);
  assert!(
    (price.accrued_interest - accrued).abs() < 1e-12,
    "{price:?}"
  );
  assert!(price.dirty_price.is_finite(), "{price:?}");
  assert_eq!(
    price.clean_price,
    price.dirty_price - price.accrued_interest
  );
}

/// On the roll date the coupon paid that day prices from its fixing at its full accrual.
#[test]
fn a_coupon_paid_on_the_valuation_date_prices_from_its_fixing() {
  let schedule = ScheduleBuilder::new(d(2024, 1, 5), d(2027, 1, 5))
    .frequency(Frequency::Quarterly)
    .build();
  let periods = schedule.adjusted_dates.len() - 1;
  let ibor = FloatingIndex::Ibor(IborIndex::new("EUR-3M", RateTenor::ThreeMonths, ACT360));
  let leg = Leg::floating_rate(
    &schedule,
    NotionalSchedule::bullet(periods, 1.0),
    ibor,
    0.0,
    ACT360,
  );
  let (pricer, curve) = (CashflowPricer::new(d(2025, 1, 5), AXIS), curve());
  assert!(pricer.leg_npv(&leg, &curve).is_nan());
  let fixings = FixingHistory::new();
  fixings.add_fixing(d(2024, 10, 5), 0.037);
  let fixed = leg.with_fixings(&fixings);
  let paid_today = fixed
    .cashflows()
    .iter()
    .find(|cashflow| cashflow.payment_date() == d(2025, 1, 5))
    .unwrap();
  let full = 0.037 * ACT360.year_fraction::<f64>(d(2024, 10, 5), d(2025, 1, 5));
  assert_eq!(pricer.cashflow_npv(paid_today, &curve), full);
  assert!(pricer.leg_npv(&fixed, &curve).is_finite());
}
