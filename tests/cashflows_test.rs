//! Comparison tests for the cashflow engine.
//!
//! Fixed-leg PV and CMS forwards are verified against direct discounted-cashflow
//! formulas. Floating coupon projection is verified against the multicurve
//! forward-rate identity.

use chrono::NaiveDate;
use ndarray::array;
use stochastic_rs::quant::calendar::DayCountConvention;
use stochastic_rs::quant::calendar::Frequency;
use stochastic_rs::quant::calendar::ScheduleBuilder;
use stochastic_rs::quant::cashflows::CashflowPricer;
use stochastic_rs::quant::cashflows::CmsIndex;
use stochastic_rs::quant::cashflows::FloatingIndex;
use stochastic_rs::quant::cashflows::IborIndex;
use stochastic_rs::quant::cashflows::Leg;
use stochastic_rs::quant::cashflows::NotionalSchedule;
use stochastic_rs::quant::cashflows::RateTenor;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::InterpolationMethod;
use stochastic_rs::quant::curves::MultiCurve;

const TOL: f64 = 1e-10;

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
  NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

fn flat_curve(rate: f64, max_t: f64) -> DiscountCurve<f64> {
  let times = array![0.0, max_t / 4.0, max_t / 2.0, 3.0 * max_t / 4.0, max_t];
  let rates = array![rate, rate, rate, rate, rate];
  DiscountCurve::from_zero_rates(&times, &rates, InterpolationMethod::LinearOnZeroRates)
}

fn approx(a: f64, b: f64, tol: f64) {
  assert!(
    (a - b).abs() < tol,
    "expected {b:.12}, got {a:.12}, diff {:.2e}",
    (a - b).abs()
  );
}

#[test]
fn fixed_rate_leg_npv_matches_manual_discounting() {
  let valuation = d(2024, 1, 15);
  let schedule = ScheduleBuilder::new(valuation, d(2025, 1, 15))
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let leg = Leg::fixed_rate(
    &schedule,
    NotionalSchedule::bullet(4, 100.0),
    0.05,
    DayCountConvention::Actual360,
  )
  .with_redemption(d(2025, 1, 15), 100.0);

  let curve = flat_curve(0.04, 1.1);
  let pricer = CashflowPricer::new(valuation, DayCountConvention::Actual365Fixed);
  let summary = pricer.summarize_leg(&leg, &curve);

  let mut manual = 0.0;
  for window in schedule.adjusted_dates.windows(2) {
    let delta: f64 = DayCountConvention::Actual360.year_fraction(window[0], window[1]);
    let tau: f64 = DayCountConvention::Actual365Fixed.year_fraction(valuation, window[1]);
    let amount = 100.0 * 0.05 * delta;
    manual += curve.discount_factor(tau) * amount;
  }
  let maturity_tau: f64 =
    DayCountConvention::Actual365Fixed.year_fraction(valuation, d(2025, 1, 15));
  manual += curve.discount_factor(maturity_tau) * 100.0;

  approx(summary.dirty_npv, manual, 1e-10);
  approx(summary.accrued_interest, 0.0, TOL);
  approx(summary.clean_npv, manual, 1e-10);
}

#[test]
fn floating_coupon_uses_multicurve_forecast_curve() {
  let valuation = d(2024, 1, 15);
  let maturity = d(2024, 4, 15);
  let schedule = ScheduleBuilder::new(valuation, maturity)
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let leg = Leg::floating_rate(
    &schedule,
    NotionalSchedule::bullet(1, 250.0),
    FloatingIndex::Ibor(IborIndex::new(
      "USD-LIBOR-3M",
      RateTenor::ThreeMonths,
      DayCountConvention::Actual365Fixed,
    )),
    0.0010,
    DayCountConvention::Actual365Fixed,
  );

  let discount_curve = flat_curve(0.02, 0.5);
  let forecast_curve = flat_curve(0.04, 0.5);
  let mut curves = MultiCurve::new(discount_curve);
  curves.add_forecast("3M", forecast_curve);

  let coupon = &leg.cashflows()[0];
  let delta: f64 = DayCountConvention::Actual365Fixed.year_fraction(valuation, maturity);
  let expected_forward = ((0.04 * delta).exp() - 1.0) / delta;
  let expected_amount = 250.0 * delta * (expected_forward + 0.0010);

  approx(coupon.amount(&curves, valuation), expected_amount, 1e-10);
}

#[test]
fn cms_coupon_matches_forward_swap_rate_under_flat_curve() {
  let valuation = d(2024, 1, 15);
  let schedule = ScheduleBuilder::new(valuation, d(2024, 7, 15))
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let leg = Leg::cms(
    &schedule,
    NotionalSchedule::bullet(1, 100.0),
    CmsIndex::new(
      "USD-CMS-2Y",
      24,
      Frequency::SemiAnnual,
      DayCountConvention::Actual365Fixed,
      "CMS",
    ),
    0.0,
    DayCountConvention::Actual365Fixed,
  );

  let curve = flat_curve(0.04, 2.5);
  let coupon = &leg.cashflows()[0];
  let swap_dates = [
    d(2024, 1, 15),
    d(2024, 7, 15),
    d(2025, 1, 15),
    d(2025, 7, 15),
    d(2026, 1, 15),
  ];
  let mut annuity = 0.0;
  for window in swap_dates.windows(2) {
    let delta: f64 = DayCountConvention::Actual365Fixed.year_fraction(window[0], window[1]);
    let tau: f64 = DayCountConvention::Actual365Fixed.year_fraction(valuation, window[1]);
    annuity += delta * curve.discount_factor(tau);
  }
  let end_tau: f64 = DayCountConvention::Actual365Fixed.year_fraction(valuation, d(2026, 1, 15));
  let swap_rate = (1.0 - curve.discount_factor(end_tau)) / annuity;
  let coupon_delta: f64 =
    DayCountConvention::Actual365Fixed.year_fraction(valuation, d(2024, 7, 15));
  let expected_amount = 100.0 * coupon_delta * swap_rate;

  approx(coupon.amount(&curve, valuation), expected_amount, 1e-10);
}

#[test]
fn accrued_interest_matches_partial_fixed_period() {
  let start = d(2024, 1, 15);
  let as_of = d(2024, 4, 15);
  let end = d(2024, 7, 15);
  let schedule = ScheduleBuilder::new(start, end)
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let leg = Leg::fixed_rate(
    &schedule,
    NotionalSchedule::bullet(1, 500.0),
    0.06,
    DayCountConvention::Actual360,
  );

  let elapsed: f64 = DayCountConvention::Actual360.year_fraction(start, as_of);
  let expected = 500.0 * 0.06 * elapsed;
  let accrued = leg.cashflows()[0].accrued_interest(&flat_curve(0.03, 1.0), as_of, as_of);
  approx(accrued, expected, 1e-10);
}
