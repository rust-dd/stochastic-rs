//! Cap, floor, and collar comparison tests.
//!
//! Verifies that cap NPVs equal the sum of constituent caplet prices, and
//! that cap minus floor equals the intrinsic value of the underlying forward
//! swap (the Black-76 put-call parity in caplet form).

use chrono::NaiveDate;
use ndarray::array;
use stochastic_rs::quant::calendar::DayCountConvention;
use stochastic_rs::quant::calendar::Frequency;
use stochastic_rs::quant::calendar::ScheduleBuilder;
use stochastic_rs::quant::cashflows::FloatingIndex;
use stochastic_rs::quant::cashflows::IborIndex;
use stochastic_rs::quant::cashflows::Leg;
use stochastic_rs::quant::cashflows::NotionalSchedule;
use stochastic_rs::quant::cashflows::RateTenor;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::InterpolationMethod;
use stochastic_rs::quant::instruments::BlackVolatility;
use stochastic_rs::quant::instruments::Cap;
use stochastic_rs::quant::instruments::Collar;
use stochastic_rs::quant::instruments::Floor;

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

fn floating_leg(
  valuation: NaiveDate,
  end: NaiveDate,
  notional: f64,
  day_count: DayCountConvention,
) -> Leg<f64> {
  let schedule = ScheduleBuilder::new(valuation, end)
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let periods = schedule.adjusted_dates.len() - 1;
  let index = FloatingIndex::Ibor(IborIndex::new(
    "LIBOR_3M",
    RateTenor::ThreeMonths,
    day_count,
  ));
  Leg::floating_rate(
    &schedule,
    NotionalSchedule::bullet(periods, notional),
    index,
    0.0,
    day_count,
  )
}

#[test]
fn cap_equals_sum_of_caplet_prices() {
  let valuation = d(2024, 1, 15);
  let leg = floating_leg(
    valuation,
    d(2026, 1, 15),
    1_000_000.0,
    DayCountConvention::Actual360,
  );
  let curve = flat_curve(0.04, 2.2);
  let cap = Cap::new(0.045, leg, BlackVolatility::new(0.25));
  let val = cap.valuation(
    valuation,
    DayCountConvention::Actual365Fixed,
    DayCountConvention::Actual365Fixed,
    &curve,
  );
  let sum: f64 = val.caplet_prices.iter().sum();
  approx(val.npv, sum, TOL);
  assert!(val.npv > 0.0, "cap must have positive value");
}

#[test]
fn cap_minus_floor_equals_forward_swap_intrinsic() {
  use stochastic_rs::quant::cashflows::Cashflow;

  let valuation = d(2024, 1, 15);
  let notional = 1_000_000.0;
  let strike = 0.04;
  let leg = floating_leg(
    valuation,
    d(2026, 1, 15),
    notional,
    DayCountConvention::Actual360,
  );
  let curve = flat_curve(0.04, 2.2);

  let cap = Cap::new(strike, leg.clone(), BlackVolatility::new(0.3));
  let floor = Floor::new(strike, leg.clone(), BlackVolatility::new(0.3));
  let collar = Collar::new(cap, floor);
  let val = collar.valuation(
    valuation,
    DayCountConvention::Actual365Fixed,
    DayCountConvention::Actual365Fixed,
    &curve,
  );

  let mut manual = 0.0;
  for (i, cashflow) in leg.cashflows().iter().enumerate() {
    let Cashflow::Floating(coupon) = cashflow else {
      continue;
    };
    let tau_payment =
      DayCountConvention::Actual365Fixed.year_fraction(valuation, coupon.period.payment_date);
    let df = curve.discount_factor(tau_payment);
    let forward = val.cap.forward_rates[i];
    manual += df * notional * coupon.period.accrual_factor * (forward - strike);
  }

  approx(val.npv, manual, TOL);
}
