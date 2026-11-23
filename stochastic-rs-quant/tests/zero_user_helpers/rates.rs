use chrono::NaiveDate;
use ndarray::array;
use stochastic_rs_quant::calendar::DateGenerationRule;
use stochastic_rs_quant::calendar::DayCountConvention;
use stochastic_rs_quant::calendar::Frequency;
use stochastic_rs_quant::calendar::ScheduleBuilder;
use stochastic_rs_quant::cashflows::AccrualPeriod;
use stochastic_rs_quant::cashflows::NotionalSchedule;
use stochastic_rs_quant::cashflows::RateTenor;
use stochastic_rs_quant::curves::Compounding;
use stochastic_rs_quant::curves::CurvePoint;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::curves::MultiCurve;
use stochastic_rs_quant::curves::interpolation::zero_rates_from_points;
use stochastic_rs_quant::fx::CurrencyPair;
use stochastic_rs_quant::fx::currency;
use stochastic_rs_quant::fx::forward::FxForward;
use stochastic_rs_quant::instruments::FixedRateBond;
use stochastic_rs_quant::market::ibor;
use stochastic_rs_quant::market::overnight;

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
  NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

#[test]
fn invert_rate_is_the_reciprocal() {
  let pair = CurrencyPair::new(currency::EUR, currency::USD);
  assert_eq!(pair.invert_rate(1.25_f64), 1.0 / 1.25);
}

/// The price at an OAS is the Z-spread price at that spread less the option value, and the
/// implied OAS inverts it.
#[test]
fn dirty_price_from_option_adjusted_spread_is_the_z_spread_price_less_the_option() {
  let schedule = ScheduleBuilder::new(d(2024, 1, 15), d(2029, 1, 15))
    .frequency(Frequency::SemiAnnual)
    .rule(DateGenerationRule::Forward)
    .build();
  let bond = FixedRateBond::new(
    &schedule,
    100.0,
    0.05,
    Frequency::SemiAnnual,
    DayCountConvention::Thirty360,
  );
  let times = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
  let curve = DiscountCurve::from_zero_rates(
    times.view(),
    times.mapv(|_| 0.04).view(),
    InterpolationMethod::LinearOnZeroRates,
  );
  let (valuation, dcc) = (d(2024, 4, 15), DayCountConvention::Actual365Fixed);
  let (oas, option_value) = (0.0125_f64, 1.75);
  let price =
    bond.dirty_price_from_option_adjusted_spread(valuation, dcc, &curve, oas, option_value);
  assert_eq!(
    price,
    bond.dirty_price_from_z_spread(valuation, dcc, &curve, oas) - option_value
  );
  let implied =
    bond.option_adjusted_spread_from_dirty_price(valuation, price, dcc, &curve, option_value);
  assert!((implied - oas).abs() < 1e-9, "{implied}");
}

/// Six-month Euribor: Act/360, a two-day spot lag and the TARGET calendar, which closes on
/// 1 May and opens on 4 July.
#[test]
fn euribor_6m_carries_its_tenor_day_count_and_calendar() {
  let index = ibor::euribor_6m::<f64>();
  assert_eq!(index.index.name, "EURIBOR6M");
  assert_eq!(index.index.tenor, RateTenor::SixMonths);
  assert_eq!(index.index.day_count, DayCountConvention::Actual360);
  assert_eq!(index.currency, currency::EUR);
  assert_eq!(index.spot_lag, 2);
  assert!(!index.calendar.is_business_day(d(2025, 5, 1)));
  assert!(index.calendar.is_business_day(d(2025, 7, 4)));
}

/// Fed funds: overnight, Act/360 and the US calendar, which opens on 1 May and closes on
/// 4 July.
#[test]
fn fed_funds_carries_its_tenor_day_count_and_calendar() {
  let index = overnight::fed_funds::<f64>();
  assert_eq!(index.index.name, "EFFR");
  assert_eq!(index.index.tenor, RateTenor::Overnight);
  assert_eq!(index.index.day_count, DayCountConvention::Actual360);
  assert_eq!(index.currency, currency::USD);
  assert!(index.calendar.is_business_day(d(2025, 5, 1)));
  assert!(!index.calendar.is_business_day(d(2025, 7, 4)));
}

/// The year fraction from the valuation date to the payment date under the given day count;
/// zero for a payment on or before the valuation date.
#[test]
fn payment_time_from_is_the_year_fraction_to_the_payment_date() {
  let period = AccrualPeriod::<f64>::new(
    d(2025, 1, 15),
    d(2025, 7, 15),
    d(2025, 7, 17),
    DayCountConvention::Thirty360,
  );
  let dcc = DayCountConvention::Actual365Fixed;
  let valuation = d(2025, 3, 1);
  assert_eq!(
    period.payment_time_from(valuation, dcc),
    dcc.year_fraction::<f64>(valuation, d(2025, 7, 17))
  );
  assert_eq!(period.payment_time_from(valuation, dcc), 138.0 / 365.0);
  assert_eq!(period.payment_time_from(d(2025, 7, 17), dcc), 0.0);
}

/// Both profiles are linear from `start` to `end`: `startᵢ = start + (end − start)·i/(n − 1)`.
#[test]
fn accreting_and_amortizing_are_linear_between_their_ends() {
  let linear = |start: f64, end: f64| (0..5).map(move |i| start + (end - start) * i as f64 / 4.0);
  let accreting = NotionalSchedule::accreting(5, 100.0, 200.0);
  assert_eq!(
    accreting.notionals().to_vec(),
    linear(100.0, 200.0).collect::<Vec<_>>()
  );
  let amortizing = NotionalSchedule::amortizing(5, 200.0, 100.0);
  assert_eq!(
    amortizing.notionals().to_vec(),
    linear(200.0, 100.0).collect::<Vec<_>>()
  );
}

#[test]
#[should_panic(expected = "start must satisfy `start <= end`")]
fn accreting_panics_on_a_falling_profile() {
  let _ = NotionalSchedule::accreting(5, 200.0, 100.0);
}

#[test]
#[should_panic(expected = "start must satisfy `start >= end`")]
fn amortizing_panics_on_a_rising_profile() {
  let _ = NotionalSchedule::amortizing(5, 100.0, 200.0);
}

/// `−ln Dᵢ / tᵢ` per pillar.
#[test]
fn zero_rates_from_points_is_minus_log_discount_over_time() {
  let point = |time: f64, rate: f64| CurvePoint {
    time,
    discount_factor: (-rate * time).exp(),
  };
  let points = [point(0.5, 0.03), point(2.0, 0.035), point(10.0, 0.041)];
  let rates = zero_rates_from_points(&points);
  for (rate, expected) in rates.iter().zip([0.03, 0.035, 0.041]) {
    assert!((rate - expected).abs() < 1e-15, "{rate} against {expected}");
  }
}

/// A flat 3 % continuous curve, `D(t) = e^{−0.03t}` at every `t`.
fn flat_three_percent() -> DiscountCurve<f64> {
  DiscountCurve::from_zero_rates(
    array![0.5, 1.0, 5.0].view(),
    array![0.03, 0.03, 0.03].view(),
    InterpolationMethod::LogLinearOnDiscountFactors,
  )
}

/// `Σᵢ (tᵢ − tᵢ₋₁)·D(tᵢ)·rate·notional` over the schedule's four periods from its first date.
#[test]
fn pv_fixed_leg_sums_each_accrual_discounted_to_its_end() {
  let curves = MultiCurve::new(flat_three_percent());
  let schedule = [0.5_f64, 1.0, 1.5, 2.25, 3.0];
  let by_hand: f64 = schedule
    .windows(2)
    .map(|w| (w[1] - w[0]) * (-0.03 * w[1]).exp() * 0.04 * 1e6)
    .sum();
  let got = curves.pv_fixed_leg(array![0.5, 1.0, 1.5, 2.25, 3.0].view(), 0.04, 1e6);
  assert!((got - by_hand).abs() < 1e-9, "{got} vs {by_hand}");
}

/// The zero rate under each compounding inverts `D(2) = e^{−0.06}`.
#[test]
fn zero_rate_with_compounding_inverts_the_discount_factor() {
  let (curve, t) = (flat_three_percent(), 2.0);
  let df = (-0.03_f64 * t).exp();
  for (compounding, want) in [
    (Compounding::Continuous, 0.03),
    (Compounding::Simple, (1.0 / df - 1.0) / t),
    (Compounding::Periodic(1), df.powf(-1.0 / t) - 1.0),
    (
      Compounding::Periodic(4),
      4.0 * (df.powf(-1.0 / (4.0 * t)) - 1.0),
    ),
  ] {
    let got = curve.zero_rate_with_compounding(t, compounding);
    assert!((got - want).abs() < 1e-15, "{compounding}: {got} vs {want}");
  }
  assert_eq!(
    curve.zero_rate_with_compounding(t, Compounding::Continuous),
    curve.zero_rate(t)
  );
}

/// `r_f = r_d − ln(F/S)/τ` recovers the foreign rate a forward was priced with.
#[test]
fn implied_foreign_rate_round_trips_the_forward() {
  let pair = CurrencyPair::new(currency::EUR, currency::USD);
  let forward = FxForward::new(pair, 1.1, 0.03, 0.01, 0.75).forward_rate();
  let got = FxForward::<f64>::implied_foreign_rate(1.1, forward, 0.03, 0.75);
  assert!((got - 0.01).abs() < 1e-15, "{got}");
}
