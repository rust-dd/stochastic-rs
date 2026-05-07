//! Comparison tests for Tier 1 fixed-income instruments.
//!
//! Bond prices are checked against direct discounted-cashflow formulas and
//! swap par rates are checked against textbook annuity identities.

use chrono::NaiveDate;
use ndarray::array;
use stochastic_rs::quant::calendar::DayCountConvention;
use stochastic_rs::quant::calendar::Frequency;
use stochastic_rs::quant::calendar::ScheduleBuilder;
use stochastic_rs::quant::cashflows::FloatingIndex;
use stochastic_rs::quant::cashflows::IborIndex;
use stochastic_rs::quant::cashflows::NotionalSchedule;
use stochastic_rs::quant::cashflows::OvernightIndex;
use stochastic_rs::quant::cashflows::RateTenor;
use stochastic_rs::quant::curves::Compounding;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::InterpolationMethod;
use stochastic_rs::quant::curves::MultiCurve;
use stochastic_rs::quant::fx::currency::EUR;
use stochastic_rs::quant::fx::currency::USD;
use stochastic_rs::quant::instruments::AmortizingFixedRateBond;
use stochastic_rs::quant::instruments::BasisSwap;
use stochastic_rs::quant::instruments::CrossCurrencyBasisSwap;
use stochastic_rs::quant::instruments::CrossCurrencySwapDirection;
use stochastic_rs::quant::instruments::FixedRateBond;
use stochastic_rs::quant::instruments::FloatingRateBond;
use stochastic_rs::quant::instruments::InflationLinkedBond;
use stochastic_rs::quant::instruments::SwapDirection;
use stochastic_rs::quant::instruments::VanillaInterestRateSwap;
use stochastic_rs::quant::instruments::ZeroCouponBond;

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
fn fixed_rate_bond_curve_price_matches_manual_discounting() {
  let valuation = d(2024, 1, 15);
  let schedule = ScheduleBuilder::new(valuation, d(2026, 1, 15))
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let bond = FixedRateBond::new(
    &schedule,
    100.0,
    0.05,
    Frequency::SemiAnnual,
    DayCountConvention::Thirty360,
  );
  let curve = flat_curve(0.04, 2.2);
  let price = bond.price_from_curve(valuation, DayCountConvention::Actual365Fixed, &curve);

  let mut manual = 0.0;
  for window in schedule.adjusted_dates.windows(2) {
    let delta: f64 = DayCountConvention::Thirty360.year_fraction(window[0], window[1]);
    let tau: f64 = DayCountConvention::Actual365Fixed.year_fraction(valuation, window[1]);
    manual += 100.0 * 0.05 * delta * curve.discount_factor(tau);
  }
  let maturity_tau: f64 =
    DayCountConvention::Actual365Fixed.year_fraction(valuation, d(2026, 1, 15));
  manual += 100.0 * curve.discount_factor(maturity_tau);

  approx(price.dirty_price, manual, TOL);
  approx(price.clean_price, manual, TOL);
  approx(price.accrued_interest, 0.0, TOL);
}

#[test]
fn fixed_rate_bond_ytm_reprices_clean_price() {
  let valuation = d(2024, 4, 15);
  let schedule = ScheduleBuilder::new(d(2024, 1, 15), d(2027, 1, 15))
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let bond = FixedRateBond::new(
    &schedule,
    100.0,
    0.045,
    Frequency::SemiAnnual,
    DayCountConvention::Thirty360,
  );

  let compounding = bond.standard_yield_compounding();
  let clean = bond.clean_price_from_yield(
    valuation,
    0.0425,
    DayCountConvention::Actual365Fixed,
    compounding,
  );
  let implied = bond.yield_to_maturity_from_clean_price(
    valuation,
    clean,
    DayCountConvention::Actual365Fixed,
    compounding,
  );

  approx(implied, 0.0425, 1e-9);
}

#[test]
fn fixed_rate_bond_spread_analytics_are_consistent() {
  let valuation = d(2024, 1, 15);
  let fixed_schedule = ScheduleBuilder::new(valuation, d(2027, 1, 15))
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let float_schedule = ScheduleBuilder::new(valuation, d(2027, 1, 15))
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let discount_curve = flat_curve(0.03, 3.2);
  let mut curves = MultiCurve::new(discount_curve.clone());
  curves.add_forecast("3M", discount_curve.clone());

  let par_swap = VanillaInterestRateSwap::new(
    SwapDirection::Receiver,
    &fixed_schedule,
    &float_schedule,
    100.0,
    0.0,
    DayCountConvention::Thirty360,
    FloatingIndex::Ibor(IborIndex::new(
      "USD-LIBOR-3M",
      RateTenor::ThreeMonths,
      DayCountConvention::Actual365Fixed,
    )),
    0.0,
    DayCountConvention::Actual365Fixed,
  );
  let par_coupon = par_swap.fair_rate(valuation, DayCountConvention::Actual365Fixed, &curves);
  let bond = FixedRateBond::new(
    &fixed_schedule,
    100.0,
    par_coupon,
    Frequency::SemiAnnual,
    DayCountConvention::Thirty360,
  );
  let model_price = bond.price_from_curve(valuation, DayCountConvention::Actual365Fixed, &curves);

  approx(
    bond.z_spread_from_dirty_price(
      valuation,
      model_price.dirty_price,
      DayCountConvention::Actual365Fixed,
      &curves,
    ),
    0.0,
    1e-10,
  );
  approx(
    bond.option_adjusted_spread_from_dirty_price(
      valuation,
      model_price.dirty_price - 2.0,
      DayCountConvention::Actual365Fixed,
      &curves,
      2.0,
    ),
    0.0,
    1e-10,
  );
  approx(
    bond.asset_swap_spread_from_dirty_price(
      valuation,
      model_price.dirty_price,
      DayCountConvention::Actual365Fixed,
      &curves,
    ),
    0.0,
    1e-8,
  );
}

#[test]
fn floating_rate_bond_is_near_par_under_single_curve_projection() {
  let valuation = d(2024, 1, 15);
  let schedule = ScheduleBuilder::new(valuation, d(2025, 1, 15))
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let bond = FloatingRateBond::new(
    &schedule,
    100.0,
    FloatingIndex::Ibor(IborIndex::new(
      "USD-LIBOR-3M",
      RateTenor::ThreeMonths,
      DayCountConvention::Actual365Fixed,
    )),
    0.0,
    DayCountConvention::Actual365Fixed,
  );
  let curve = flat_curve(0.0325, 1.2);
  let price = bond.price_from_curve(valuation, DayCountConvention::Actual365Fixed, &curve);

  approx(price.dirty_price, 100.0, 1e-8);
  approx(price.clean_price, 100.0, 1e-8);
  approx(price.accrued_interest, 0.0, 1e-10);
}

#[test]
fn amortizing_bond_curve_price_matches_manual_cashflows() {
  let valuation = d(2024, 1, 15);
  let schedule = ScheduleBuilder::new(valuation, d(2025, 1, 15))
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let notionals = NotionalSchedule::from_array(array![100.0, 80.0, 60.0, 40.0]);
  let bond = AmortizingFixedRateBond::new(
    &schedule,
    notionals.clone(),
    0.05,
    Frequency::Quarterly,
    DayCountConvention::Actual365Fixed,
  );
  let curve = flat_curve(0.04, 1.2);
  let price = bond.price_from_curve(valuation, DayCountConvention::Actual365Fixed, &curve);

  let mut manual = 0.0;
  for (idx, window) in schedule.adjusted_dates.windows(2).enumerate() {
    let delta: f64 = DayCountConvention::Actual365Fixed.year_fraction(window[0], window[1]);
    let tau: f64 = DayCountConvention::Actual365Fixed.year_fraction(valuation, window[1]);
    let outstanding = notionals.notionals()[idx];
    let next = if idx + 1 < notionals.len() {
      notionals.notionals()[idx + 1]
    } else {
      0.0
    };
    let coupon = outstanding * 0.05 * delta;
    let principal = outstanding - next;
    manual += (coupon + principal) * curve.discount_factor(tau);
  }

  approx(price.dirty_price, manual, TOL);
}

#[test]
fn inflation_linked_bond_prices_projected_cashflows_and_accrual() {
  let valuation = d(2024, 1, 15);
  let schedule = ScheduleBuilder::new(valuation, d(2025, 1, 15))
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let bond = InflationLinkedBond::new(
    &schedule,
    100.0,
    0.02,
    Frequency::SemiAnnual,
    DayCountConvention::Actual365Fixed,
    1.0,
    array![1.02, 1.04],
  );
  let curve = flat_curve(0.03, 1.2);
  let price = bond.price_from_curve(valuation, DayCountConvention::Actual365Fixed, &curve);

  let first_end = d(2024, 7, 15);
  let second_end = d(2025, 1, 15);
  let delta1: f64 = DayCountConvention::Actual365Fixed.year_fraction(valuation, first_end);
  let delta2: f64 = DayCountConvention::Actual365Fixed.year_fraction(first_end, second_end);
  let tau1: f64 = DayCountConvention::Actual365Fixed.year_fraction(valuation, first_end);
  let tau2: f64 = DayCountConvention::Actual365Fixed.year_fraction(valuation, second_end);
  let manual = 100.0 * 0.02 * delta1 * 1.02 * curve.discount_factor(tau1)
    + 100.0 * 0.02 * delta2 * 1.04 * curve.discount_factor(tau2)
    + 100.0 * 1.04 * curve.discount_factor(tau2);
  approx(price.dirty_price, manual, TOL);

  let as_of = d(2024, 4, 15);
  let elapsed: f64 = DayCountConvention::Actual365Fixed.year_fraction(valuation, as_of);
  let full: f64 = DayCountConvention::Actual365Fixed.year_fraction(valuation, first_end);
  let interpolated_ratio = 1.0 + (1.02 - 1.0) * (elapsed / full);
  let accrued_expected = 100.0 * 0.02 * elapsed * interpolated_ratio;
  approx(bond.accrued_interest(as_of), accrued_expected, 1e-10);
}

#[test]
fn zero_coupon_bond_matches_discount_factor_identity() {
  let valuation = d(2024, 1, 15);
  let maturity = d(2025, 7, 15);
  let zcb = ZeroCouponBond::new(1000.0, maturity);
  let curve = flat_curve(0.0375, 2.0);
  let tau: f64 = DayCountConvention::Actual365Fixed.year_fraction(valuation, maturity);
  let manual = 1000.0 * curve.discount_factor(tau);

  approx(
    zcb.price_from_curve(valuation, DayCountConvention::Actual365Fixed, &curve),
    manual,
    TOL,
  );
  approx(
    zcb.price_from_yield(
      valuation,
      0.0375,
      DayCountConvention::Actual365Fixed,
      Compounding::Continuous,
    ),
    manual,
    5e-6,
  );
}

#[test]
fn vanilla_swap_fair_rate_matches_annuity_formula_and_npv_is_zero_at_par() {
  let valuation = d(2024, 1, 15);
  let fixed_schedule = ScheduleBuilder::new(valuation, d(2026, 1, 15))
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let float_schedule = ScheduleBuilder::new(valuation, d(2026, 1, 15))
    .frequency(Frequency::Quarterly)
    .forward()
    .build();

  let discount_curve = flat_curve(0.03, 2.2);
  let forecast_curve = flat_curve(0.035, 2.2);
  let mut curves = MultiCurve::new(discount_curve.clone());
  curves.add_forecast("3M", forecast_curve.clone());

  let par_swap = VanillaInterestRateSwap::new(
    SwapDirection::Receiver,
    &fixed_schedule,
    &float_schedule,
    1_000_000.0,
    0.0,
    DayCountConvention::Actual360,
    FloatingIndex::Ibor(IborIndex::new(
      "USD-LIBOR-3M",
      RateTenor::ThreeMonths,
      DayCountConvention::Actual365Fixed,
    )),
    0.0,
    DayCountConvention::Actual365Fixed,
  );
  let fair_rate = par_swap.fair_rate(valuation, DayCountConvention::Actual365Fixed, &curves);

  let mut manual_annuity = 0.0;
  for window in fixed_schedule.adjusted_dates.windows(2) {
    let delta: f64 = DayCountConvention::Actual360.year_fraction(window[0], window[1]);
    let tau: f64 = DayCountConvention::Actual365Fixed.year_fraction(valuation, window[1]);
    manual_annuity += discount_curve.discount_factor(tau) * delta * 1_000_000.0;
  }

  let mut manual_float = 0.0;
  for window in float_schedule.adjusted_dates.windows(2) {
    let start_tau: f64 = DayCountConvention::Actual365Fixed.year_fraction(valuation, window[0]);
    let end_tau: f64 = DayCountConvention::Actual365Fixed.year_fraction(valuation, window[1]);
    let delta: f64 = DayCountConvention::Actual365Fixed.year_fraction(window[0], window[1]);
    let df = discount_curve.discount_factor(end_tau);
    let fwd = forecast_curve.simple_forward_rate(start_tau, end_tau);
    manual_float += df * delta * fwd * 1_000_000.0;
  }
  let manual_fair = manual_float / manual_annuity;
  approx(fair_rate, manual_fair, 1e-10);

  let priced_swap = VanillaInterestRateSwap::new(
    SwapDirection::Receiver,
    &fixed_schedule,
    &float_schedule,
    1_000_000.0,
    fair_rate,
    DayCountConvention::Actual360,
    FloatingIndex::Ibor(IborIndex::new(
      "USD-LIBOR-3M",
      RateTenor::ThreeMonths,
      DayCountConvention::Actual365Fixed,
    )),
    0.0,
    DayCountConvention::Actual365Fixed,
  );
  let valuation_summary =
    priced_swap.valuation(valuation, DayCountConvention::Actual365Fixed, &curves);
  approx(valuation_summary.net_npv, 0.0, 1e-8);
  approx(valuation_summary.bpv, manual_annuity * 1e-4, 1e-8);
  approx(valuation_summary.dv01, manual_annuity * 1e-4, 1e-8);
}

#[test]
fn basis_swap_fair_pay_leg_spread_zeroes_npv() {
  let valuation = d(2024, 1, 15);
  let pay_schedule = ScheduleBuilder::new(valuation, d(2026, 1, 15))
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let receive_schedule = ScheduleBuilder::new(valuation, d(2026, 1, 15))
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let discount_curve = flat_curve(0.03, 2.2);
  let mut curves = MultiCurve::new(discount_curve);
  curves.add_forecast("3M", flat_curve(0.034, 2.2));
  curves.add_forecast("6M", flat_curve(0.036, 2.2));

  let basis = BasisSwap::new(
    &pay_schedule,
    &receive_schedule,
    1_000_000.0,
    FloatingIndex::Ibor(IborIndex::new(
      "USD-LIBOR-3M",
      RateTenor::ThreeMonths,
      DayCountConvention::Actual365Fixed,
    )),
    0.0,
    DayCountConvention::Actual365Fixed,
    1_000_000.0,
    FloatingIndex::Ibor(IborIndex::new(
      "USD-LIBOR-6M",
      RateTenor::SixMonths,
      DayCountConvention::Actual365Fixed,
    )),
    0.0,
    DayCountConvention::Actual365Fixed,
  );
  let valuation_summary = basis.valuation(valuation, DayCountConvention::Actual365Fixed, &curves);

  let repriced = BasisSwap::new(
    &pay_schedule,
    &receive_schedule,
    1_000_000.0,
    FloatingIndex::Ibor(IborIndex::new(
      "USD-LIBOR-3M",
      RateTenor::ThreeMonths,
      DayCountConvention::Actual365Fixed,
    )),
    valuation_summary.fair_spread_on_pay_leg,
    DayCountConvention::Actual365Fixed,
    1_000_000.0,
    FloatingIndex::Ibor(IborIndex::new(
      "USD-LIBOR-6M",
      RateTenor::SixMonths,
      DayCountConvention::Actual365Fixed,
    )),
    0.0,
    DayCountConvention::Actual365Fixed,
  );
  approx(
    repriced.npv(valuation, DayCountConvention::Actual365Fixed, &curves),
    0.0,
    1e-8,
  );
}

#[test]
fn cross_currency_basis_swap_fair_domestic_spread_zeroes_npv() {
  let valuation = d(2024, 1, 15);
  let domestic_schedule = ScheduleBuilder::new(valuation, d(2025, 1, 15))
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let foreign_schedule = ScheduleBuilder::new(valuation, d(2025, 1, 15))
    .frequency(Frequency::Quarterly)
    .forward()
    .build();

  let mut domestic_curves = MultiCurve::new(flat_curve(0.03, 1.2));
  domestic_curves.add_forecast("3M", flat_curve(0.031, 1.2));
  let mut foreign_curves = MultiCurve::new(flat_curve(0.02, 1.2));
  foreign_curves.add_forecast("3M", flat_curve(0.021, 1.2));

  let xccy = CrossCurrencyBasisSwap::new(
    CrossCurrencySwapDirection::PayDomesticReceiveForeign,
    USD,
    EUR,
    1.2,
    &domestic_schedule,
    &foreign_schedule,
    1_200_000.0,
    FloatingIndex::Ibor(IborIndex::new(
      "USD-LIBOR-3M",
      RateTenor::ThreeMonths,
      DayCountConvention::Actual365Fixed,
    )),
    0.0,
    DayCountConvention::Actual365Fixed,
    1_000_000.0,
    FloatingIndex::Ibor(IborIndex::new(
      "EUR-EURIBOR-3M",
      RateTenor::ThreeMonths,
      DayCountConvention::Actual365Fixed,
    )),
    0.0,
    DayCountConvention::Actual365Fixed,
    false,
  );
  let valuation_summary = xccy.valuation(
    valuation,
    DayCountConvention::Actual365Fixed,
    &domestic_curves,
    DayCountConvention::Actual365Fixed,
    &foreign_curves,
  );

  let repriced = CrossCurrencyBasisSwap::new(
    CrossCurrencySwapDirection::PayDomesticReceiveForeign,
    USD,
    EUR,
    1.2,
    &domestic_schedule,
    &foreign_schedule,
    1_200_000.0,
    FloatingIndex::Ibor(IborIndex::new(
      "USD-LIBOR-3M",
      RateTenor::ThreeMonths,
      DayCountConvention::Actual365Fixed,
    )),
    valuation_summary.fair_domestic_spread,
    DayCountConvention::Actual365Fixed,
    1_000_000.0,
    FloatingIndex::Ibor(IborIndex::new(
      "EUR-EURIBOR-3M",
      RateTenor::ThreeMonths,
      DayCountConvention::Actual365Fixed,
    )),
    0.0,
    DayCountConvention::Actual365Fixed,
    false,
  );
  approx(
    repriced.npv(
      valuation,
      DayCountConvention::Actual365Fixed,
      &domestic_curves,
      DayCountConvention::Actual365Fixed,
      &foreign_curves,
    ),
    0.0,
    1e-8,
  );
}

#[test]
fn overnight_indexed_swap_constructor_reuses_overnight_float_leg() {
  let valuation = d(2024, 1, 15);
  let fixed_schedule = ScheduleBuilder::new(valuation, d(2025, 1, 15))
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let float_schedule = ScheduleBuilder::new(valuation, d(2025, 1, 15))
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let curve = flat_curve(0.025, 1.1);
  let mut curves = MultiCurve::new(curve.clone());
  curves.add_forecast("O/N", curve);

  let ois = VanillaInterestRateSwap::overnight_indexed(
    SwapDirection::Payer,
    &fixed_schedule,
    &float_schedule,
    5_000_000.0,
    0.026,
    DayCountConvention::Actual360,
    OvernightIndex::new("SOFR", DayCountConvention::Actual365Fixed),
    0.0,
    DayCountConvention::Actual365Fixed,
  );
  let valuation_summary = ois.valuation(valuation, DayCountConvention::Actual365Fixed, &curves);

  assert!(valuation_summary.fair_rate.is_finite());
  assert!(valuation_summary.fixed_leg_npv.is_finite());
  assert!(valuation_summary.floating_leg_npv.is_finite());
  assert!(valuation_summary.net_npv.is_finite());
  assert!(valuation_summary.dv01 < 0.0);
}
