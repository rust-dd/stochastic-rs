//! European swaption pricing tests.
//!
//! Covers Black-76 put-call parity for payer vs receiver swaptions, the
//! Bachelier-vs-Black ATM agreement at small volatility, and SABR /
//! shifted-SABR sanity checks.

use chrono::NaiveDate;
use ndarray::array;
use stochastic_rs::quant::calendar::DayCountConvention;
use stochastic_rs::quant::calendar::Frequency;
use stochastic_rs::quant::calendar::ScheduleBuilder;
use stochastic_rs::quant::cashflows::FloatingIndex;
use stochastic_rs::quant::cashflows::IborIndex;
use stochastic_rs::quant::cashflows::RateTenor;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::InterpolationMethod;
use stochastic_rs::quant::instruments::BachelierVolatility;
use stochastic_rs::quant::instruments::BlackVolatility;
use stochastic_rs::quant::instruments::EuropeanSwaption;
use stochastic_rs::quant::instruments::SabrVolatility;
use stochastic_rs::quant::instruments::ShiftedSabrVolatility;
use stochastic_rs::quant::instruments::SwapDirection;
use stochastic_rs::quant::instruments::SwaptionDirection;
use stochastic_rs::quant::instruments::VanillaInterestRateSwap;

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
fn european_payer_put_call_parity_against_forward_swap_value() {
  let valuation = d(2024, 1, 15);
  let expiry = d(2025, 1, 15);
  let start = d(2025, 1, 15);
  let end = d(2030, 1, 15);
  let notional = 10_000_000.0;
  let strike = 0.045;

  let fixed_schedule = ScheduleBuilder::new(start, end)
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let float_schedule = ScheduleBuilder::new(start, end)
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let index = FloatingIndex::Ibor(IborIndex::new(
    "LIBOR_3M",
    RateTenor::ThreeMonths,
    DayCountConvention::Actual360,
  ));
  let swap = VanillaInterestRateSwap::new(
    SwapDirection::Payer,
    &fixed_schedule,
    &float_schedule,
    notional,
    strike,
    DayCountConvention::Thirty360,
    index,
    0.0,
    DayCountConvention::Actual360,
  );
  let curve = flat_curve(0.04, 6.5);
  let vol = BlackVolatility::new(0.3);

  let payer = EuropeanSwaption::new(SwaptionDirection::Payer, strike, expiry, swap.clone(), vol);
  let receiver = EuropeanSwaption::new(
    SwaptionDirection::Receiver,
    strike,
    expiry,
    swap.clone(),
    vol,
  );

  let payer_val = payer.valuation(
    valuation,
    DayCountConvention::Actual365Fixed,
    DayCountConvention::Actual365Fixed,
    &curve,
  );
  let receiver_val = receiver.valuation(
    valuation,
    DayCountConvention::Actual365Fixed,
    DayCountConvention::Actual365Fixed,
    &curve,
  );

  let expected_diff = payer_val.annuity * (payer_val.forward_swap_rate - strike);
  let actual_diff = payer_val.npv - receiver_val.npv;
  approx(actual_diff, expected_diff, 1e-6);
}

#[test]
fn bachelier_swaption_matches_black_atm_at_small_vol() {
  let valuation = d(2024, 1, 15);
  let expiry = d(2025, 1, 15);
  let start = d(2025, 1, 15);
  let end = d(2030, 1, 15);
  let notional = 1_000_000.0;

  let fixed_schedule = ScheduleBuilder::new(start, end)
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let float_schedule = ScheduleBuilder::new(start, end)
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let index = FloatingIndex::Ibor(IborIndex::new(
    "LIBOR_3M",
    RateTenor::ThreeMonths,
    DayCountConvention::Actual360,
  ));
  let curve = flat_curve(0.04, 6.5);

  let forward_swap_rate = {
    let swap = VanillaInterestRateSwap::new(
      SwapDirection::Payer,
      &fixed_schedule,
      &float_schedule,
      notional,
      0.04,
      DayCountConvention::Thirty360,
      index.clone(),
      0.0,
      DayCountConvention::Actual360,
    );
    swap
      .valuation(valuation, DayCountConvention::Actual365Fixed, &curve)
      .fair_rate
  };

  let strike = forward_swap_rate;
  let swap = VanillaInterestRateSwap::new(
    SwapDirection::Payer,
    &fixed_schedule,
    &float_schedule,
    notional,
    strike,
    DayCountConvention::Thirty360,
    index,
    0.0,
    DayCountConvention::Actual360,
  );

  let sigma_lognormal = 0.2;
  let sigma_normal = sigma_lognormal * forward_swap_rate;

  let black = EuropeanSwaption::new(
    SwaptionDirection::Payer,
    strike,
    expiry,
    swap.clone(),
    BlackVolatility::new(sigma_lognormal),
  );
  let bachelier = EuropeanSwaption::new(
    SwaptionDirection::Payer,
    strike,
    expiry,
    swap,
    BachelierVolatility::new(sigma_normal),
  );

  let black_npv = black.npv(
    valuation,
    DayCountConvention::Actual365Fixed,
    DayCountConvention::Actual365Fixed,
    &curve,
  );
  let bachelier_npv = bachelier.npv(
    valuation,
    DayCountConvention::Actual365Fixed,
    DayCountConvention::Actual365Fixed,
    &curve,
  );

  let rel_err = (black_npv - bachelier_npv).abs() / black_npv;
  assert!(rel_err < 0.05, "ATM Bachelier/Black mismatch: {rel_err:.4}");
}

#[test]
fn sabr_swaption_produces_positive_price() {
  let valuation = d(2024, 1, 15);
  let expiry = d(2025, 1, 15);
  let start = d(2025, 1, 15);
  let end = d(2030, 1, 15);
  let notional = 1_000_000.0;
  let strike = 0.045;

  let fixed_schedule = ScheduleBuilder::new(start, end)
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let float_schedule = ScheduleBuilder::new(start, end)
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let index = FloatingIndex::Ibor(IborIndex::new(
    "LIBOR_3M",
    RateTenor::ThreeMonths,
    DayCountConvention::Actual360,
  ));
  let swap = VanillaInterestRateSwap::new(
    SwapDirection::Payer,
    &fixed_schedule,
    &float_schedule,
    notional,
    strike,
    DayCountConvention::Thirty360,
    index,
    0.0,
    DayCountConvention::Actual360,
  );
  let curve = flat_curve(0.04, 6.5);

  let swpn = EuropeanSwaption::new(
    SwaptionDirection::Payer,
    strike,
    expiry,
    swap,
    SabrVolatility::new(0.3, 0.5, 0.4, -0.2),
  );
  let npv = swpn.npv(
    valuation,
    DayCountConvention::Actual365Fixed,
    DayCountConvention::Actual365Fixed,
    &curve,
  );
  assert!(npv > 0.0, "Sabr swaption must have positive value");
  assert!(npv.is_finite());
}

#[test]
fn shifted_sabr_matches_plain_sabr_at_zero_shift() {
  let plain = SabrVolatility::new(0.3_f64, 0.5, 0.4, -0.2);
  let shifted = ShiftedSabrVolatility::new(0.3_f64, 0.5, 0.4, -0.2, 0.0);
  use stochastic_rs::quant::instruments::VolatilityModel;
  let v_plain = plain.implied_volatility(0.04, 0.045, 1.5);
  let v_shift = shifted.implied_volatility(0.04, 0.045, 1.5);
  approx(v_plain, v_shift, 1e-12);
}

#[test]
fn shifted_sabr_handles_zero_rate() {
  let shifted = ShiftedSabrVolatility::new(0.25_f64, 0.5, 0.4, -0.3, 0.03);
  use stochastic_rs::quant::instruments::VolatilityModel;
  let v = shifted.implied_volatility(0.001, 0.0, 2.0);
  assert!(
    v.is_finite() && v > 0.0,
    "shifted vol at zero strike must be positive, got {v}"
  );
}
