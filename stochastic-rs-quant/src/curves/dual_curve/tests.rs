//! The forecast bootstrap recovers a tenor curve from quotes priced off it.

use ndarray::Array1;

use super::*;
use crate::QuantError;
use crate::cashflows::RateTenor;
use crate::curves::bootstrap::BootstrapSpec;
use crate::curves::bootstrap::bootstrap;
use crate::curves::multi_curve::MultiCurve;
use crate::curves::types::BootstrapInstrument;

fn forecast_curve(
  quotes: &[BootstrapInstrument<f64>],
  ois: &DiscountCurve<f64>,
) -> Result<DiscountCurve<f64>, QuantError> {
  let method = InterpolationMethod::LogLinearOnDiscountFactors;
  bootstrap(BootstrapSpec::new(quotes, method).discounting_with(ois))
}

fn flat(rate: f64) -> DiscountCurve<f64> {
  DiscountCurve::from_zero_rates(
    &Array1::from_vec(vec![0.25, 1.0, 5.0, 10.0]),
    &Array1::from_vec(vec![rate; 4]),
    InterpolationMethod::LogLinearOnDiscountFactors,
  )
}

/// Par rate of a swap against the tenor from the true curves.
fn par_rate(
  discount: &DiscountCurve<f64>,
  tenor_df: impl Fn(f64) -> f64,
  fixed: &[f64],
  float: &[f64],
) -> f64 {
  let mut float_pv = 0.0;
  let mut prev = 0.0;
  for t in float {
    float_pv += discount.discount_factor(*t) * (tenor_df(prev) / tenor_df(*t) - 1.0);
    prev = *t;
  }
  let mut annuity = 0.0;
  prev = 0.0;
  for t in fixed {
    annuity += (t - prev) * discount.discount_factor(*t);
    prev = *t;
  }
  float_pv / annuity
}

fn quotes(
  discount: &DiscountCurve<f64>,
  tenor_df: impl Fn(f64) -> f64 + Copy,
) -> Vec<BootstrapInstrument<f64>> {
  let mut q = vec![
    BootstrapInstrument::Deposit {
      maturity: 0.25,
      rate: (1.0 / tenor_df(0.25) - 1.0) / 0.25,
    },
    BootstrapInstrument::Fra {
      start: 0.25,
      end: 0.5,
      rate: (tenor_df(0.25) / tenor_df(0.5) - 1.0) / 0.25,
    },
  ];
  for years in [1_usize, 2, 3, 5, 7] {
    let fixed: Vec<f64> = (1..=years).map(|i| i as f64).collect();
    let float: Vec<f64> = (1..=4 * years).map(|i| 0.25 * i as f64).collect();
    q.push(BootstrapInstrument::SwapWithSchedule {
      rate: par_rate(discount, tenor_df, &fixed, &float),
      fixed_times: fixed,
      float_times: float,
    });
  }
  q
}

#[test]
fn single_curve_quotes_reproduce_the_discount_curve() {
  let ois = flat(0.02);
  let df = |t: f64| (-0.02_f64 * t).exp();
  let forecast = forecast_curve(&quotes(&ois, df), &ois).unwrap();
  for t in [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0] {
    assert!(
      (forecast.discount_factor(t) - df(t)).abs() < 1e-10,
      "t = {t}: {}",
      forecast.discount_factor(t)
    );
  }
}

/// A tenor trading 50 bp above OIS: the bootstrap recovers its
/// pseudo-discount factors at every pillar and the multi-curve basis reads
/// the spread back.
#[test]
fn recovers_a_tenor_curve_with_basis_over_ois() {
  let ois = flat(0.02);
  let tenor = |t: f64| (-0.025_f64 * t).exp();
  let forecast = forecast_curve(&quotes(&ois, tenor), &ois).unwrap();
  for t in [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0] {
    assert!(
      (forecast.discount_factor(t) - tenor(t)).abs() < 1e-9,
      "t = {t}: {}",
      forecast.discount_factor(t)
    );
  }
  let mut multi = MultiCurve::new(ois);
  multi.add_forecast(RateTenor::ThreeMonths, forecast);
  let basis = multi
    .basis_spread(RateTenor::ThreeMonths, 1.0, 1.25)
    .expect("tenor registered");
  assert!((basis - 0.005).abs() < 2e-4, "basis {basis}");
}

#[test]
fn deposit_and_fra_extend_the_short_end_multiplicatively() {
  let ois = flat(0.02);
  let q = [
    BootstrapInstrument::Deposit {
      maturity: 0.5,
      rate: 0.03,
    },
    BootstrapInstrument::Fra {
      start: 0.5,
      end: 1.0,
      rate: 0.032,
    },
  ];
  let forecast = forecast_curve(&q, &ois).unwrap();
  let p_half = 1.0 / (1.0 + 0.03 * 0.5);
  assert!((forecast.discount_factor(0.5) - p_half).abs() < 1e-14);
  assert!((forecast.discount_factor(1.0) - p_half / (1.0 + 0.032 * 0.5)).abs() < 1e-14);
}

#[test]
fn rejects_swap_schedules_with_different_maturities() {
  let ois = flat(0.02);
  let q = [BootstrapInstrument::SwapWithSchedule {
    rate: 0.02,
    fixed_times: vec![1.0, 2.0],
    float_times: vec![0.5, 1.0, 1.5],
  }];
  assert!(matches!(
    forecast_curve(&q, &ois),
    Err(QuantError::InvalidQuote { index: 0, .. })
  ));
}
