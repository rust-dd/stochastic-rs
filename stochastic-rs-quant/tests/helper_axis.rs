//! A curve built from the crate's own helpers holds one time axis, the pricer's: each helper
//! places its pillar at its date's time on that axis and accrues on its own day count.

use std::sync::Arc;

use chrono::NaiveDate;
use stochastic_rs_quant::calendar::DayCountConvention;
use stochastic_rs_quant::calendar::Frequency;
use stochastic_rs_quant::calendar::ScheduleBuilder;
use stochastic_rs_quant::cashflows::FloatingIndex;
use stochastic_rs_quant::cashflows::IborIndex;
use stochastic_rs_quant::cashflows::RateTenor;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::instruments::ForwardRateAgreement;
use stochastic_rs_quant::instruments::FraPosition;
use stochastic_rs_quant::instruments::SwaptionDirection;
use stochastic_rs_quant::instruments::VanillaInterestRateSwap;
use stochastic_rs_quant::market::DepositRateHelper;
use stochastic_rs_quant::market::FraRateHelper;
use stochastic_rs_quant::market::FuturesRateHelper;
use stochastic_rs_quant::market::Handle;
use stochastic_rs_quant::market::Quote;
use stochastic_rs_quant::market::RateHelper;
use stochastic_rs_quant::market::SimpleQuote;
use stochastic_rs_quant::market::SwapRateHelper;
use stochastic_rs_quant::market::build_curve;

const ACT360: DayCountConvention = DayCountConvention::Actual360;
const AXIS: DayCountConvention = DayCountConvention::Actual365Fixed;
const METHODS: [InterpolationMethod; 3] = [
  InterpolationMethod::LogLinearOnDiscountFactors,
  InterpolationMethod::CubicSplineOnZeroRates,
  InterpolationMethod::MonotoneConvex,
];

fn d(year: i32, month: u32, day: u32) -> NaiveDate {
  NaiveDate::from_ymd_opt(year, month, day).unwrap()
}

fn quote(rate: f64) -> Handle<dyn Quote<f64>> {
  Handle::new(Arc::new(SimpleQuote::new(rate)) as Arc<dyn Quote<f64>>)
}

/// The FRA's forward on `curve` read on the ACT/365F axis.
fn forward(
  curve: &DiscountCurve<f64>,
  valuation: NaiveDate,
  start: NaiveDate,
  end: NaiveDate,
) -> f64 {
  let index = IborIndex::new("EUR-6M", RateTenor::SixMonths, ACT360);
  let fra = ForwardRateAgreement::new(FraPosition::Long, 1.0, 0.03, start, end, ACT360, index);
  fra.valuation(valuation, AXIS, curve).forward_rate
}

/// The `market` bench's mix: ACT/360 money market, ACT/365F swaps. rc.6's first cut read the 3.5 %
/// FRA 5.5 bp low and the 3 % deposit 4.1 bp low here.
#[test]
fn act360_deposit_and_fra_helpers_reprice_on_an_act365f_axis() {
  let valuation = d(2025, 1, 1);
  let deposit = DepositRateHelper::new(quote(0.03), valuation, d(2025, 7, 1), ACT360);
  let fra = FraRateHelper::new(quote(0.035), d(2025, 7, 1), d(2026, 1, 1), ACT360);
  let swaps = [(2, 0.038), (5, 0.04), (10, 0.042)].map(|(y, r)| {
    SwapRateHelper::new(
      quote(r),
      valuation,
      d(2025 + y, 1, 1),
      Frequency::SemiAnnual,
      AXIS,
    )
  });
  let mut helpers: Vec<&dyn RateHelper<f64>> = vec![&deposit, &fra];
  helpers.extend(swaps.iter().map(|h| h as &dyn RateHelper<f64>));
  for method in METHODS {
    let curve = build_curve(&helpers, valuation, AXIS, method).unwrap();
    let fra_rate = forward(&curve, valuation, d(2025, 7, 1), d(2026, 1, 1));
    let deposit_rate = forward(&curve, valuation, valuation, d(2025, 7, 1));
    assert!(
      (fra_rate - 0.035).abs() < 1e-14,
      "{method:?}: FRA {fra_rate}"
    );
    assert!(
      (deposit_rate - 0.03).abs() < 1e-14,
      "{method:?}: deposit {deposit_rate}"
    );
  }
}

/// Each swap helper's curve gives its own dated swap, fixed leg on its day count and ACT/360
/// floating leg, its quote as the fair rate; 40y and 50y quarterly and 20y monthly included.
#[test]
fn a_swap_helpers_curve_reprices_the_dated_swap_it_quotes() {
  let valuation = d(2025, 1, 1);
  let deposit = DepositRateHelper::new(quote(0.03), valuation, d(2025, 7, 1), ACT360);
  let index = FloatingIndex::Ibor(IborIndex::new("X-6M", RateTenor::SixMonths, ACT360));
  for fixed_day_count in [AXIS, DayCountConvention::Thirty360] {
    let swaps = [
      (2, 0.038, Frequency::SemiAnnual),
      (5, 0.04, Frequency::SemiAnnual),
      (10, 0.042, Frequency::SemiAnnual),
      (20, 0.0425, Frequency::Monthly),
      (30, 0.043, Frequency::SemiAnnual),
      (40, 0.0435, Frequency::Quarterly),
      (50, 0.044, Frequency::Quarterly),
    ]
    .map(|(years, rate, frequency)| {
      let maturity = d(2025 + years, 1, 1);
      let helper =
        SwapRateHelper::new(quote(rate), valuation, maturity, frequency, fixed_day_count);
      (maturity, rate, frequency, helper)
    });
    let mut helpers: Vec<&dyn RateHelper<f64>> = vec![&deposit];
    helpers.extend(swaps.iter().map(|(.., h)| h as &dyn RateHelper<f64>));
    for method in METHODS {
      let curve = build_curve(&helpers, valuation, AXIS, method).unwrap();
      for (maturity, rate, frequency, _) in &swaps {
        let schedule = |frequency| {
          ScheduleBuilder::new(valuation, *maturity)
            .frequency(frequency)
            .build()
        };
        let swap = VanillaInterestRateSwap::new(
          SwaptionDirection::Payer,
          &schedule(*frequency),
          &schedule(Frequency::SemiAnnual),
          1.0,
          *rate,
          fixed_day_count,
          index.clone(),
          0.0,
          ACT360,
        );
        let fair = swap.valuation(valuation, AXIS, &curve).fair_rate;
        assert!(
          (fair - rate).abs() < 1e-13,
          "{fixed_day_count:?} {method:?} {maturity}: {fair} against {rate}"
        );
      }
    }
  }
}

/// A futures helper's convexity-adjusted rate, `σ²·t₁·t₂/2` on the axis' times, is the ACT/360
/// forward its period reads on the ACT/365F axis.
#[test]
fn a_futures_helper_reprices_its_adjusted_forward_on_the_axis() {
  let valuation = d(2025, 1, 1);
  let (start, end) = (d(2025, 3, 19), d(2025, 6, 18));
  let deposit = DepositRateHelper::new(quote(0.03), valuation, start, ACT360);
  let future = FuturesRateHelper::new(quote(96.5), start, end, ACT360, 0.01);
  let helpers: [&dyn RateHelper<f64>; 2] = [&deposit, &future];
  let (t1, t2) = (
    AXIS.year_fraction::<f64>(valuation, start),
    AXIS.year_fraction::<f64>(valuation, end),
  );
  let adjusted = 0.035 - 0.5 * 0.01 * 0.01 * t1 * t2;
  for method in METHODS {
    let curve = build_curve(&helpers, valuation, AXIS, method).unwrap();
    let rate = forward(&curve, valuation, start, end);
    assert!(
      (rate - adjusted).abs() < 1e-14,
      "{method:?}: {rate} against {adjusted}"
    );
  }
}
