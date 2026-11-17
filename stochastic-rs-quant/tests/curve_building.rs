//! Curves and the instruments read off them: tenor-keyed forecast curves, `npv` beside `valuation`.

use chrono::NaiveDate;
use ndarray::Array1;
use ndarray::array;
use stochastic_rs_quant::calendar::DayCountConvention;
use stochastic_rs_quant::calendar::Frequency;
use stochastic_rs_quant::cashflows::CurveProvider;
use stochastic_rs_quant::cashflows::IborIndex;
use stochastic_rs_quant::cashflows::RateIndex;
use stochastic_rs_quant::cashflows::RateTenor;
use stochastic_rs_quant::credit::CdoTranche;
use stochastic_rs_quant::credit::CdsIndex;
use stochastic_rs_quant::credit::PoolName;
use stochastic_rs_quant::credit::index::flat_survival;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::curves::MultiCurve;
use stochastic_rs_quant::inflation::YearOnYearInflationSwap;
use stochastic_rs_quant::inflation::ZeroCouponInflationCurve;
use stochastic_rs_quant::inflation::ZeroCouponInflationSwap;
use stochastic_rs_quant::instruments::BlackVolatility;
use stochastic_rs_quant::instruments::CmsCaplet;
use stochastic_rs_quant::instruments::CmsFloorlet;
use stochastic_rs_quant::instruments::Deposit;
use stochastic_rs_quant::instruments::ForwardRateAgreement;
use stochastic_rs_quant::instruments::fra::FraPosition;
use stochastic_rs_quant::instruments::money_market::DepositValuation;
use stochastic_rs_quant::instruments::option::cms::hagan_linear_tsr_convexity_adjustment;

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
  NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

fn flat(rate: f64) -> DiscountCurve<f64> {
  DiscountCurve::from_zero_rates(
    &array![0.5, 1.0, 5.0, 10.0],
    &array![rate, rate, rate, rate],
    InterpolationMethod::LogLinearOnDiscountFactors,
  )
}

#[test]
fn forecast_curves_are_keyed_by_tenor() {
  let mut mc = MultiCurve::new(flat(0.02));
  mc.add_forecast(RateTenor::ThreeMonths, flat(0.025));
  assert!(mc.forecast(RateTenor::ThreeMonths).is_some());
  assert!(mc.forecast(RateTenor::SixMonths).is_none());
  let discount_at_one = mc.discount.discount_factor(1.0);
  let fallback = mc.forecast_curve(RateTenor::SixMonths).unwrap();
  assert_eq!(fallback.discount_factor(1.0), discount_at_one);
  let basis = mc.basis_spread(RateTenor::ThreeMonths, 1.0, 1.25).unwrap();
  assert!((basis - 0.005).abs() < 2e-4, "{basis}");
  assert_eq!(mc.basis_spread(RateTenor::SixMonths, 1.0, 1.25), None);
  let index = IborIndex::<f64>::new(
    "EURIBOR3M",
    RateTenor::ThreeMonths,
    DayCountConvention::Actual360,
  );
  assert_eq!(index.tenor(), RateTenor::ThreeMonths);
}

#[test]
fn a_tenor_parses_from_the_string_it_prints() {
  for tenor in [
    RateTenor::Overnight,
    RateTenor::OneMonth,
    RateTenor::ThreeMonths,
    RateTenor::SixMonths,
    RateTenor::TwelveMonths,
  ] {
    assert_eq!(tenor.to_string().parse::<RateTenor>().unwrap(), tenor);
  }
  assert_eq!("ON".parse::<RateTenor>().unwrap(), RateTenor::Overnight);
  assert_eq!("1Y".parse::<RateTenor>().unwrap(), RateTenor::TwelveMonths);
  let err = "2W".parse::<RateTenor>().unwrap_err().to_string();
  assert!(err.contains("2W"), "{err}");
}

#[test]
fn deposits_and_fras_are_instruments() {
  let curve = flat(0.03);
  let deposit = Deposit::new(
    1.0e6,
    0.035,
    d(2026, 1, 5),
    d(2026, 7, 6),
    DayCountConvention::Actual360,
  );
  let valuation: DepositValuation<f64> =
    deposit.valuation(d(2026, 1, 5), DayCountConvention::Actual365Fixed, &curve);
  assert_eq!(
    deposit.npv(d(2026, 1, 5), DayCountConvention::Actual365Fixed, &curve),
    valuation.npv
  );
  assert!(valuation.npv > 0.0, "a 3.5 % deposit on a 3 % curve");
  let index = IborIndex::<f64>::new(
    "EURIBOR6M",
    RateTenor::SixMonths,
    DayCountConvention::Actual360,
  );
  let fra = ForwardRateAgreement::new(
    FraPosition::Long,
    1.0e6,
    0.02,
    d(2026, 7, 6),
    d(2027, 1, 6),
    DayCountConvention::Actual360,
    index,
  );
  let v = fra.valuation(d(2026, 1, 5), DayCountConvention::Actual365Fixed, &curve);
  assert_eq!(
    fra.npv(d(2026, 1, 5), DayCountConvention::Actual365Fixed, &curve),
    v.npv
  );
  assert!(v.npv > 0.0, "long a 2 % strike on a 3 % curve");
}

#[test]
fn an_index_and_a_tranche_answer_npv_from_the_protection_buyers_side() {
  let curve = flat(0.03);
  let (start, end) = (d(2026, 3, 20), d(2031, 6, 20));
  let index = CdsIndex::homogeneous(vec![flat_survival(0.02); 5], 0.4, 0.01, 1.0e7, start, end);
  let v = index.valuation(start, &curve);
  assert_eq!(index.npv(start, &curve), v.net_npv);
  assert!(v.net_npv > 0.0, "a 1 % coupon under a 1.2 % fair spread");

  let pool: Vec<PoolName> = (0..20)
    .map(|_| PoolName {
      weight: 0.05,
      recovery: 0.4,
      survival: flat_survival(0.02),
    })
    .collect();
  let tranche = CdoTranche::new(0.03, 0.07, 0.01, vec![1.0, 2.0, 3.0, 4.0, 5.0], 1.0, 0.3);
  let tv = tranche.valuation(&pool, &curve);
  assert_eq!(
    tranche.npv(&pool, &curve),
    tv.protection_leg - tv.premium_leg
  );
  assert_eq!(tranche.npv(&pool, &curve), tv.upfront);
}

#[test]
fn the_inflation_swaps_answer_a_valuation_beside_npv() {
  let curve =
    ZeroCouponInflationCurve::<f64>::new(array![1.0, 3.0, 5.0], array![0.02, 0.025, 0.03]);
  let zc = ZeroCouponInflationSwap::<f64> {
    notional: 1.0e6,
    fixed_rate: 0.025,
    tau: 5.0,
  };
  let df = (-0.04_f64 * 5.0).exp();
  let v = zc.valuation(&curve, df);
  assert_eq!(v.npv, zc.npv(&curve, df));
  assert_eq!(v.fair_fixed_rate, zc.fair_fixed_rate(&curve));
  assert!((v.inflation_leg - v.fixed_leg - v.npv).abs() < 1e-9);
  assert!((v.inflation_leg - df * 1.0e6 * (1.03_f64.powf(5.0) - 1.0)).abs() < 1e-6);

  let times = Array1::from_iter((1..=5).map(f64::from));
  let dfs = times.mapv(|t| (-0.035 * t).exp());
  let yoy = YearOnYearInflationSwap::<f64>::new(100.0, 0.02, times, Array1::ones(5), dfs);
  let v = yoy.valuation(&curve);
  assert_eq!(v.inflation_leg, yoy.inflation_leg_pv(&curve));
  assert_eq!(v.fixed_leg, yoy.fixed_leg_pv());
  assert_eq!(v.npv, yoy.npv(&curve));
  assert_eq!(v.fair_fixed_rate, yoy.fair_fixed_rate(&curve));
}

#[test]
fn a_cms_caplet_answers_npv_and_a_valuation_with_its_convexity_adjustment() {
  let vol = BlackVolatility::new(0.3_f64);
  let caplet = CmsCaplet {
    strike: 0.03,
    notional: 1.0e6,
    accrual_factor: 1.0,
    discount_factor: 0.9,
    forward_cms: 0.04,
    t_fix: 2.0,
    swap_years: 10.0,
    fixed_freq: Frequency::SemiAnnual,
    payment_delay: 0.0,
    vol,
  };
  let v = caplet.valuation();
  assert_eq!(v.npv, caplet.npv());
  assert_eq!(v.forward_swap_rate, 0.04);
  assert_eq!(
    v.convexity_adjustment,
    hagan_linear_tsr_convexity_adjustment(0.04, 0.3, 2.0, 10.0, Frequency::SemiAnnual, 0.0)
  );
  assert!(v.convexity_adjustment != 0.0 && v.npv > 0.0, "{v:?}");
  let floorlet = CmsFloorlet {
    strike: 0.05,
    notional: 1.0e6,
    accrual_factor: 1.0,
    discount_factor: 0.9,
    forward_cms: 0.04,
    t_fix: 0.0,
    swap_years: 10.0,
    fixed_freq: Frequency::SemiAnnual,
    payment_delay: 0.0,
    vol,
  };
  let v = floorlet.valuation();
  assert_eq!(
    v.convexity_adjustment, 0.0,
    "a known fixing has no adjustment"
  );
  assert!((floorlet.npv() - 9000.0).abs() < 1e-9 && v.npv == floorlet.npv());
}
