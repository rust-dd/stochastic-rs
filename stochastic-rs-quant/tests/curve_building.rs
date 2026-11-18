//! Curves and the instruments read off them: tenor-keyed forecast curves, `npv` beside `valuation`,
//! one bootstrap entry and the flat zero rate below a curve's first pillar.

use chrono::NaiveDate;
use ndarray::Array1;
use ndarray::array;
use stochastic_rs_quant::QuantError;
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
use stochastic_rs_quant::curves::BootstrapInstrument;
use stochastic_rs_quant::curves::BootstrapSpec;
use stochastic_rs_quant::curves::CurvePoint;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::curves::IterativeOptions;
use stochastic_rs_quant::curves::MultiCurve;
use stochastic_rs_quant::curves::bootstrap;
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
    array![0.5, 1.0, 5.0, 10.0].view(),
    array![rate, rate, rate, rate].view(),
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

fn deposit(maturity: f64, rate: f64) -> BootstrapInstrument<f64> {
  BootstrapInstrument::Deposit { maturity, rate }
}

fn schedule_swap(rate: f64, fixed: &[f64], float: &[f64]) -> BootstrapInstrument<f64> {
  BootstrapInstrument::SwapWithSchedule {
    rate,
    fixed_times: fixed.to_vec(),
    float_times: float.to_vec(),
  }
}

fn quarters(n: usize) -> Vec<f64> {
  (1..=n).map(|i| 0.25 * i as f64).collect()
}

#[test]
fn one_bootstrap_entry_and_a_flat_zero_short_end() {
  let instruments = [
    deposit(0.5, 0.03),
    BootstrapInstrument::Swap {
      maturity: 2.0,
      rate: 0.032,
      frequency: Frequency::SemiAnnual,
    },
    BootstrapInstrument::Swap {
      maturity: 5.0,
      rate: 0.035,
      frequency: Frequency::SemiAnnual,
    },
  ];
  let first_rate = -(1.0_f64 / 1.015).ln() / 0.5;
  for method in [
    InterpolationMethod::LinearOnZeroRates,
    InterpolationMethod::LogLinearOnDiscountFactors,
    InterpolationMethod::CubicSplineOnZeroRates,
    InterpolationMethod::MonotoneConvex,
  ] {
    let direct = bootstrap(BootstrapSpec::new(&instruments, method)).unwrap();
    assert_eq!(direct.len(), 3, "the origin is not a pillar");
    // Monotone convex shapes [0, t₁] itself; the other three hold the first zero rate there.
    let hagan_west = method == InterpolationMethod::MonotoneConvex;
    for t in [0.1, 0.25, 0.4] {
      let flat = (-first_rate * t).exp();
      let gap = (direct.discount_factor(t) - flat).abs();
      assert!(
        if hagan_west { gap > 1e-6 } else { gap < 1e-15 },
        "{method:?} at {t}: {}",
        direct.discount_factor(t)
      );
    }
  }
  let user = DiscountCurve::<f64>::from_zero_rates(
    array![0.5, 2.0].view(),
    array![0.03, 0.032].view(),
    InterpolationMethod::LinearOnZeroRates,
  );
  assert!(
    (user.discount_factor(0.1) - 0.997_004_495_503).abs() < 1e-12,
    "{}",
    user.discount_factor(0.1)
  );
  let ois = bootstrap(BootstrapSpec::new(
    &instruments,
    InterpolationMethod::LogLinearOnDiscountFactors,
  ))
  .unwrap();
  let tenor = [
    deposit(0.25, 0.034),
    schedule_swap(0.036, &[0.5, 1.0], &quarters(4)),
  ];
  let spec = BootstrapSpec::new(&tenor, InterpolationMethod::LogLinearOnDiscountFactors);
  let forecast = bootstrap(spec.discounting_with(&ois)).unwrap();
  assert!(
    forecast.discount_factor(1.0) < ois.discount_factor(1.0),
    "a 3M tenor curve above OIS discounts more"
  );
  assert!(matches!(
    bootstrap(BootstrapSpec::<f64>::new(
      &[],
      InterpolationMethod::LinearOnZeroRates
    ))
    .unwrap_err(),
    QuantError::InsufficientData {
      what: "instruments",
      needed: 1,
      got: 0
    }
  ));
}

#[test]
fn a_quote_that_gives_no_pillar_is_an_error_at_its_index() {
  let nan = f64::NAN;
  let semi = Frequency::SemiAnnual;
  let bad: Vec<(&str, BootstrapInstrument<f64>)> = vec![
    ("maturity", deposit(nan, 0.03)),
    ("maturity", deposit(0.0, 0.03)),
    ("maturity", deposit(0.5, 0.031)),
    ("rate", deposit(1.0, nan)),
    ("discount factor", deposit(1.0, -1.0)),
    (
      "start",
      BootstrapInstrument::Fra {
        start: nan,
        end: 1.0,
        rate: 0.03,
      },
    ),
    (
      "start",
      BootstrapInstrument::Fra {
        start: 1.0,
        end: 1.0,
        rate: 0.03,
      },
    ),
    (
      "rate",
      BootstrapInstrument::Fra {
        start: 0.5,
        end: 1.0,
        rate: nan,
      },
    ),
    (
      "start",
      BootstrapInstrument::Future {
        start: nan,
        end: 1.0,
        price: 97.0,
        sigma: 0.01,
      },
    ),
    (
      "price",
      BootstrapInstrument::Future {
        start: 0.5,
        end: 1.0,
        price: nan,
        sigma: 0.01,
      },
    ),
    (
      "sigma",
      BootstrapInstrument::Future {
        start: 0.5,
        end: 1.0,
        price: 97.0,
        sigma: nan,
      },
    ),
    (
      "rate",
      BootstrapInstrument::Swap {
        maturity: 2.0,
        rate: nan,
        frequency: semi,
      },
    ),
    (
      "maturity",
      BootstrapInstrument::Swap {
        maturity: 1.4,
        rate: 0.03,
        frequency: semi,
      },
    ),
    ("fixed_times", schedule_swap(0.03, &[], &[])),
    ("fixed_times", schedule_swap(0.03, &[1.0, 0.75, 2.0], &[])),
    ("fixed_times", schedule_swap(0.03, &[1.0, nan, 2.0], &[])),
    ("maturity", schedule_swap(0.03, &[1.0, nan], &[])),
  ];
  let method = InterpolationMethod::LinearOnZeroRates;
  for (field, instrument) in bad {
    let quotes = [deposit(0.5, 0.03), instrument];
    for spec in [
      BootstrapSpec::new(&quotes, method),
      BootstrapSpec::new(&quotes, method).iterative(IterativeOptions::default()),
    ] {
      match bootstrap(spec) {
        Err(QuantError::InvalidQuote { index: 1, reason }) => {
          assert!(reason.contains(field), "{:?}: {reason}", quotes[1])
        }
        other => panic!("{:?}: {other:?}", quotes[1]),
      }
    }
  }
  let lone = [BootstrapInstrument::Fra {
    start: 0.25,
    end: 0.5,
    rate: 0.03,
  }];
  assert!(matches!(
    bootstrap(BootstrapSpec::new(&lone, method)),
    Err(QuantError::InvalidQuote { index: 0, .. })
  ));

  let ois = flat(0.02);
  let dual_bad = [
    ("float_times", schedule_swap(0.03, &[1.0], &[])),
    (
      "float_times",
      schedule_swap(0.03, &[1.0, 2.0], &[0.5, 1.0, 1.5]),
    ),
    ("float_times", schedule_swap(0.03, &[1.0], &[0.5, nan, 1.0])),
    (
      "SwapWithSchedule",
      BootstrapInstrument::Swap {
        maturity: 1.0,
        rate: 0.03,
        frequency: semi,
      },
    ),
  ];
  for (field, instrument) in dual_bad {
    let quotes = [deposit(0.5, 0.03), instrument];
    match bootstrap(BootstrapSpec::new(&quotes, method).discounting_with(&ois)) {
      Err(QuantError::InvalidQuote { index: 1, reason }) => {
        assert!(reason.contains(field), "{:?}: {reason}", quotes[1])
      }
      other => panic!("{:?}: {other:?}", quotes[1]),
    }
  }
}

#[test]
fn a_solve_with_no_root_is_an_optimizer_error() {
  let method = InterpolationMethod::LogLinearOnDiscountFactors;
  let unbracketed = [deposit(0.5, 0.03), schedule_swap(-5.0, &[1.0], &[0.5, 1.0])];
  let ois = flat(0.02);
  assert!(matches!(
    bootstrap(BootstrapSpec::new(&unbracketed, method).discounting_with(&ois)),
    Err(QuantError::Optimizer {
      what: "bootstrap",
      ..
    })
  ));
  // A negative par rate puts the discount factor above one, outside the bisection's bracket.
  let negative = [
    deposit(0.5, -0.004),
    BootstrapInstrument::Swap {
      maturity: 1.0,
      rate: -0.005,
      frequency: Frequency::SemiAnnual,
    },
  ];
  let direct = bootstrap(BootstrapSpec::new(&negative, method)).unwrap();
  assert!(direct.discount_factor(1.0) > 1.0);
  assert!(matches!(
    bootstrap(BootstrapSpec::new(&negative, method).iterative(IterativeOptions::default())),
    Err(QuantError::Optimizer {
      what: "bootstrap",
      ..
    })
  ));
}

#[test]
fn a_read_below_the_first_pillar_moves_the_pillars_after_it_except_under_log_linear() {
  let quotes = [
    schedule_swap(0.03, &[0.5], &[]),
    schedule_swap(0.031, &quarters(4), &[]),
    schedule_swap(0.033, &quarters(8), &[]),
  ];
  let pillars = |method| {
    let curve = bootstrap(BootstrapSpec::new(&quotes, method)).unwrap();
    [0.25, 0.5, 1.0, 2.0].map(|t| curve.discount_factor(t))
  };
  // The 1Y and 2Y swaps pay at 0.25, before the 6M pillar.
  assert_eq!(
    pillars(InterpolationMethod::LinearOnZeroRates),
    [
      0.9925833339709303,
      0.9852216748768474,
      0.9695789361618652,
      0.9362657494680776
    ]
  );
  assert_eq!(
    pillars(InterpolationMethod::LogLinearOnDiscountFactors),
    [
      0.9925833339709303,
      0.9852216748768474,
      0.9695789361618652,
      0.9362668653346841
    ]
  );
}

#[test]
fn a_discount_curve_drops_a_pillar_at_the_origin() {
  let point = |time: f64, discount_factor: f64| CurvePoint {
    time,
    discount_factor,
  };
  let method = InterpolationMethod::LinearOnZeroRates;
  let with_origin = DiscountCurve::new(
    vec![point(2.0, 0.93), point(0.0, 1.0), point(0.5, 0.985)],
    method,
  );
  let without = DiscountCurve::new(vec![point(0.5, 0.985), point(2.0, 0.93)], method);
  assert_eq!(with_origin.len(), 2);
  assert_eq!(with_origin.points()[0].time, 0.5);
  for t in [0.0, 0.1, 0.5, 1.0, 3.0] {
    assert_eq!(
      with_origin.discount_factor(t),
      without.discount_factor(t),
      "{t}"
    );
  }
  assert_eq!(without.discount_factor(0.0), 1.0);
  assert!((without.zero_rate(0.1) - without.zero_rate(0.5)).abs() < 1e-15);
}

#[test]
#[should_panic(
  expected = "points must satisfy `a.time != b.time` for every pair, got time = 1.0 twice"
)]
fn a_discount_curve_needs_distinct_times() {
  let point = |discount_factor: f64| CurvePoint {
    time: 1.0,
    discount_factor,
  };
  DiscountCurve::new(
    vec![point(0.97), point(0.96)],
    InterpolationMethod::LinearOnZeroRates,
  );
}

#[test]
#[should_panic(
  expected = "points must satisfy `p.time >= T::zero()` for every p, got p.time = -1.0"
)]
fn a_discount_curve_needs_times_from_the_origin_on() {
  DiscountCurve::from_zero_rates(
    array![-1.0, 1.0].view(),
    array![0.03, 0.03].view(),
    InterpolationMethod::LinearOnZeroRates,
  );
}
