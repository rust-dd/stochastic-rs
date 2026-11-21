//! The error policy at its edge cases: NaN for an evaluation outside its domain, a named panic for a
//! broken precondition, never a plausible sentinel.

use chrono::NaiveDate;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::array;
use stochastic_rs_quant::OptionQuote;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::QuantError;
use stochastic_rs_quant::calendar::DayCountConvention;
use stochastic_rs_quant::calendar::Frequency;
use stochastic_rs_quant::credit::CdsPosition;
use stochastic_rs_quant::credit::CdsQuote;
use stochastic_rs_quant::credit::CreditDefaultSwap;
use stochastic_rs_quant::credit::bootstrap_hazard;
use stochastic_rs_quant::curves::Compounding;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::inflation::InflationCurve;
use stochastic_rs_quant::inflation::ZeroCouponInflationCurve;
use stochastic_rs_quant::microstructure::AlmgrenChrissParams;
use stochastic_rs_quant::order_book::Side;
use stochastic_rs_quant::portfolio::momentum::WeightScheme;
use stochastic_rs_quant::portfolio::optimizers::empirical_cvar;
use stochastic_rs_quant::portfolio::types::OptimizerMethod;
use stochastic_rs_quant::pricing::sabr::SabrParams;
use stochastic_rs_quant::pricing::sabr::hagan_implied_vol;
use stochastic_rs_quant::risk::DrawdownStats;
use stochastic_rs_quant::risk::PnlOrLoss;
use stochastic_rs_quant::risk::VarMethod;
use stochastic_rs_quant::risk::calmar_ratio;
use stochastic_rs_quant::risk::gaussian_es;
use stochastic_rs_quant::risk::gaussian_var;
use stochastic_rs_quant::risk::historical_es;
use stochastic_rs_quant::risk::historical_var;
use stochastic_rs_quant::risk::information_ratio;
use stochastic_rs_quant::risk::liquidity_adjusted_var;
use stochastic_rs_quant::risk::max_drawdown;
use stochastic_rs_quant::risk::running_drawdown;
use stochastic_rs_quant::risk::sharpe_ratio;
use stochastic_rs_quant::risk::sortino_ratio;
use stochastic_rs_quant::risk::value_at_risk;
use stochastic_rs_quant::risk::xva::ExposureProfile;
use stochastic_rs_quant::vol_surface::implied::ImpliedVolSurface;

mod query_limits;

fn flat_curve() -> DiscountCurve<f64> {
  DiscountCurve::from_zero_rates(
    Array1::from_vec(vec![0.5, 1.0, 5.0]).view(),
    Array1::from_vec(vec![0.03; 3]).view(),
    InterpolationMethod::LogLinearOnDiscountFactors,
  )
}

#[test]
fn a_yield_at_a_non_positive_maturity_is_nan() {
  assert!(Compounding::Continuous.zero_rate(0.97_f64, 0.0).is_nan());
  assert!(Compounding::Simple.zero_rate(0.97_f64, -1.0).is_nan());
  assert!(flat_curve().zero_rate(0.0).is_nan());
  assert!((flat_curve().zero_rate(1.0) - 0.03).abs() < 1e-12);
}

#[test]
fn a_breakeven_rate_at_a_non_positive_maturity_is_nan() {
  let curve = ZeroCouponInflationCurve::<f64>::new(
    Array1::from_vec(vec![1.0, 5.0]),
    Array1::from_vec(vec![0.02, 0.025]),
  );
  assert!(curve.breakeven_rate(0.0).is_nan());
  assert!(curve.breakeven_rate(-1.0).is_nan());
  assert!((curve.breakeven_rate(5.0) - 0.025).abs() < 1e-12);
}

#[test]
fn a_query_outside_the_domain_is_nan_not_a_panic() {
  let p = SabrParams {
    alpha: 0.2,
    beta: 0.5,
    rho: -0.2,
    nu: 0.3,
  };
  assert!(hagan_implied_vol(-1.0, 100.0, 1.0, &p).is_nan());
  assert!(hagan_implied_vol(100.0, 0.0, 1.0, &p).is_nan());
  assert!(hagan_implied_vol(100.0, 100.0, 1.0, &p).is_finite());
  assert!(hagan_implied_vol(100.0, 100.0, -1.0, &p).is_nan());
  assert!(hagan_implied_vol(100.0, 100.0, f64::INFINITY, &p).is_nan());
  assert!(hagan_implied_vol(100.0, 100.0, 0.0, &p).is_finite());
}

#[test]
fn an_empty_sample_is_insufficient_data() {
  let err = empirical_cvar(&mut [], 0.05).unwrap_err();
  assert_eq!(err.to_string(), "empirical CVaR: needs at least 1, got 0");
}

#[test]
fn the_risk_estimators_report_short_or_bad_data_as_errors_and_bad_parameters_as_panics() {
  let empty = Array1::<f64>::zeros(0);
  let one = Array1::from_vec(vec![0.01]);
  let bad = Array1::from_vec(vec![0.01, f64::NAN, -0.02]);
  assert!(matches!(
    historical_var(empty.view(), 0.99, PnlOrLoss::Pnl).unwrap_err(),
    QuantError::InsufficientData {
      needed: 1,
      got: 0,
      ..
    }
  ));
  assert!(matches!(
    gaussian_var(one.view(), 0.99, PnlOrLoss::Pnl).unwrap_err(),
    QuantError::InsufficientData {
      needed: 2,
      got: 1,
      ..
    }
  ));
  assert!(matches!(
    value_at_risk(bad.view(), 0.99, PnlOrLoss::Pnl, VarMethod::Historical).unwrap_err(),
    QuantError::InvalidInput {
      what: "samples",
      ..
    }
  ));
  assert!(matches!(
    historical_es(empty.view(), 0.975, PnlOrLoss::Loss).unwrap_err(),
    QuantError::InsufficientData { .. }
  ));
  assert!(matches!(
    sharpe_ratio(one.view(), 0.0, 252.0).unwrap_err(),
    QuantError::InsufficientData { needed: 2, .. }
  ));
  assert!(matches!(
    empirical_cvar(&mut [], 0.05).unwrap_err(),
    QuantError::InsufficientData { .. }
  ));
  let ok = historical_var(
    Array1::from_vec(vec![-0.03_f64, 0.01, 0.02, -0.01, 0.0]).view(),
    0.8,
    PnlOrLoss::Pnl,
  )
  .unwrap();
  assert!(ok.is_finite());
  let panicked = std::panic::catch_unwind(|| historical_var(one.view(), 1.0, PnlOrLoss::Pnl));
  assert!(panicked.is_err(), "confidence is a parameter");
}

#[test]
fn a_non_finite_sample_is_invalid_input_at_its_index() {
  let bad = array![0.01, -0.02, f64::INFINITY, 0.03];
  for err in [
    gaussian_var(bad.view(), 0.99, PnlOrLoss::Pnl).unwrap_err(),
    gaussian_es(bad.view(), 0.99, PnlOrLoss::Loss).unwrap_err(),
    historical_es(bad.view(), 0.99, PnlOrLoss::Pnl).unwrap_err(),
    sortino_ratio(bad.view(), 0.0, 252.0).unwrap_err(),
    information_ratio(bad.view(), Array1::zeros(4).view(), 252.0).unwrap_err(),
    DrawdownStats::from_equity(bad.view()).unwrap_err(),
  ] {
    assert_eq!(err.to_string(), "invalid samples: sample 2 is inf");
  }
  let err = empirical_cvar(&mut [0.1, f64::NAN], 0.05).unwrap_err();
  assert_eq!(err.to_string(), "invalid samples: sample 1 is NaN");
}

#[test]
#[should_panic(
  expected = "confidence must satisfy `confidence > 0.0 && confidence < 1.0`, got confidence = 1.0"
)]
fn a_confidence_of_one_is_a_precondition() {
  let _ = gaussian_var(array![0.01, -0.02].view(), 1.0, PnlOrLoss::Pnl);
}

#[test]
fn a_ratio_with_a_zero_denominator_is_nan() {
  let flat = array![0.01_f64, 0.01, 0.01, 0.01];
  let tenth = array![0.1_f64, 0.1, 0.1];
  let up = array![0.01_f64, 0.02, 0.03, 0.04];
  assert!(sharpe_ratio(flat.view(), 0.0, 252.0).unwrap().is_nan());
  assert!(sharpe_ratio(tenth.view(), 0.0, 252.0).unwrap().is_nan());
  let zero = Array1::zeros(3);
  assert!(
    information_ratio(tenth.view(), zero.view(), 252.0)
      .unwrap()
      .is_nan()
  );
  assert!(sortino_ratio(up.view(), 0.0, 252.0).unwrap().is_nan());
  assert!(
    information_ratio(up.view(), up.view(), 252.0)
      .unwrap()
      .is_nan()
  );
  assert!(calmar_ratio(up.view(), 252.0).unwrap().is_nan());
  assert!(matches!(
    information_ratio(up.view(), array![0.0, 0.0].view(), 252.0).unwrap_err(),
    QuantError::DimensionMismatch {
      expected: 4,
      got: 2,
      ..
    }
  ));
  assert!(matches!(
    calmar_ratio(Array1::<f64>::zeros(0).view(), 252.0).unwrap_err(),
    QuantError::InsufficientData {
      needed: 1,
      got: 0,
      ..
    }
  ));
}

#[test]
fn an_empty_equity_series_has_no_drawdown() {
  let empty = Array1::<f64>::zeros(0);
  assert!(max_drawdown(empty.view()).is_nan());
  assert!(max_drawdown(array![100.0, f64::NAN, 90.0].view()).is_nan());
  let err = DrawdownStats::from_equity(empty.view()).unwrap_err();
  assert_eq!(
    err.to_string(),
    "drawdown statistics: needs at least 1, got 0"
  );
}

#[test]
fn a_drawdown_from_a_non_positive_peak_is_nan() {
  assert!(max_drawdown(array![f64::NAN, 100.0, 90.0].view()).is_nan());
  assert!(max_drawdown(array![-10.0_f64, -20.0].view()).is_nan());
  let running = running_drawdown(array![-1.0_f64, 2.0, 1.0].view());
  assert!(
    running[0].is_nan() && running[1] == 0.0 && running[2] == -0.5,
    "{running}"
  );
  let err = DrawdownStats::from_equity(array![-10.0, -20.0].view()).unwrap_err();
  assert_eq!(
    err.to_string(),
    "invalid samples: sample 0 is -10.0, not positive"
  );
}

#[test]
fn finite_samples_whose_moments_overflow_or_underflow_are_a_numerical_error() {
  let big = array![1e200, -1e200, 1e200];
  let tiny = array![1e-200, 2e-200, 1e-200];
  let mean_overflows = array![1e308, 1e308, -0.5];
  let mut equity_overflows = vec![1.0; 1100];
  equity_overflows.push(-0.5);
  let equity_overflows = Array1::from_vec(equity_overflows);
  let results = [
    gaussian_var(array![1e308, 1e308, 1e308].view(), 0.99, PnlOrLoss::Pnl),
    gaussian_var(array![1e200, -1e200].view(), 0.99, PnlOrLoss::Pnl),
    gaussian_es(array![1e200, -1e200].view(), 0.99, PnlOrLoss::Loss),
    sharpe_ratio(big.view(), 0.0, 252.0),
    sharpe_ratio(tiny.view(), 0.0, 252.0),
    sortino_ratio(big.view(), 0.0, 252.0),
    sortino_ratio(tiny.view(), 1.5e-200, 252.0),
    sortino_ratio(mean_overflows.view(), 0.0, 252.0),
    information_ratio(big.view(), Array1::zeros(3).view(), 252.0),
    calmar_ratio(mean_overflows.view(), 252.0),
    calmar_ratio(equity_overflows.view(), 252.0),
  ];
  for result in results {
    assert!(
      matches!(result, Err(QuantError::Numerical { .. })),
      "{result:?}"
    );
  }
  let err = gaussian_var(array![1e308, 1e308].view(), 0.99, PnlOrLoss::Pnl).unwrap_err();
  assert_eq!(
    err.to_string(),
    "gaussian VaR: the sum of finite samples overflows to -inf"
  );
  let err = sharpe_ratio(tiny.view(), 0.0, 252.0).unwrap_err();
  assert_eq!(
    err.to_string(),
    "Sharpe ratio: the deviation of finite samples underflows to 0.0"
  );
}

#[test]
fn liquidity_adjusted_var_reports_the_var_error() {
  let execution = AlmgrenChrissParams::<f64>::new(Side::Sell, 1e4, 1.0, 10, 0.02, 1e-6, 1e-5, 1e-6);
  let err = liquidity_adjusted_var(
    Array1::zeros(0).view(),
    0.99,
    PnlOrLoss::Pnl,
    VarMethod::Historical,
    &execution,
  )
  .unwrap_err();
  assert_eq!(err.to_string(), "historical VaR: needs at least 1, got 0");
}

#[test]
#[should_panic(expected = "strikes must satisfy `strikes.windows(2).all(|w| w[0] < w[1])`")]
fn an_unsorted_strike_grid_is_rejected() {
  let ivs = Array2::from_elem((1, 3), 0.2);
  let _ = ImpliedVolSurface::from_iv_grid(vec![90.0, 110.0, 100.0], vec![1.0], vec![100.0], ivs);
}

#[test]
#[should_panic(expected = "maturities must satisfy `maturities.windows(2).all(|w| w[0] < w[1])`")]
fn an_unsorted_maturity_grid_is_rejected() {
  let ivs = Array2::from_elem((2, 2), 0.2);
  let _ = ImpliedVolSurface::from_iv_grid(vec![90.0, 110.0], vec![1.0, 0.5], vec![100.0; 2], ivs);
}

#[test]
#[should_panic(expected = "strikes must satisfy `strikes.windows(2).all(|w| w[0] < w[1])`")]
fn a_price_grid_shares_the_ascending_axes_check() {
  let prices = Array2::from_elem((1, 2), 5.0);
  let _ = ImpliedVolSurface::from_prices(
    vec![110.0, 90.0],
    vec![1.0],
    vec![100.0],
    prices.view(),
    OptionType::Call,
  );
}

#[test]
fn quant_errors_are_one_type_with_a_readable_display() {
  fn assert_send_sync<T: Send + Sync + std::error::Error + Clone>() {}
  assert_send_sync::<QuantError>();
  let e = QuantError::InvalidQuote {
    index: 3,
    reason: "strike is NaN".to_string(),
  };
  assert_eq!(e.to_string(), "quote 3 cannot be used: strike is NaN");
  let e = QuantError::InsufficientData {
    what: "prices for the optimal cutting frequency",
    needed: 8,
    got: 5,
  };
  assert_eq!(
    e.to_string(),
    "prices for the optimal cutting frequency: needs at least 8, got 5"
  );
}

#[test]
fn an_unknown_method_string_is_invalid_input_listing_the_accepted_ones() {
  let e = "nope".parse::<OptimizerMethod>().unwrap_err();
  assert!(matches!(
    e,
    QuantError::InvalidInput {
      what: "optimizer method",
      ..
    }
  ));
  assert!(e.to_string().contains("'nope'") && e.to_string().contains("black-litterman"));
  let e = "nope".parse::<WeightScheme>().unwrap_err();
  assert!(matches!(
    e,
    QuantError::InvalidInput {
      what: "weight scheme",
      ..
    }
  ));
  assert!(e.to_string().contains("score-weighted"));
}

#[test]
#[should_panic(expected = "MtM columns, one per exposure date: expected 3, got 2")]
fn a_panicking_twin_panics_with_the_display_of_its_error() {
  ExposureProfile::from_mtm(Array2::zeros((2, 2)).view(), vec![0.5, 1.0, 1.5], 0.95);
}

#[test]
fn a_nan_quote_is_an_error_not_a_panic() {
  let quotes = [
    OptionQuote::new(100.0, 1.0, 8.0, OptionType::Call),
    OptionQuote::new(f64::NAN, 1.0, 8.0, OptionType::Call),
  ];
  let err = ImpliedVolSurface::try_from_quotes(&quotes, &[(1.0, 102.0)], 0.0).unwrap_err();
  assert!(
    matches!(err, QuantError::InvalidQuote { index: 1, .. }),
    "{err}"
  );
  let missing = ImpliedVolSurface::try_from_quotes(&quotes[..1], &[(0.5, 101.0)], 0.0).unwrap_err();
  assert!(
    matches!(
      missing,
      QuantError::MissingInput {
        what: "forward",
        ..
      }
    ),
    "{missing}"
  );
}

#[test]
fn no_quote_is_an_insufficient_data_error() {
  let empty = ImpliedVolSurface::try_from_quotes(&[], &[], 0.0);
  assert!(matches!(
    empty,
    Err(QuantError::InsufficientData { got: 0, .. })
  ));
  let nan_rate = ImpliedVolSurface::try_from_quotes(
    &[OptionQuote::new(100.0, 1.0, 8.0, OptionType::Call)],
    &[(1.0, 102.0)],
    f64::NAN,
  );
  assert!(matches!(
    nan_rate,
    Err(QuantError::InvalidInput { what: "r", .. })
  ));
}

fn date(y: i32, m: u32, day: u32) -> NaiveDate {
  NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

#[test]
fn a_cds_quote_outside_its_domain_is_an_invalid_quote_at_its_index_not_a_panic() {
  let (now, year_ago) = (date(2026, 6, 20), date(2025, 6, 20));
  let one_year = |spread, recovery_rate| CdsQuote {
    recovery_rate,
    ..CdsQuote::isda(date(2027, 6, 20), spread)
  };
  let at = |maturity| CdsQuote::isda(maturity, 0.01);
  let cases = [
    (now, one_year(f64::NAN, None), "spread = NaN"),
    (now, one_year(0.01, Some(f64::NAN)), "recovery_rate = NaN"),
    (now, one_year(0.01, Some(1.0)), "recovery_rate = 1.0"),
    (now, one_year(0.01, Some(-0.2)), "recovery_rate = -0.2"),
    (now, at(now), "maturity = 2026-06-20"),
    (now, at(year_ago), "maturity = 2025-06-20"),
    (year_ago, at(date(2026, 1, 20)), "maturity = 2026-01-20"),
  ];
  for (effective, bad, got) in cases {
    let quotes = [CdsQuote::isda(date(2031, 6, 20), 0.01), bad];
    let dcc = DayCountConvention::Actual365Fixed;
    let err = bootstrap_hazard(now, effective, &quotes, 0.4, &flat_curve(), dcc).unwrap_err();
    assert!(
      matches!(&err, QuantError::InvalidQuote { index: 1, reason } if reason.contains(&format!("got {got}"))),
      "{err}"
    );
  }
}

#[test]
fn a_recovery_that_rounds_to_one_in_the_curves_float_is_an_invalid_quote() {
  let curve = DiscountCurve::<f32>::from_zero_rates(
    Array1::from_vec(vec![0.5, 1.0, 5.0]).view(),
    Array1::from_vec(vec![0.03; 3]).view(),
    InterpolationMethod::LogLinearOnDiscountFactors,
  );
  // Below 1 as an f64, 1.0 as the f32 the contract holds.
  let quotes = [CdsQuote {
    recovery_rate: Some(0.999_999_99),
    ..CdsQuote::isda(date(2031, 6, 20), 0.01)
  }];
  let (valuation, dcc) = (date(2026, 6, 20), DayCountConvention::Actual365Fixed);
  let err = bootstrap_hazard(valuation, valuation, &quotes, 0.4_f32, &curve, dcc).unwrap_err();
  assert!(
    matches!(&err, QuantError::InvalidQuote { index: 0, reason } if reason.ends_with("got recovery_rate = 1.0")),
    "{err}"
  );
}

#[test]
#[should_panic(
  expected = "default_recovery must satisfy `default_recovery >= T::zero() && default_recovery < T::one()`, got default_recovery = 1.0"
)]
fn a_default_recovery_of_one_is_a_precondition() {
  let quotes = [CdsQuote::isda(date(2031, 6, 20), 0.01)];
  let (valuation, dcc) = (date(2026, 6, 20), DayCountConvention::Actual365Fixed);
  let _ = bootstrap_hazard(valuation, valuation, &quotes, 1.0, &flat_curve(), dcc);
}

fn vanilla_cds(recovery: f64, effective: NaiveDate, maturity: NaiveDate) -> CreditDefaultSwap<f64> {
  let (frequency, dcc) = (Frequency::Quarterly, DayCountConvention::Actual360);
  CreditDefaultSwap::vanilla(
    CdsPosition::Buyer,
    1.0,
    0.01,
    recovery,
    effective,
    maturity,
    frequency,
    dcc,
  )
}

#[test]
#[should_panic(
  expected = "recovery_rate must satisfy `recovery_rate >= T::zero() && recovery_rate < T::one()`, got recovery_rate = 1.0"
)]
fn a_cds_recovery_of_one_is_a_precondition() {
  vanilla_cds(1.0, date(2026, 6, 20), date(2031, 6, 20));
}

#[test]
#[should_panic(
  expected = "premium_schedule must satisfy `premium_schedule.adjusted_dates.len() >= 2`, got premium_schedule.adjusted_dates.len() = 1"
)]
fn a_cds_without_a_premium_period_is_a_precondition() {
  vanilla_cds(0.4, date(2026, 6, 20), date(2026, 6, 20));
}

#[test]
#[should_panic(
  expected = "effective_date must satisfy `effective_date < maturity_date`, got effective_date = 2031-06-20, maturity_date = 2026-06-20"
)]
fn a_cds_maturing_before_it_starts_is_a_precondition() {
  vanilla_cds(0.4, date(2031, 6, 20), date(2026, 6, 20));
}
