//! The error policy at its edge cases: NaN for an evaluation outside its domain, a named panic for a
//! broken precondition, never a plausible sentinel.

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_quant::OptionQuote;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::QuantError;
use stochastic_rs_quant::curves::Compounding;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::inflation::InflationCurve;
use stochastic_rs_quant::inflation::ZeroCouponInflationCurve;
use stochastic_rs_quant::portfolio::momentum::WeightScheme;
use stochastic_rs_quant::portfolio::optimizers::empirical_cvar;
use stochastic_rs_quant::portfolio::types::OptimizerMethod;
use stochastic_rs_quant::pricing::sabr::SabrParams;
use stochastic_rs_quant::pricing::sabr::hagan_implied_vol;
use stochastic_rs_quant::risk::xva::ExposureProfile;
use stochastic_rs_quant::vol_surface::implied::ImpliedVolSurface;

fn flat_curve() -> DiscountCurve<f64> {
  DiscountCurve::from_zero_rates(
    &Array1::from_vec(vec![0.5, 1.0, 5.0]),
    &Array1::from_vec(vec![0.03; 3]),
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
#[should_panic(expected = "returns must satisfy `!returns.is_empty()`")]
fn an_empty_sample_is_a_precondition() {
  let mut empty: Vec<f64> = Vec::new();
  empirical_cvar(&mut empty, 0.05);
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
  let _ = ImpliedVolSurface::from_prices(vec![110.0, 90.0], vec![1.0], vec![100.0], &prices, true);
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
  ExposureProfile::from_mtm(&Array2::zeros((2, 2)), vec![0.5, 1.0, 1.5], 0.95);
}

#[test]
fn a_nan_quote_is_an_error_not_a_panic() {
  let quotes = [
    OptionQuote::new(100.0, 1.0, 8.0, OptionType::Call),
    OptionQuote::new(f64::NAN, 1.0, 8.0, OptionType::Call),
  ];
  let err = ImpliedVolSurface::try_from_quotes(&quotes, &[(1.0, 102.0)]).unwrap_err();
  assert!(
    matches!(err, QuantError::InvalidQuote { index: 1, .. }),
    "{err}"
  );
  let missing = ImpliedVolSurface::try_from_quotes(&quotes[..1], &[(0.5, 101.0)]).unwrap_err();
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
fn no_quote_is_an_error_and_a_put_quote_inverts_as_a_put() {
  use stochastic_rs_quant::distributions::special::norm_cdf;

  let empty = ImpliedVolSurface::try_from_quotes(&[], &[]);
  assert!(matches!(
    empty,
    Err(QuantError::InsufficientData { got: 0, .. })
  ));
  let (f, k, t, sigma) = (100.0_f64, 110.0_f64, 0.5_f64, 0.25_f64);
  let d1 = ((f / k).ln() + 0.5 * sigma * sigma * t) / (sigma * t.sqrt());
  let put = k * norm_cdf(-(d1 - sigma * t.sqrt())) - f * norm_cdf(-d1);
  let quote = OptionQuote::new(k, t, put, OptionType::Put);
  let surface = ImpliedVolSurface::try_from_quotes(&[quote], &[(t, f)]).unwrap();
  assert!((surface.ivs[[0, 0]] - sigma).abs() < 1e-10);
}
