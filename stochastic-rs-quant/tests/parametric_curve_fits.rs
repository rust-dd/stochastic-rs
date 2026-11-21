//! `NelsonSiegel::fit` and `Svensson::fit` recover a curve their own model generated, with its
//! decay parameters on the search grid, and refuse quotes that cannot identify one.

use ndarray::Array1;
use ndarray::array;
use ndarray::s;
use stochastic_rs_quant::QuantError;
use stochastic_rs_quant::curves::NelsonSiegel;
use stochastic_rs_quant::curves::Svensson;

fn maturities() -> Array1<f64> {
  array![0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
}

#[test]
fn nelson_siegel_fit_recovers_its_generating_parameters() {
  let truth = NelsonSiegel::new(0.06, -0.02, 0.01, 1.5);
  let taus = maturities();
  let rates = taus.mapv(|tau| truth.zero_rate(tau));
  let fit = NelsonSiegel::fit(taus.view(), rates.view()).unwrap();
  for (&tau, &rate) in taus.iter().zip(&rates) {
    assert!((fit.zero_rate(tau) - rate).abs() < 1e-12, "tau = {tau}");
  }
  assert!((fit.lambda - truth.lambda).abs() < 1e-12, "{}", fit.lambda);
  for (got, want) in [
    (fit.beta0, truth.beta0),
    (fit.beta1, truth.beta1),
    (fit.beta2, truth.beta2),
  ] {
    assert!((got - want).abs() < 1e-12, "{got} against {want}");
  }
}

#[test]
fn svensson_fit_recovers_its_generating_parameters() {
  let truth = Svensson::new(0.05, -0.02, 0.01, 0.015, 1.5, 3.0);
  let taus = maturities();
  let rates = taus.mapv(|tau| truth.zero_rate(tau));
  let fit = Svensson::fit(taus.view(), rates.view()).unwrap();
  for (&tau, &rate) in taus.iter().zip(&rates) {
    assert!((fit.zero_rate(tau) - rate).abs() < 1e-12, "tau = {tau}");
  }
  assert!(
    (fit.lambda1 - truth.lambda1).abs() < 1e-12,
    "{}",
    fit.lambda1
  );
  assert!(
    (fit.lambda2 - truth.lambda2).abs() < 1e-12,
    "{}",
    fit.lambda2
  );
  for (got, want) in [
    (fit.beta0, truth.beta0),
    (fit.beta1, truth.beta1),
    (fit.beta2, truth.beta2),
    (fit.beta3, truth.beta3),
  ] {
    assert!((got - want).abs() < 1e-12, "{got} against {want}");
  }
}

#[test]
fn svensson_fit_refuses_quotes_that_cannot_identify_its_curve() {
  let truth = Svensson::new(0.05, -0.02, 0.01, 0.015, 1.5, 3.0);
  let taus = maturities();
  let rates = taus.mapv(|tau| truth.zero_rate(tau));
  let mismatch = QuantError::DimensionMismatch {
    what: "market rates, one per maturity",
    expected: 11,
    got: 10,
  };
  assert_eq!(
    Svensson::fit(taus.view(), rates.slice(s![..10])).unwrap_err(),
    mismatch
  );
  let short = QuantError::InsufficientData {
    what: "quotes for a Svensson fit",
    needed: 6,
    got: 5,
  };
  assert_eq!(
    Svensson::fit(taus.slice(s![..5]), rates.slice(s![..5])).unwrap_err(),
    short
  );
  let mut nan_rate = rates.clone();
  nan_rate[4] = f64::NAN;
  let mut infinite = taus.clone();
  infinite[10] = f64::INFINITY;
  for (taus, rates, index) in [
    (taus.view(), nan_rate.view(), 4),
    (infinite.view(), rates.view(), 10),
  ] {
    let refused = Svensson::fit(taus, rates).unwrap_err();
    assert!(
      matches!(refused, QuantError::InvalidQuote { index: i, .. } if i == index),
      "{refused}"
    );
  }
  let negated = taus.mapv(|tau| -tau);
  let refused = Svensson::fit(negated.view(), rates.view()).unwrap_err();
  assert!(
    matches!(refused, QuantError::InvalidQuote { index: 0, .. }),
    "{refused}"
  );
  let at_origin = Array1::<f64>::zeros(6);
  let unsolvable = Svensson::fit(at_origin.view(), rates.slice(s![..6])).unwrap_err();
  assert!(
    matches!(unsolvable, QuantError::Numerical { .. }),
    "{unsolvable}"
  );
}

#[test]
fn nelson_siegel_fit_refuses_quotes_that_cannot_identify_its_curve() {
  let truth = NelsonSiegel::new(0.06, -0.02, 0.01, 1.5);
  let taus = maturities();
  let rates = taus.mapv(|tau| truth.zero_rate(tau));
  let mismatch = QuantError::DimensionMismatch {
    what: "market rates, one per maturity",
    expected: 11,
    got: 10,
  };
  assert_eq!(
    NelsonSiegel::fit(taus.view(), rates.slice(s![..10])).unwrap_err(),
    mismatch
  );
  let short = QuantError::InsufficientData {
    what: "quotes for a Nelson-Siegel fit",
    needed: 4,
    got: 3,
  };
  assert_eq!(
    NelsonSiegel::fit(taus.slice(s![..3]), rates.slice(s![..3])).unwrap_err(),
    short
  );
  let mut nan_rate = rates.clone();
  nan_rate[4] = f64::NAN;
  let refused = NelsonSiegel::fit(taus.view(), nan_rate.view()).unwrap_err();
  assert!(
    matches!(refused, QuantError::InvalidQuote { index: 4, .. }),
    "{refused}"
  );
  let negated = taus.mapv(|tau| -tau);
  let refused = NelsonSiegel::fit(negated.view(), rates.view()).unwrap_err();
  assert!(
    matches!(refused, QuantError::InvalidQuote { index: 0, .. }),
    "{refused}"
  );
  let at_origin = Array1::<f64>::zeros(4);
  let unsolvable = NelsonSiegel::fit(at_origin.view(), rates.slice(s![..4])).unwrap_err();
  assert!(
    matches!(unsolvable, QuantError::Numerical { .. }),
    "{unsolvable}"
  );
}
