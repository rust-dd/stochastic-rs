//! `NelsonSiegel::fit` and `Svensson::fit` recover a curve their own model generated, with its
//! decay parameters on the search grid.

use ndarray::Array1;
use ndarray::array;
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
  let fit = NelsonSiegel::fit(taus.view(), rates.view());
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
  let fit = Svensson::fit(taus.view(), rates.view());
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
