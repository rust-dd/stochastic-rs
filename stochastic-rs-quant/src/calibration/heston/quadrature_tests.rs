use super::HestonCalibrator;
use super::HestonParams;
use crate::OptionType;
use crate::pricing::heston::HestonPricer;
use crate::traits::PricerExt;

fn converged_call(params: &HestonParams, k: f64, tau: f64) -> f64 {
  HestonPricer::new(
    100.0,
    params.v0,
    k,
    0.05,
    Some(0.0),
    params.rho,
    params.kappa,
    params.theta,
    params.sigma,
    Some(0.0),
    Some(tau),
    None,
    None,
  )
  .calculate_call_put()
  .0
}

fn calibrator(params: &HestonParams, k: f64, tau: f64) -> HestonCalibrator {
  HestonCalibrator::new(
    Some(params.clone()),
    vec![0.0].into(),
    vec![100.0].into(),
    vec![k].into(),
    0.05,
    Some(0.0),
    tau,
    OptionType::Call,
    None,
    None,
    None,
    false,
  )
}

/// The Cui price integral must use the same converged tail as the standalone
/// Heston pricer for short maturities and low variances.
#[test]
fn cui_price_matches_converged_short_dated_references() {
  let standard = HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  };
  let low_variance = HestonParams {
    v0: 0.01,
    kappa: 1.5,
    theta: 0.01,
    sigma: 0.2,
    rho: -0.7,
  };
  let cases = [
    (&standard, 100.0, 0.005, 0.576581),
    (&standard, 105.0, 0.01, 0.002966),
    (&standard, 95.0, 0.01, 5.052617),
    (&standard, 100.0, 0.02, 1.177515),
    (&low_variance, 100.0, 0.03, 0.768268),
  ];

  for (params, k, tau, expected) in cases {
    let (cui, _) = calibrator(params, k, tau)
      .cui_price_and_grad_for_quote(params, 100.0, k, tau)
      .expect("Cui price and gradient should converge");
    let reference = converged_call(params, k, tau);
    assert!(
      (cui - expected).abs() < 2e-3,
      "Cui K={k}, τ={tau}: got {cui}, expected {expected}"
    );
    assert!(
      (cui - reference).abs() < 2e-3,
      "Cui K={k}, τ={tau}: got {cui}, Heston reference {reference}"
    );
  }
}

/// The analytic gradient must agree with finite differences of the independently
/// converged Heston pricer, including in the slowly decaying short-time regime.
#[test]
fn cui_gradient_matches_converged_finite_difference() {
  let params = HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  };
  let tau = 0.02;
  let (_, analytic) = calibrator(&params, 100.0, tau)
    .cui_price_and_grad_for_quote(&params, 100.0, 100.0, tau)
    .expect("Cui price and gradient should converge");
  let steps = [1e-5, 1e-4, 1e-5, 1e-5, 1e-5];

  for (component, step) in steps.into_iter().enumerate() {
    let mut plus = params.clone();
    let mut minus = params.clone();
    match component {
      0 => {
        plus.v0 += step;
        minus.v0 -= step;
      }
      1 => {
        plus.kappa += step;
        minus.kappa -= step;
      }
      2 => {
        plus.theta += step;
        minus.theta -= step;
      }
      3 => {
        plus.sigma += step;
        minus.sigma -= step;
      }
      4 => {
        plus.rho += step;
        minus.rho -= step;
      }
      _ => unreachable!(),
    }

    let numeric =
      (converged_call(&plus, 100.0, tau) - converged_call(&minus, 100.0, tau)) / (2.0 * step);
    let scaled_error = (analytic[component] - numeric).abs() / (1.0 + numeric.abs());
    assert!(
      scaled_error < 5e-3,
      "gradient {component}: analytic={}, numeric={numeric}, scaled error={scaled_error}",
      analytic[component]
    );
  }
}
