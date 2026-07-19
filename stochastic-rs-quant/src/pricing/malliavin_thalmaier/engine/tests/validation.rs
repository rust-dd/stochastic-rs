use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_distributions::special::norm_pdf;

use super::*;

#[test]
fn constructor_rejects_invalid_regularization() {
  for h in [0.0, -0.01, f64::NAN, f64::INFINITY] {
    let error = MtGreeks::try_new(constant_volatility_params(), h, 8).unwrap_err();
    assert!(
      error.to_string().contains("h must be finite and positive"),
      "h={h}: unexpected error: {error}"
    );
  }
}

#[test]
fn constructor_rejects_zero_paths() {
  let error = MtGreeks::try_new(constant_volatility_params(), 0.01, 0).unwrap_err();
  assert!(
    error.to_string().contains("n_paths must be positive"),
    "unexpected error: {error}"
  );
}

#[test]
fn correlated_stochastic_volatility_returns_a_recoverable_error() {
  let mut params = constant_volatility_params();
  params.assets[0].xi = 0.3;
  params.assets[0].rho = -0.7;
  let engine = MtGreeks::new(params, 0.01, 8);
  let payoff = MtPayoff::DigitalPut2D {
    strikes: [100.0, 100.0],
  };
  let error = engine
    .try_delta_with_seed(&payoff, 0, 0xc077_e1a7)
    .unwrap_err();
  let cross_error = engine
    .try_cross_gamma_with_seed(&payoff, 0, 1, 0xc077_e1a7)
    .unwrap_err();

  assert!(
    error.to_string().contains("xi == 0 or rho == 0"),
    "unexpected error: {error}"
  );
  assert!(
    cross_error.to_string().contains("xi == 0 or rho == 0"),
    "unexpected cross-Gamma error: {cross_error}"
  );
}

#[test]
fn stochastic_volatility_is_supported_when_price_noise_is_independent() {
  let mut params = constant_volatility_params();
  params.assets[0].xi = 0.2;
  params.assets[0].rho = 0.0;
  let engine = MtGreeks::new(params, 0.01, 64);
  let payoff = MtPayoff::DigitalPut2D {
    strikes: [100.0, 100.0],
  };
  assert!(engine.try_delta_with_seed(&payoff, 0, 0x1ade_0e0d).is_ok());
}

/// Conditional on variance paths with `rho=0` and independent asset-price
/// shocks, each terminal log-price is Gaussian with integrated variance
/// `I_i`. Differentiating `Phi(a_1) Phi(a_2)` gives this independent benchmark.
#[test]
fn independent_stochastic_vol_delta_matches_conditional_benchmark() {
  let mut params = constant_volatility_params();
  params.cross_corr = Array2::<f64>::eye(2);
  params.assets[0].xi = 0.18;
  params.assets[1].xi = 0.14;
  params.assets[0].rho = 0.0;
  params.assets[1].rho = 0.0;
  params.n_steps = 17;
  let payoff = MtPayoff::DigitalPut2D {
    strikes: [100.0, 100.0],
  };
  let n_paths = 10_000;
  let seed = 0x1ade_c011_d17a;
  let estimate = MtGreeks::new(params.clone(), 0.01, n_paths)
    .try_delta_with_seed(&payoff, 0, seed)
    .unwrap();
  let dt = params.tau / (params.n_steps - 1) as f64;
  let discount = (-params.r * params.tau).exp();
  let mut seed_state = seed;
  let mut conditional_sum = 0.0;

  for _ in 0..n_paths {
    let path_seed = crate::simd_rng::derive_seed(&mut seed_state);
    let paths = params.sample_with_seed(path_seed);
    let integrated = (0..2)
      .map(|asset| {
        (0..params.n_steps - 1)
          .map(|step| paths.vols[[asset, step]] * dt)
          .sum::<f64>()
      })
      .collect::<Vec<_>>();
    let a = (0..2)
      .map(|asset| {
        let root_i = integrated[asset].sqrt();
        ((100.0 / params.assets[asset].s0).ln() - params.r * params.tau + 0.5 * integrated[asset])
          / root_i
      })
      .collect::<Vec<_>>();
    conditional_sum +=
      -discount * norm_pdf(a[0]) * norm_cdf(a[1]) / (params.assets[0].s0 * integrated[0].sqrt());
  }

  let benchmark = conditional_sum / n_paths as f64;
  assert!(
    (estimate - benchmark).abs() < 7.5e-4,
    "M-T delta={estimate}, conditional benchmark={benchmark}"
  );
}

#[test]
fn cross_gamma_keeps_the_down_bump_positive_for_small_spots() {
  let mut params = constant_volatility_params();
  params.assets[0].s0 = 0.005;
  let engine = MtGreeks::new(params, 0.01, 8);
  let payoff = MtPayoff::DigitalPut2D {
    strikes: [0.005, 100.0],
  };

  assert!(
    engine
      .try_cross_gamma_with_seed(&payoff, 1, 0, 0x5a11_5a07)
      .is_ok()
  );
}

#[test]
fn fallible_delta_rejects_invalid_payoff_parameters() {
  let engine = MtGreeks::new(constant_volatility_params(), 0.01, 8);
  let invalid = [
    MtPayoff::DigitalPut2D {
      strikes: [f64::NAN, 100.0],
    },
    MtPayoff::DigitalPut2D {
      strikes: [0.0, 100.0],
    },
    MtPayoff::Call {
      asset: 0,
      strike: f64::INFINITY,
    },
    MtPayoff::BasketCall {
      weights: vec![1.0, f64::NAN],
      strike: 100.0,
    },
    MtPayoff::WorstOfPut { strike: f64::NAN },
  ];

  for payoff in invalid {
    assert!(engine.try_delta_with_seed(&payoff, 0, 0xf1a1_7e00).is_err());
  }
}
