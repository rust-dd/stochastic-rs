//! Public-API regression tests for the Malliavin--Thalmaier Greeks engine.

#![cfg(feature = "openblas")]

use ndarray::Array2;
use stochastic_rs::quant::pricing::malliavin_thalmaier::AssetParams;
use stochastic_rs::quant::pricing::malliavin_thalmaier::MtGreeks;
use stochastic_rs::quant::pricing::malliavin_thalmaier::MtPayoff;
use stochastic_rs::quant::pricing::malliavin_thalmaier::MultiHestonParams;
use stochastic_rs::quant::pricing::malliavin_thalmaier::g_digital_put_2d;

fn paper_params() -> MultiHestonParams<f64> {
  let first = AssetParams {
    s0: 100.0,
    v0: 0.25 * 0.25,
    kappa: 1.0,
    theta: 0.25 * 0.25,
    xi: 0.0,
    rho: 0.0,
  };
  let second = AssetParams {
    s0: 100.0,
    v0: 0.20 * 0.20,
    kappa: 1.0,
    theta: 0.20 * 0.20,
    xi: 0.0,
    rho: 0.0,
  };
  let mut cross_corr = Array2::<f64>::eye(2);
  cross_corr[[0, 1]] = 0.2;
  cross_corr[[1, 0]] = 0.2;
  MultiHestonParams {
    assets: vec![first, second],
    cross_corr,
    r: 0.0,
    tau: 1.0,
    n_steps: 9,
  }
}

fn vega_params() -> MultiHestonParams<f64> {
  let mut params = paper_params();
  params.assets[0].kappa = 0.0;
  params
}

fn black_scholes_vega(spot: f64, strike: f64, sigma: f64, rate: f64, tau: f64) -> f64 {
  let d1 = ((spot / strike).ln() + (rate + 0.5 * sigma * sigma) * tau) / (sigma * tau.sqrt());
  let density = (-0.5 * d1 * d1).exp() / (2.0 * std::f64::consts::PI).sqrt();
  spot * density * tau.sqrt()
}

fn black_scholes_digital_cross_gamma(
  spot: [f64; 2],
  strike: [f64; 2],
  sigma: [f64; 2],
  correlation: f64,
  rate: f64,
  tau: f64,
) -> f64 {
  let root_tau = tau.sqrt();
  let a1 =
    ((strike[0] / spot[0]).ln() - (rate - 0.5 * sigma[0] * sigma[0]) * tau) / (sigma[0] * root_tau);
  let a2 =
    ((strike[1] / spot[1]).ln() - (rate - 0.5 * sigma[1] * sigma[1]) * tau) / (sigma[1] * root_tau);
  let root_one_minus_rho_sq = (1.0 - correlation * correlation).sqrt();
  let conditional = (a2 - correlation * a1) / root_one_minus_rho_sq;
  let density_a1 = (-0.5 * a1 * a1).exp() / (2.0 * std::f64::consts::PI).sqrt();
  let density_conditional =
    (-0.5 * conditional * conditional).exp() / (2.0 * std::f64::consts::PI).sqrt();
  (-rate * tau).exp() * density_a1 * density_conditional
    / (spot[0] * spot[1] * sigma[0] * sigma[1] * tau * root_one_minus_rho_sq)
}

/// Reference: Kohatsu-Higa and Yasuda, "Estimating Multi-dimensional Density
/// Functions through the Malliavin-Thalmaier Formula and Its Application to
/// Finance", RIMS Kokyuroku 1580 (2008), Section 7, equation (7.1), with
/// `(sigma1,sigma2)=(0.25,0.20)`, price correlation `0.2`, `r=0`, `tau=1`,
/// and eight Euler intervals. The exact bivariate Black--Scholes Deltas are
/// `-0.008399800188218226` and `-0.010770732373919333`.
#[test]
fn paper_digital_put_deltas_match_exact_values() {
  let payoff = MtPayoff::DigitalPut2D {
    strikes: [100.0, 100.0],
  };
  let deltas = MtGreeks::new(paper_params(), 0.01, 25_000)
    .try_all_deltas_with_seed(&payoff, 0x1580_0011)
    .unwrap();
  let expected = [-0.008399800188218226, -0.010770732373919333];

  for asset in 0..2 {
    let error = (deltas[asset] - expected[asset]).abs();
    assert!(
      error < 5.0e-4,
      "asset {asset}: delta={}, exact={}, abs_error={error}",
      deltas[asset],
      expected[asset]
    );
  }
}

/// Reference: Black--Scholes vega for an ATM call with volatility `0.25` is
/// `39.583768694474955`; `dP/dv0 = (dP/dsigma)/(2 sigma)`.
#[test]
fn public_vega_uses_volatility_units() {
  let payoff = MtPayoff::Call {
    asset: 0,
    strike: 100.0,
  };
  let engine = MtGreeks::new(vega_params(), 0.01, 30_000);
  let seed = 0x5e6a_0001;
  let sigma_vega = engine.vega_with_seed(&payoff, 0, seed);
  let variance_vega = engine.vega_v0_with_seed(&payoff, 0, seed);
  let expected = black_scholes_vega(100.0, 100.0, 0.25, 0.0, 1.0);

  assert!(
    (sigma_vega - expected).abs() < 0.8,
    "dP/dsigma={sigma_vega}, Black--Scholes={expected}"
  );
  assert!(
    (variance_vega - expected / 0.5).abs() < 1.7,
    "dP/dv0={variance_vega}, Black--Scholes={}",
    expected / 0.5
  );
  assert!(
    (sigma_vega - 0.5 * variance_vega).abs() < 0.04,
    "chain rule mismatch: dP/dsigma={sigma_vega}, dP/dv0={variance_vega}"
  );
}

/// Reference: differentiating the bivariate Black--Scholes digital-put price
/// twice gives the mixed spot derivative used here as the golden value.
#[test]
fn public_cross_gamma_is_seeded_and_matches_black_scholes() {
  let payoff = MtPayoff::DigitalPut2D {
    strikes: [100.0, 100.0],
  };
  let seed = 0xc205_5a11;
  let expected =
    black_scholes_digital_cross_gamma([100.0, 100.0], [100.0, 100.0], [0.25, 0.20], 0.2, 0.0, 1.0);
  let estimate =
    MtGreeks::new(paper_params(), 0.01, 30_000).cross_gamma_with_seed(&payoff, 0, 1, seed);
  let reproducibility_engine = MtGreeks::new(paper_params(), 0.01, 128);
  let first = reproducibility_engine.cross_gamma_with_seed(&payoff, 0, 1, seed);
  let second = reproducibility_engine.cross_gamma_with_seed(&payoff, 0, 1, seed);

  assert_eq!(first.to_bits(), second.to_bits());
  assert!(
    (estimate - expected).abs() < 5.0e-5,
    "cross-gamma={estimate}, Black--Scholes={expected}"
  );
}

#[test]
fn stochastic_leverage_is_reported_as_unsupported() {
  let mut params = paper_params();
  params.assets[0].xi = 0.3;
  params.assets[0].rho = -0.7;
  let engine = MtGreeks::new(params, 0.01, 16);
  let payoff = MtPayoff::DigitalPut2D {
    strikes: [100.0, 100.0],
  };
  let error = engine
    .try_delta_with_seed(&payoff, 0, 0xc077_e1a7)
    .unwrap_err();

  assert!(
    error.to_string().contains("xi == 0 or rho == 0"),
    "unexpected error: {error}"
  );
}

#[test]
fn digital_put_kernel_has_the_correct_second_diagonal() {
  let kernel = g_digital_put_2d::<f64>([90.0, 90.0], [100.0, 100.0]);

  assert!((kernel[0][0] - 0.5).abs() < 1.0e-12);
  assert!((kernel[1][1] - 0.5).abs() < 1.0e-12);
  assert!((kernel[0][1] - kernel[1][0]).abs() < 1.0e-12);
}
