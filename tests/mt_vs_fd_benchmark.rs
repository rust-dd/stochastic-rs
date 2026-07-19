//! Ignored performance comparisons for seeded M-T and finite-difference Greeks.
//!
//! Run with `cargo test --release --features openblas --test mt_vs_fd_benchmark
//! -- --ignored --test-threads=1`.

#![cfg(feature = "openblas")]

use ndarray::Array1;
use ndarray::Array2;
use owens_t::biv_norm;
use stochastic_rs::quant::pricing::malliavin_thalmaier::AssetParams;
use stochastic_rs::quant::pricing::malliavin_thalmaier::MtGreeks;
use stochastic_rs::quant::pricing::malliavin_thalmaier::MtPayoff;
use stochastic_rs::quant::pricing::malliavin_thalmaier::MultiHestonParams;

fn make_params_2d() -> MultiHestonParams<f64> {
  let first = AssetParams {
    s0: 100.0,
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    xi: 0.0,
    rho: 0.0,
  };
  let second = first.clone();
  let mut cross_corr = Array2::<f64>::eye(2);
  cross_corr[[0, 1]] = 0.5;
  cross_corr[[1, 0]] = 0.5;
  MultiHestonParams {
    assets: vec![first, second],
    cross_corr,
    r: 0.05,
    tau: 1.0,
    n_steps: 252,
  }
}

fn fd_all_deltas(
  params: &MultiHestonParams<f64>,
  payoff: &MtPayoff<f64>,
  n_paths: usize,
  seed: u64,
) -> Array1<f64> {
  let bump = 0.5;
  Array1::from_shape_fn(params.n_assets(), |asset| {
    let mut up = params.clone();
    up.assets[asset].s0 += bump;
    let mut down = params.clone();
    down.assets[asset].s0 -= bump;
    let price_up = MtGreeks::new(up, 0.01, n_paths).price_with_seed(payoff, seed);
    let price_down = MtGreeks::new(down, 0.01, n_paths).price_with_seed(payoff, seed);
    (price_up - price_down) / (2.0 * bump)
  })
}

/// Seeded common-random-number comparison for the digital-put Delta vector.
#[test]
#[ignore = "50,000-path M-T versus finite-difference performance comparison"]
fn mt_vs_fd_digital_put_2d() {
  let n_paths = 50_000;
  let seed = 0xd161_7a1f;
  let params = make_params_2d();
  let payoff = MtPayoff::DigitalPut2D {
    strikes: [100.0, 100.0],
  };
  let fd_deltas = fd_all_deltas(&params, &payoff, n_paths, seed);
  let mt_deltas = MtGreeks::new(params, 0.01, n_paths)
    .try_all_deltas_with_seed(&payoff, seed)
    .unwrap();

  for asset in 0..2 {
    let error = (mt_deltas[asset] - fd_deltas[asset]).abs();
    assert!(
      error < 0.002,
      "asset {asset}: MT={}, FD={}, abs_error={error}",
      mt_deltas[asset],
      fd_deltas[asset]
    );
  }
}

/// Reference: the constant-volatility bivariate Black--Scholes digital-put
/// price is the discounted bivariate normal probability at the two log-strike
/// thresholds.
#[test]
#[ignore = "50,000-path digital-put price comparison"]
fn digital_put_price_matches_bivariate_black_scholes() {
  let n_paths = 50_000;
  let params = make_params_2d();
  let payoff = MtPayoff::DigitalPut2D {
    strikes: [100.0, 100.0],
  };
  let price = MtGreeks::new(params, 0.01, n_paths).price_with_seed(&payoff, 0xb52d_0001);
  let sigma = 0.2_f64;
  let rate = 0.05;
  let tau = 1.0;
  let threshold = (0.5 * sigma * sigma - rate) * tau / (sigma * tau.sqrt());
  let expected = (-rate * tau).exp() * biv_norm(-threshold, -threshold, 0.5);

  assert!(
    (price - expected).abs() < 0.01,
    "MC price={price}, Black--Scholes={expected}"
  );
}
