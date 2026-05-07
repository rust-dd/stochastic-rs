//! Comparison tests for the lattice framework.
//!
//! Binomial pricing is verified against Black-Scholes. Short-rate trees are
//! checked against analytic Gaussian-model bond prices or deterministic limits.

use ndarray::Array1;
use stochastic_rs::quant::OptionType;
use stochastic_rs::quant::lattice::BinomialTree;
use stochastic_rs::quant::lattice::BlackKarasinskiTree;
use stochastic_rs::quant::lattice::BlackKarasinskiTreeModel;
use stochastic_rs::quant::lattice::G2ppTree;
use stochastic_rs::quant::lattice::G2ppTreeModel;
use stochastic_rs::quant::lattice::HullWhiteTree;
use stochastic_rs::quant::lattice::HullWhiteTreeModel;
use stochastic_rs::quant::pricing::BSMCoc;
use stochastic_rs::quant::pricing::BSMPricer;
use stochastic_rs::traits::PricerExt;

fn approx(a: f64, b: f64, tol: f64) {
  assert!(
    (a - b).abs() < tol,
    "expected {b:.12}, got {a:.12}, diff {:.2e}",
    (a - b).abs()
  );
}

#[test]
fn crr_binomial_tree_matches_black_scholes_call() {
  let s0 = 100.0;
  let k = 100.0;
  let r = 0.05;
  let sigma = 0.2;
  let tau = 1.0;
  let steps = 512;
  let dt = tau / steps as f64;
  let u = (sigma * dt.sqrt()).exp();
  let d = 1.0 / u;
  let p = ((r * dt).exp() - d) / (u - d);

  let tree = BinomialTree::from_crr(s0, u, d, p, steps, dt);
  let terminal = Array1::from_iter(
    tree
      .states
      .last()
      .unwrap()
      .iter()
      .map(|&spot| (spot - k).max(0.0)),
  );
  let lattice_price = tree.backward_induct(terminal, |_level, _node, _state| (-r * dt).exp());

  let bsm = BSMPricer::new(
    s0,
    sigma,
    k,
    r,
    None,
    None,
    Some(0.0),
    Some(tau),
    None,
    None,
    OptionType::Call,
    BSMCoc::Merton1973,
  )
  .calculate_price();

  approx(lattice_price, bsm, 2e-2);
}

#[test]
fn hull_white_tree_matches_vasicek_zero_coupon_formula() {
  let r0: f64 = 0.03;
  let a: f64 = 0.4;
  let theta: f64 = 0.05;
  let sigma: f64 = 0.02;
  let maturity: f64 = 5.0;
  let tree = HullWhiteTree::new(HullWhiteTreeModel::new(r0, a, theta, sigma), maturity, 400);

  let lattice_price = tree.zero_coupon_bond_price();
  let b = (1.0 - (-a * maturity).exp()) / a;
  let a_term = ((theta - sigma * sigma / (2.0 * a * a)) * (b - maturity)
    - sigma * sigma * b * b / (4.0 * a))
    .exp();
  let reference = a_term * (-b * r0).exp();

  approx(lattice_price, reference, 6e-3);
}

#[test]
fn black_karasinski_tree_matches_low_vol_deterministic_limit() {
  let r0: f64 = 0.03;
  let mean_reversion: f64 = 0.7;
  let long_run_rate: f64 = 0.05;
  let sigma: f64 = 1e-6;
  let maturity: f64 = 3.0;
  let tree = BlackKarasinskiTree::new(
    BlackKarasinskiTreeModel::new(r0, mean_reversion, long_run_rate, sigma),
    maturity,
    240,
  );

  let lattice_price = tree.zero_coupon_bond_price();

  let theta_log = long_run_rate.ln();
  let x0 = r0.ln();
  let n = 20_000;
  let dt = maturity / n as f64;
  let mut integral = 0.0;
  for i in 0..n {
    let t = (i as f64 + 0.5) * dt;
    let x_t = theta_log + (x0 - theta_log) * (-mean_reversion * t).exp();
    integral += x_t.exp() * dt;
  }
  let reference = (-integral).exp();

  approx(lattice_price, reference, 3e-3);
}

#[test]
fn g2pp_tree_matches_independent_factor_reference_when_rho_zero() {
  let maturity = 4.0;
  let model = G2ppTreeModel::new(0.01, 0.015, 0.005, 0.4, 0.8, 0.02, 0.015, 0.0);
  let tree = G2ppTree::new(model.clone(), maturity, 240);
  let lattice_price = tree.zero_coupon_bond_price();

  let ref_x = vasicek_bond_price(
    model.initial_x,
    0.0,
    model.mean_reversion_x,
    model.sigma_x,
    maturity,
  );
  let ref_y = vasicek_bond_price(
    model.initial_y,
    0.0,
    model.mean_reversion_y,
    model.sigma_y,
    maturity,
  );
  let reference = (-model.phi * maturity).exp() * ref_x * ref_y;

  approx(lattice_price, reference, 8e-3);
}

fn vasicek_bond_price(r0: f64, theta: f64, a: f64, sigma: f64, maturity: f64) -> f64 {
  let b = (1.0 - (-a * maturity).exp()) / a;
  let a_term = ((theta - sigma * sigma / (2.0 * a * a)) * (b - maturity)
    - sigma * sigma * b * b / (4.0 * a))
    .exp();
  a_term * (-b * r0).exp()
}
