//! Comparison tests for the lattice framework: binomial pricing against Black-Scholes, short-rate
//! trees against analytic Gaussian-model bond prices or deterministic limits.

use ndarray::Array1;
use stochastic_rs::quant::calibration::BlackKarasinskiParams;
use stochastic_rs::quant::calibration::G2ppParams;
use stochastic_rs::quant::calibration::HullWhiteParams;
use stochastic_rs::quant::lattice::BinomialTree;
use stochastic_rs::quant::lattice::BlackKarasinskiTree;
use stochastic_rs::quant::lattice::BlackKarasinskiTreeModel;
use stochastic_rs::quant::lattice::G2ppTree;
use stochastic_rs::quant::lattice::G2ppTreeModel;
use stochastic_rs::quant::lattice::HullWhiteTree;
use stochastic_rs::quant::lattice::HullWhiteTreeModel;
use stochastic_rs::quant::pricing::BsmCoc;
use stochastic_rs::quant::pricing::BsmPricer;
use stochastic_rs::traits::ModelPricer;

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

  let bsm = BsmPricer::new(sigma, BsmCoc::Merton1973).price_call(s0, k, r, 0.0, tau);

  approx(lattice_price, bsm, 2e-2);
}

#[test]
fn hull_white_tree_matches_vasicek_zero_coupon_formula() {
  let r0: f64 = 0.03;
  let a: f64 = 0.4;
  let theta: f64 = 0.05;
  let sigma: f64 = 0.02;
  let maturity: f64 = 5.0;
  let tree = HullWhiteTree::new(
    HullWhiteTreeModel::new(HullWhiteParams {
      mean_reversion: a,
      sigma,
      initial_rate: r0,
      theta,
    }),
    maturity,
    400,
  );

  let lattice_price = tree.zero_coupon_bond_price();
  let b = (1.0 - (-a * maturity).exp()) / a;
  let a_term = ((theta - sigma * sigma / (2.0 * a * a)) * (b - maturity)
    - sigma * sigma * b * b / (4.0 * a))
    .exp();
  let reference = a_term * (-b * r0).exp();

  // The tree discounts at each step's start, a first-order error: `+8.3e-5` at 400 steps.
  approx(lattice_price, reference, 1e-4);
}

#[test]
fn black_karasinski_tree_matches_low_vol_deterministic_limit() {
  let r0: f64 = 0.03;
  let mean_reversion: f64 = 0.7;
  let long_run_rate: f64 = 0.05;
  let sigma: f64 = 1e-6;
  let maturity: f64 = 3.0;
  let tree = BlackKarasinskiTree::new(
    BlackKarasinskiTreeModel::new(BlackKarasinskiParams {
      mean_reversion,
      sigma,
      initial_rate: r0,
      long_run_rate,
    }),
    maturity,
    240,
  );

  let lattice_price = tree.zero_coupon_bond_price();

  let theta_log = long_run_rate.ln();
  let x0 = r0.ln();
  let log_rate = |t: f64| theta_log + (x0 - theta_log) * (-mean_reversion * t).exp();
  let n = 20_000;
  let dt = maturity / n as f64;
  let integral = (0..n)
    .map(|i| log_rate((i as f64 + 0.5) * dt).exp() * dt)
    .sum::<f64>();
  approx(lattice_price, (-integral).exp(), 1e-4);
  // Each level sits on the log-rate's exact mean, so at σ → 0 the tree is that mean's left sum.
  let step = maturity / 240.0;
  let left = (0..240)
    .map(|i| log_rate(i as f64 * step).exp() * step)
    .sum::<f64>();
  approx(lattice_price, (-left).exp(), 1e-12);
}

#[test]
fn g2pp_tree_matches_independent_factor_reference_when_rho_zero() {
  let maturity = 4.0;
  let model = G2ppTreeModel::new(
    0.01,
    0.015,
    G2ppParams {
      mean_reversion_x: 0.4,
      mean_reversion_y: 0.8,
      sigma_x: 0.02,
      sigma_y: 0.015,
      rho: 0.0,
      phi: 0.005,
    },
  );
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

  approx(lattice_price, reference, 2.5e-4);
}

fn vasicek_bond_price(r0: f64, theta: f64, a: f64, sigma: f64, maturity: f64) -> f64 {
  let b = (1.0 - (-a * maturity).exp()) / a;
  let a_term = ((theta - sigma * sigma / (2.0 * a * a)) * (b - maturity)
    - sigma * sigma * b * b / (4.0 * a))
    .exp();
  a_term * (-b * r0).exp()
}
