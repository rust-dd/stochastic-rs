//! Closed-form anchors of [`HestonHullWhite2F`], one per leg of the model, each exact in continuous time:
//!
//! 1. rate volatilities at zero: the Heston closed form of [`HestonPricer`], under both variance steps;
//! 2. the rate leg alone: `∫₀ᵀ r` is Gaussian with a known mean and variance, and the bond price is its
//!    exponential moment, under both variance steps since each correlates the rate shocks its own way;
//! 3. variance frozen: the spot is lognormal under two-factor Gaussian rates and a European option is
//!    Black's formula on the forward, whose total variance carries both equity-rate correlations, under
//!    both variance steps since each builds the spot shock from the rate shocks its own way.
//!
//! Each test also asserts its own power: the statistic it holds to the reference must reject the
//! reference with the tested ingredient changed, so a test that could not fail does not pass.
//!
//! With `τ = T − s`, `B_c(τ) = (1 − e^{−cτ})/c` and `r(t) = m(t) + σ₁ ∫ e^{−a(t−s)} dW^r_s +
//! σ₂/(a−b) ∫ (e^{−b(t−s)} − e^{−a(t−s)}) dW^u_s`, the integrated rate loads `g₁(τ) = σ₁ B_a(τ)` on `W^r`
//! and `g₂(τ) = σ₂ (B_b(τ) − B_a(τ))/(a − b)` on `W^u`, so
//!
//! - `Var ∫₀ᵀ r = ∫₀ᵀ (g₁² + g₂² + 2 ρ_ru g₁ g₂) dτ`, the `V(0, T)` of Brigo and Mercurio (2006), eq. 4.10,
//!   through the map of their section 4.2.5 between this formulation and G2++;
//! - `Cov(W^x_T, ∫₀ᵀ r) = ∫₀ᵀ (ρ_xr g₁ + ρ_xu g₂) dτ`;
//! - with the variance frozen at `θ`, `Var log S_T = θT + Var ∫r + 2 √θ Cov(W^x_T, ∫r)` and
//!   `P(0, T) = exp(−E ∫r + ½ Var ∫r)`.
//!
//! The one-factor case of the third formula is QuantLib's `AnalyticBSMHullWhiteEngine`, which it matches
//! to `1e-16` at `T = 1, 5, 10` and three strikes.
//!
//! Reference: Brigo D. & Mercurio F. (2006), *Interest Rate Models: Theory and Practice*, 2nd ed.,
//! Springer, sections 4.2.3 to 4.2.5, DOI: 10.1007/978-3-540-34604-3.

use stochastic_rs::ndarray::Array2;
use stochastic_rs::ndarray::array;
use stochastic_rs::quant::pricing::bsm::BsmCoc;
use stochastic_rs::quant::pricing::bsm::BsmPricer;
use stochastic_rs::quant::pricing::heston::HestonPricer;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::volatility::heston_hull_white_2f::HestonHullWhite2F;
use stochastic_rs::stochastic::volatility::heston_hull_white_2f::HybridScheme;
use stochastic_rs::traits::ProcessExt;

const PATHS: usize = 100_000;
/// Standard errors a statistic may sit from its closed form.
const Z_MAX: f64 = 3.0;
/// Standard errors by which a statistic must reject a reference with the tested ingredient changed.
const Z_POWER: f64 = 5.0;
const FLAT_RATE: f64 = 0.03;

/// Rate leg of tests 2 and 3: slower mean reversion and larger volatilities than a calibrated curve
/// would give, so the rate terms are a large share of what is measured.
const A: f64 = 0.3;
const B: f64 = 0.1;
const SIGMA1: f64 = 0.02;
const SIGMA2: f64 = 0.01;
const RHO_RU: f64 = -0.4;
const RHO_XR: f64 = 0.4;
const RHO_XU: f64 = 0.3;

/// Holds the mean short rate at [`FLAT_RATE`] from `r₀ = FLAT_RATE`, so `E ∫₀ᵀ r = FLAT_RATE · T`.
fn flat_mean_theta(_t: f64) -> f64 {
  A * FLAT_RATE
}

/// `B_c(τ) = (1 − e^{−cτ})/c`.
fn b_fn(c: f64, tau: f64) -> f64 {
  (1.0 - (-c * tau).exp()) / c
}

/// `∫₀ᵀ B_c(τ) B_d(τ) dτ`.
fn b_product_integral(c: f64, d: f64, t: f64) -> f64 {
  (t - b_fn(c, t) - b_fn(d, t) + b_fn(c + d, t)) / (c * d)
}

/// `Var ∫₀ᵀ r` of the rate leg with `W^r`, `W^u` correlation `rho_ru`.
fn var_int_r(rho_ru: f64, t: f64) -> f64 {
  let k = SIGMA2 / (A - B);
  let (jaa, jab, jbb) = (
    b_product_integral(A, A, t),
    b_product_integral(A, B, t),
    b_product_integral(B, B, t),
  );
  SIGMA1 * SIGMA1 * jaa + k * k * (jbb - 2.0 * jab + jaa) + 2.0 * rho_ru * SIGMA1 * k * (jab - jaa)
}

/// `Cov(W^x_T, ∫₀ᵀ r)` for equity-rate correlations `rho_xr`, `rho_xu`.
fn cov_wx_int_r(rho_xr: f64, rho_xu: f64, t: f64) -> f64 {
  let k = SIGMA2 / (A - B);
  let (ia, ib) = ((t - b_fn(A, t)) / A, (t - b_fn(B, t)) / B);
  rho_xr * SIGMA1 * ia + rho_xu * k * (ib - ia)
}

fn rho(rho_xv: f64, rho_xr: f64, rho_xu: f64, rho_ru: f64) -> Array2<f64> {
  array![
    [1.0, rho_xv, rho_xr, rho_xu],
    [rho_xv, 1.0, 0.0, 0.0],
    [rho_xr, 0.0, 1.0, rho_ru],
    [rho_xu, 0.0, rho_ru, 1.0],
  ]
}

fn mean_and_se(values: impl Iterator<Item = f64>) -> (f64, f64) {
  let values = values.collect::<Vec<_>>();
  let n = values.len() as f64;
  let mean = values.iter().sum::<f64>() / n;
  let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
  (mean, (var / n).sqrt())
}

/// Sample variance and its standard error under a Gaussian law of variance `reference`.
fn variance_z(values: impl Iterator<Item = f64>, reference: f64) -> (f64, f64) {
  let values = values.collect::<Vec<_>>();
  let n = values.len() as f64;
  let mean = values.iter().sum::<f64>() / n;
  let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
  (
    var,
    (var - reference) / (reference * (2.0 / (n - 1.0)).sqrt()),
  )
}

/// Discounted put prices at `strikes` from `(S_T, ∫₀ᵀ r)` pairs, with their standard errors.
fn put_prices(terminal: &[(f64, f64)], strikes: &[f64]) -> Vec<(f64, f64)> {
  strikes
    .iter()
    .map(|k| {
      mean_and_se(
        terminal
          .iter()
          .map(|(s, int_r)| (k - s).max(0.0) * (-int_r).exp()),
      )
    })
    .collect()
}

/// Test 1. `κ = 1.5`, `θ = v₀ = 0.04`, `σ = 0.3`, `ρ_xv = −0.7`, a constant 3 percent rate, `T = 2` on 100
/// steps a year: with `σ₁ = σ₂ = 0` and `θ_r = a r₀` the short rate never moves and the model is Heston.
fn pure_heston<Sch: HybridScheme>(
  model: HestonHullWhite2F<f64, Sch, Deterministic>,
  rho_xv: f64,
  what: &str,
) {
  const STRIKES: [f64; 3] = [0.8, 1.0, 1.2];
  let (tau, n) = (2.0, model.n);
  let terminal = model.sample_map(PATHS, |path| (path[(0, n - 1)], path[(4, n - 1)]));
  let prices = put_prices(&terminal, &STRIKES);
  let z_against = |rho_ref: f64| {
    let closed = HestonPricer::new(0.04, rho_ref, 1.5, 0.04, 0.3, None);
    STRIKES
      .iter()
      .zip(&prices)
      .map(|(&k, &(price, se))| {
        let reference = closed.call_put(1.0, k, FLAT_RATE, 0.0, tau).1;
        let z = (price - reference) / se;
        println!(
          "{what}, rho_xv = {rho_ref:+.1}, K = {k}: {price:.6} ± {se:.6} against {reference:.6} (z = {z:+.2})"
        );
        z.abs()
      })
      .fold(0.0_f64, f64::max)
  };
  let worst = z_against(rho_xv);
  assert!(
    worst < Z_MAX,
    "{what}: a put is {worst:.2} standard errors from the Heston closed form, above {Z_MAX}"
  );
  let flipped = z_against(-rho_xv);
  assert!(
    flipped > Z_POWER,
    "{what}: the Heston closed form with the spot-variance correlation flipped is only {flipped:.2} standard errors away"
  );
}

#[test]
fn zero_rate_volatility_prices_the_heston_closed_form() {
  let rho_xv = -0.7;
  let model = HestonHullWhite2F::new(
    1.5,
    0.04,
    0.3,
    flat_mean_theta as fn(f64) -> f64,
    A,
    0.0,
    0.0,
    B,
    rho(rho_xv, RHO_XR, RHO_XU, RHO_RU),
    201,
    Some(1.0),
    Some(0.04),
    Some(FLAT_RATE),
    Some(2.0),
    Deterministic::new(1993),
  );
  pure_heston(model.clone(), rho_xv, "Euler");
  pure_heston(model.qe(), rho_xv, "QE");
}

/// Test 2. `∫₀ᵀ r` at 1, 5 and 10 years on 100 steps a year: its mean is `FLAT_RATE · T`, its variance
/// the closed form, and `E exp(−∫r)` the bond price `exp(−FLAT_RATE · T + ½ Var ∫r)`.
fn rate_leg<Sch: HybridScheme>(model: HestonHullWhite2F<f64, Sch, Deterministic>, what: &str) {
  const MATURITIES: [(f64, usize); 3] = [(1.0, 100), (5.0, 500), (10.0, 1000)];
  let integrals = model.sample_map(PATHS, |path| MATURITIES.map(|(_, i)| path[(4, i)]));
  for (m, &(tau, _)) in MATURITIES.iter().enumerate() {
    let int_r = || integrals.iter().map(|row| row[m]);
    let v = var_int_r(RHO_RU, tau);
    let (mean, mean_se) = mean_and_se(int_r());
    let z_mean = (mean - FLAT_RATE * tau) / mean_se;
    let (var, z_var) = variance_z(int_r(), v);
    let (bond, bond_se) = mean_and_se(int_r().map(|x| (-x).exp()));
    let closed_bond = (-FLAT_RATE * tau + 0.5 * v).exp();
    let z_bond = (bond - closed_bond) / bond_se;
    println!(
      "{what}, T = {tau}: mean {mean:.6} (z = {z_mean:+.2}), variance {var:.6e} against {v:.6e} (z = {z_var:+.2}), bond {bond:.6} against {closed_bond:.6} (z = {z_bond:+.2})"
    );
    assert!(
      z_mean.abs() < Z_MAX,
      "{what}, T = {tau}: E ∫r = {mean} ± {mean_se}, {z_mean:.2} standard errors from {}",
      FLAT_RATE * tau
    );
    assert!(
      z_var.abs() < Z_MAX,
      "{what}, T = {tau}: Var ∫r = {var}, {z_var:.2} standard errors from the closed form {v}"
    );
    assert!(
      z_bond.abs() < Z_MAX,
      "{what}, T = {tau}: E exp(−∫r) = {bond} ± {bond_se}, {z_bond:.2} standard errors from the bond price {closed_bond}"
    );
    let (_, z_flipped) = variance_z(int_r(), var_int_r(-RHO_RU, tau));
    assert!(
      z_flipped.abs() > Z_POWER,
      "{what}, T = {tau}: the variance with the factor correlation flipped is only {z_flipped:.2} standard errors away"
    );
    if tau >= 5.0 {
      let z_no_convexity = (bond - (-FLAT_RATE * tau).exp()) / bond_se;
      assert!(
        z_no_convexity.abs() > Z_POWER,
        "{what}, T = {tau}: the bond price without the convexity term is only {z_no_convexity:.2} standard errors away"
      );
    }
  }
}

#[test]
fn integrated_rate_has_the_closed_form_moments_and_bond_price() {
  let model = HestonHullWhite2F::new(
    0.8,
    0.2,
    0.2,
    flat_mean_theta as fn(f64) -> f64,
    A,
    SIGMA1,
    SIGMA2,
    B,
    rho(-0.3, RHO_XR, RHO_XU, RHO_RU),
    1001,
    Some(1.0),
    Some(0.2),
    Some(FLAT_RATE),
    Some(10.0),
    Deterministic::new(2006),
  );
  rate_leg(model.clone(), "Euler");
  rate_leg(model.qe(), "QE");
}

/// Test 3. `v₀ = θ = 0.04` and a spot-variance correlation of zero, `T = 5` on 100 steps a year. With
/// the variance frozen `log S_T` is Gaussian and a put is Black's formula at total variance
/// `θT + Var ∫r + 2 √θ Cov(W^x_T, ∫r)` and bond price `P(0, T)`, here written as Black-Scholes at the
/// rate `−log P(0, T)/T`.
fn frozen_variance<Sch: HybridScheme>(
  model: HestonHullWhite2F<f64, Sch, Deterministic>,
  what: &str,
) {
  const STRIKES: [f64; 3] = [0.8, 1.0, 1.3];
  let (theta, tau, n) = (model.theta, 5.0, model.n);
  let terminal = model.sample_map(PATHS, |path| (path[(0, n - 1)], path[(4, n - 1)]));
  let prices = put_prices(&terminal, &STRIKES);
  let v = var_int_r(RHO_RU, tau);
  let total_variance = |rho_xr: f64, rho_xu: f64| {
    theta * tau + v + 2.0 * theta.sqrt() * cov_wx_int_r(rho_xr, rho_xu, tau)
  };
  let bond_rate = FLAT_RATE - 0.5 * v / tau;

  let log_spot = || terminal.iter().map(|(s, _)| s.ln());
  let (var, z_var) = variance_z(log_spot(), total_variance(RHO_XR, RHO_XU));
  println!(
    "{what}: Var log S_T = {var:.6} against {:.6} (z = {z_var:+.2})",
    total_variance(RHO_XR, RHO_XU)
  );
  assert!(
    z_var.abs() < Z_MAX,
    "{what}: Var log S_T = {var}, {z_var:.2} standard errors from the closed form {}",
    total_variance(RHO_XR, RHO_XU)
  );

  let z_prices = |rho_xr: f64, rho_xu: f64| {
    let black = BsmPricer::new(
      (total_variance(rho_xr, rho_xu) / tau).sqrt(),
      BsmCoc::Merton1973,
    );
    STRIKES
      .iter()
      .zip(&prices)
      .map(|(&k, &(price, se))| {
        let reference = black.call_put(1.0, k, bond_rate, 0.0, tau).1;
        let z = (price - reference) / se;
        println!(
          "{what}, rho_xr = {rho_xr}, rho_xu = {rho_xu}, K = {k}: {price:.6} ± {se:.6} against {reference:.6} (z = {z:+.2})"
        );
        z.abs()
      })
      .fold(0.0_f64, f64::max)
  };
  let worst = z_prices(RHO_XR, RHO_XU);
  assert!(
    worst < Z_MAX,
    "{what}: a put is {worst:.2} standard errors from Black on the forward, above {Z_MAX}"
  );

  // Power: no equity-rate correlation at all, then each factor's correlation removed on its own.
  let uncoupled = z_prices(0.0, 0.0);
  assert!(
    uncoupled > Z_POWER,
    "{what}: Black without the equity-rate correlations is only {uncoupled:.2} standard errors away"
  );
  for (missing, reference) in [
    (
      "without the equity-rate correlations",
      total_variance(0.0, 0.0),
    ),
    (
      "without the second factor's correlation",
      total_variance(RHO_XR, 0.0),
    ),
    (
      "without the first factor's correlation",
      total_variance(0.0, RHO_XU),
    ),
  ] {
    let (_, z) = variance_z(log_spot(), reference);
    assert!(
      z.abs() > Z_POWER,
      "{what}: Var log S_T {missing} is only {z:.2} standard errors away"
    );
  }
}

/// Euler runs at `σ = 0`, where the closed form is exact. The quadratic-exponential step needs a
/// positive volatility of variance and runs at `σ = 0.01`: the variance then moves by 4 percent of its
/// level, which changes `Var log S_T` through `E √v` and `Var ½∫v` by under `1e-4` of its value
/// (`Var v ≤ σ²θ/(2κ)`, `Var ∫v ≤ σ²θT/κ²`), under a fortieth of the standard error here.
#[test]
fn frozen_variance_prices_black_on_the_forward_under_two_factor_rates() {
  let theta = 0.04;
  let model = HestonHullWhite2F::new(
    0.8,
    theta,
    0.0,
    flat_mean_theta as fn(f64) -> f64,
    A,
    SIGMA1,
    SIGMA2,
    B,
    rho(0.0, RHO_XR, RHO_XU, RHO_RU),
    501,
    Some(1.0),
    Some(theta),
    Some(FLAT_RATE),
    Some(5.0),
    Deterministic::new(1973),
  );
  frozen_variance(model.clone(), "Euler");
  frozen_variance(model.with_sigma(0.01).qe(), "QE");
}
