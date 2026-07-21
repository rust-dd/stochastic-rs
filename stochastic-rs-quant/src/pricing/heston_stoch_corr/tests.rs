use super::*;
use crate::traits::PricerExt;
use crate::traits::TimeExt;

fn paper_pricer() -> HestonStochCorrPricer {
  // Parameters from Table 2 in Teng et al.
  HestonStochCorrPricer::new(
    100.0,      // s
    0.0,        // r
    100.0,      // k (ATM)
    0.02,       // v0
    2.1,        // kappa_v
    0.03,       // theta_v
    0.2,        // sigma_v
    -0.4,       // rho0
    3.4,        // kappa_r
    -0.6,       // mu_r
    0.1,        // sigma_r
    0.4,        // rho2
    1.0 / 12.0, // tau (1 month)
  )
}

/// With the correlation process frozen (σ_ρ → 0, ρ pinned to a constant) the
/// stochastic-correlation model collapses to standard Heston, so at ATM the
/// two must price the same. The Carr-Madan inversion used a fixed `φ_max = 200`
/// that truncated the short-dated tail: pre-fix at τ=0.02/ATM the two pricers
/// disagreed by ~18%. Both are now integrated to convergence and agree to
/// well under 1% down to τ=0.002.
#[test]
fn carr_madan_reduces_to_heston_short_dated() {
  use crate::pricing::heston::HestonPricer;
  let (rho, kappa, theta, sigma, v0, s, r) = (-0.7, 2.0, 0.04, 0.3, 0.04, 100.0, 0.03);
  for tau in [0.02, 0.005, 0.002] {
    let heston = HestonPricer::new(
      s, v0, s, r, None, rho, kappa, theta, sigma, Some(0.0), Some(tau), None, None,
    );
    let heston_call = heston.calculate_call_put().0;
    let hscm = HestonStochCorrPricer::new(
      s, r, s, v0, kappa, theta, sigma, rho, 10.0, rho, 1e-10, 0.0, tau,
    );
    let hscm_call = hscm.price_call_carr_madan();
    let reldiff = (heston_call - hscm_call).abs() / heston_call;
    assert!(
      reldiff < 0.01,
      "HSCM(σ_ρ→0) must match Heston at τ={tau}: Heston={heston_call:.6}, HSCM={hscm_call:.6}, reldiff={reldiff:.4}"
    );
  }
}

#[test]
fn char_func_at_zero_is_one() {
  let pricer = paper_pricer();
  let phi0 = pricer.char_func(0.0);
  assert!(
    (phi0.norm() - 1.0).abs() < 0.01,
    "φ(0) = {phi0}, expected ~1.0"
  );
}

#[test]
fn char_func_is_finite_and_bounded() {
  let pricer = paper_pricer();
  for u in [0.1, 1.0, 5.0, 10.0, 20.0] {
    let phi = pricer.char_func(u);
    assert!(phi.re.is_finite() && phi.im.is_finite(), "φ({u}) = {phi}");
    assert!(phi.norm() <= 1.0 + 0.02, "φ({u}) norm > 1: {}", phi.norm());
  }
}

#[test]
fn carr_madan_price_is_positive() {
  let pricer = HestonStochCorrPricer::new(
    100.0, 0.03, 100.0, 0.04, 2.0, 0.04, 0.3, -0.7, 5.0, -0.5, 0.2, 0.3, 0.5,
  );
  let call = pricer.price_call_carr_madan();
  assert!(call > 0.0, "call price should be positive: {call}");
  let (call2, put) = pricer.calculate_call_put();
  assert!((call - call2).abs() < 1e-10);
  assert!(put > 0.0, "put price should be positive: {put}");
}

#[test]
fn put_call_parity() {
  let pricer = HestonStochCorrPricer::new(
    100.0, 0.05, 95.0, 0.04, 2.0, 0.04, 0.3, -0.7, 5.0, -0.5, 0.2, 0.3, 0.5,
  );
  let (call, put) = pricer.calculate_call_put();
  let tau = pricer.tau().unwrap();
  // C - P = S·exp(-qτ) - K·exp(-rτ)
  let parity_rhs = pricer.s - pricer.k * (-pricer.r * tau).exp();
  let parity_lhs = call - put;
  assert!(
    (parity_lhs - parity_rhs).abs() < 0.5,
    "put-call parity violated: C-P={parity_lhs:.4}, S-K·e^(-rτ)={parity_rhs:.4}"
  );
}

/// Regression: dividend yield must enter the log-stock drift via `(r - q)`,
/// not be silently dropped. Pre-fix, the ChF used `iu * r` in the drift
/// while put-call parity used the q-discounted forward, producing
/// mutually-inconsistent call/put prices for q > 0.
#[test]
fn put_call_parity_with_dividend_yield() {
  let mut pricer = HestonStochCorrPricer::new(
    100.0, 0.05, 95.0, 0.04, 2.0, 0.04, 0.3, -0.7, 5.0, -0.5, 0.2, 0.3, 0.5,
  );
  pricer.q = Some(0.03); // 3% dividend yield
  let (call, put) = pricer.calculate_call_put();
  let tau = pricer.tau().unwrap();
  let q = pricer.q.unwrap();
  // C - P = S·exp(-qτ) - K·exp(-rτ)
  let parity_rhs = pricer.s * (-q * tau).exp() - pricer.k * (-pricer.r * tau).exp();
  let parity_lhs = call - put;
  assert!(
    (parity_lhs - parity_rhs).abs() < 0.5,
    "put-call parity with q={q} violated: C-P={parity_lhs:.4} vs S·e^(-qτ)-K·e^(-rτ)={parity_rhs:.4}"
  );
}

/// Regression: `HscmModel::price_call` must thread `q` to the underlying
/// Carr-Madan pricer. Pre-fix, `_q` was discarded so calling
/// `model.price_call(s, k, r, q=0.05, tau)` produced the q=0 price.
#[test]
fn hscm_model_pricer_uses_dividend_yield() {
  use crate::traits::ModelPricer;
  let model = HscmModel {
    v0: 0.04,
    kappa_v: 2.0,
    theta_v: 0.04,
    sigma_v: 0.3,
    rho0: -0.7,
    kappa_r: 5.0,
    mu_r: -0.5,
    sigma_r: 0.2,
    rho2: 0.3,
  };
  let s = 100.0;
  let k = 100.0;
  let r = 0.05;
  let tau = 0.5;
  let p_no_div = model.price_call(s, k, r, 0.0, tau);
  let p_with_div = model.price_call(s, k, r, 0.05, tau);
  // ATM call must be cheaper with positive dividend yield (forward shift down).
  assert!(
    p_with_div < p_no_div - 0.1,
    "HscmModel must respect dividend yield: q=0 → {p_no_div:.4}, q=0.05 → {p_with_div:.4}"
  );
}

#[test]
fn reduces_to_heston_when_sigma_r_zero() {
  let pricer = HestonStochCorrPricer::new(
    100.0, 0.03, 95.0, 0.04, 2.0, 0.04, 0.3, -0.7, 5.0, -0.7, 1e-10, 0.0, 0.5,
  );
  let call = pricer.price_call_carr_madan();
  assert!(call > 5.0 && call < 30.0, "unexpected call price: {call}");
}

#[test]
fn compare_with_standard_heston() {
  use crate::pricing::heston::HestonPricer;

  let rho = -0.7;
  let kappa = 2.0;
  let theta = 0.04;
  let sigma = 0.3;
  let v0 = 0.04;
  let s = 100.0;
  let r = 0.03;
  let k = 100.0;
  let tau = 0.5;

  let heston = HestonPricer::new(
    s,
    v0,
    k,
    r,
    None,
    rho,
    kappa,
    theta,
    sigma,
    Some(0.0),
    Some(tau),
    None,
    None,
  );
  let (h_call, _) = heston.calculate_call_put();

  // HSCM with σ_r ≈ 0 should be close to Heston
  let hscm = HestonStochCorrPricer::new(
    s, r, k, v0, kappa, theta, sigma, rho,   // rho0 = constant Heston rho
    10.0,  // kappa_r (high = fast reversion to mu_r)
    rho,   // mu_r = same as rho
    1e-10, // sigma_r ≈ 0
    0.0,   // rho2 = 0
    tau,
  );
  let hscm_call = hscm.price_call_carr_madan();

  println!(
    "Heston call: {h_call:.4}, HSCM call: {hscm_call:.4}, diff: {:.4}",
    (h_call - hscm_call).abs()
  );
  // They won't match exactly due to the affine approximation in HSCM,
  // but should be within a few percent
  assert!(
    (h_call - hscm_call).abs() / h_call < 0.15,
    "HSCM should be close to Heston: H={h_call:.4} vs HSCM={hscm_call:.4}"
  );
}

#[test]
fn price_multiple_strikes() {
  let pricer = HestonStochCorrPricer::new(
    100.0, 0.03, 100.0, 0.04, 2.0, 0.04, 0.3, -0.7, 5.0, -0.5, 0.2, 0.3, 0.5,
  );
  // Price at multiple strikes — should be monotonically decreasing for calls
  let strikes = [80.0, 90.0, 100.0, 110.0, 120.0];
  let prices: Vec<f64> = strikes
    .iter()
    .map(|&k| pricer.price_call_at_strike(k))
    .collect();
  for i in 1..prices.len() {
    assert!(
      prices[i] <= prices[i - 1] + 0.01,
      "call prices not monotone: C({})={:.4} > C({})={:.4}",
      strikes[i],
      prices[i],
      strikes[i - 1],
      prices[i - 1]
    );
  }
}
