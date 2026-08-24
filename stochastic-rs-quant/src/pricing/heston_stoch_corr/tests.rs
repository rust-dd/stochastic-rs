use super::*;
use crate::OptionType;
use crate::traits::ModelPricer;

/// Parameters from Table 2 in Teng et al.
fn paper_model() -> HestonStochCorrPricer {
  HestonStochCorrPricer::new(
    0.02, // v0
    2.1,  // kappa_v
    0.03, // theta_v
    0.2,  // sigma_v
    -0.4, // rho0
    3.4,  // kappa_r
    -0.6, // mu_r
    0.1,  // sigma_r
    0.4,  // rho2
  )
}

/// The paper's own query point: ATM, zero rate, one month.
const PAPER_QUERY: (f64, f64, f64, f64, f64) = (100.0, 100.0, 0.0, 0.0, 1.0 / 12.0);

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
    let heston = HestonPricer::new(v0, rho, kappa, theta, sigma, Some(0.0));
    let heston_call = heston.call_put(s, s, r, 0.0, tau).0;
    let hscm = HestonStochCorrPricer::new(v0, kappa, theta, sigma, rho, 10.0, rho, 1e-10, 0.0);
    let hscm_call = hscm.price_call_carr_madan(s, s, r, 0.0, tau);
    let reldiff = (heston_call - hscm_call).abs() / heston_call;
    assert!(
      reldiff < 0.01,
      "HSCM(σ_ρ→0) must match Heston at τ={tau}: Heston={heston_call:.6}, HSCM={hscm_call:.6}, reldiff={reldiff:.4}"
    );
  }
}

#[test]
fn char_func_at_zero_is_one() {
  let (s, _k, r, q, tau) = PAPER_QUERY;
  let phi0 = paper_model().char_func(0.0, s, r, q, tau);
  assert!(
    (phi0.norm() - 1.0).abs() < 0.01,
    "φ(0) = {phi0}, expected ~1.0"
  );
}

#[test]
fn char_func_is_finite_and_bounded() {
  let (s, _k, r, q, tau) = PAPER_QUERY;
  let model = paper_model();
  for u in [0.1, 1.0, 5.0, 10.0, 20.0] {
    let phi = model.char_func(u, s, r, q, tau);
    assert!(phi.re.is_finite() && phi.im.is_finite(), "φ({u}) = {phi}");
    assert!(phi.norm() <= 1.0 + 0.02, "φ({u}) norm > 1: {}", phi.norm());
  }
}

#[test]
fn carr_madan_price_is_positive() {
  let (s, k, r, q, tau) = PAPER_QUERY;
  let call = paper_model().price_call_carr_madan(s, k, r, q, tau);
  assert!(call > 0.0, "call price must be positive, got {call}");
  assert!(call < s, "call price must be below spot, got {call}");
}

#[test]
fn put_call_parity() {
  let model = HestonStochCorrPricer::new(0.04, 2.0, 0.04, 0.3, -0.7, 5.0, -0.5, 0.2, 0.3);
  let (s, k, r, q, tau) = (100.0, 95.0, 0.05, 0.0, 0.5);
  let (call, put) = model.call_put(s, k, r, q, tau);
  // C - P = S·exp(-qτ) - K·exp(-rτ)
  let parity_rhs = s - k * (-r * tau).exp();
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
  let model = HestonStochCorrPricer::new(0.04, 2.0, 0.04, 0.3, -0.7, 5.0, -0.5, 0.2, 0.3);
  let (s, k, r, q, tau) = (100.0, 95.0, 0.05, 0.03, 0.5);
  let (call, put) = model.call_put(s, k, r, q, tau);
  // C - P = S·exp(-qτ) - K·exp(-rτ)
  let parity_rhs = s * (-q * tau).exp() - k * (-r * tau).exp();
  let parity_lhs = call - put;
  assert!(
    (parity_lhs - parity_rhs).abs() < 0.5,
    "put-call parity with q={q} violated: C-P={parity_lhs:.4} vs S·e^(-qτ)-K·e^(-rτ)={parity_rhs:.4}"
  );
}

/// Regression: `price_call` must thread `q` to the Carr-Madan inversion.
/// Pre-fix (on the former `HscmModel`, whose fields and behaviour this type
/// absorbed) `_q` was discarded, so `price_call(s, k, r, q = 0.05, tau)`
/// produced the `q = 0` price.
#[test]
fn hscm_model_pricer_uses_dividend_yield() {
  let model = HestonStochCorrPricer::new(0.04, 2.0, 0.04, 0.3, -0.7, 5.0, -0.5, 0.2, 0.3);
  let (s, k, r, tau) = (100.0, 100.0, 0.05, 0.5);
  let p_no_div = model.price_call(s, k, r, 0.0, tau);
  let p_with_div = model.price_call(s, k, r, 0.05, tau);
  // ATM call must be cheaper with positive dividend yield (forward shift down).
  assert!(
    p_with_div < p_no_div - 0.1,
    "must respect dividend yield: q=0 → {p_no_div:.4}, q=0.05 → {p_with_div:.4}"
  );
}

#[test]
fn reduces_to_heston_when_sigma_r_zero() {
  let model = HestonStochCorrPricer::new(0.04, 2.0, 0.04, 0.3, -0.7, 5.0, -0.7, 1e-10, 0.0);
  let call = model.price_call_carr_madan(100.0, 95.0, 0.03, 0.0, 0.5);
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

  let heston = HestonPricer::new(v0, rho, kappa, theta, sigma, Some(0.0));
  let (h_call, _) = heston.call_put(s, k, r, 0.0, tau);

  // HSCM with σ_r ≈ 0 should be close to Heston
  let hscm = HestonStochCorrPricer::new(
    v0, kappa, theta, sigma, rho,   // rho0 = constant Heston rho
    10.0,  // kappa_r (high = fast reversion to mu_r)
    rho,   // mu_r = same as rho
    1e-10, // sigma_r ≈ 0
    0.0,   // rho2 = 0
  );
  let hscm_call = hscm.price_call_carr_madan(s, k, r, 0.0, tau);

  // They won't match exactly due to the affine approximation in HSCM,
  // but should be within a few percent
  assert!(
    (h_call - hscm_call).abs() / h_call < 0.15,
    "HSCM should be close to Heston: H={h_call:.4} vs HSCM={hscm_call:.4}"
  );
}

#[test]
fn price_multiple_strikes() {
  let model = HestonStochCorrPricer::new(0.04, 2.0, 0.04, 0.3, -0.7, 5.0, -0.5, 0.2, 0.3);
  // Price at multiple strikes — should be monotonically decreasing for calls
  let strikes = [80.0, 90.0, 100.0, 110.0, 120.0];
  let prices: Vec<f64> = strikes
    .iter()
    .map(|&k| model.price_call(100.0, k, 0.03, 0.0, 0.5))
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

/// Cross-arch tolerance: the goldens come from an adaptive quadrature over
/// an RK4-integrated ODE, so the last bits differ between aarch64-darwin
/// and CI's ubuntu x86_64.
const TOL: f64 = 1e-12;

const GOLDEN_QUERY: (f64, f64, f64, f64, f64) = (100.0, 105.0, 0.05, 0.02, 0.75);

/// Captured from `PricerExt::calculate_call_put()`, `implied_volatility`
/// and `price_call_at_strike` **before** the `ModelPricer` reshape, at the
/// paper's parameter set and `(s, k, r, q, tau) = (100, 105, 0.05, 0.02,
/// 0.75)`. The reshape (and the `HscmModel` merge) is an API change only.
///
/// **These values are known to be wrong, and deliberately so.** The pricer
/// discounts twice: `exp(-r * tau)` is applied inside `char_func_complex`
/// (`cf.rs`, in the returned exponential) and again in
/// `price_call_carr_madan` (`pricer.rs`). Every price here is therefore low
/// by exactly `1 - exp(-r * tau)` — 3.68% at this query. It is invisible at
/// the source paper's `r = 0`, which is why it survived.
///
/// The bug predates the `ModelPricer` reshape. It is pinned rather than
/// fixed so the reshape could be verified against the behaviour it actually
/// replaced; fixing it is a separate, deliberate change that must move these
/// numbers. The fix is to drop `-r * tau` from the `cf.rs` exponential and
/// leave `pricer.rs` alone — and it will also break
/// `hscm_price_put_matches_parity_but_is_floored`, whose zero-floor assertion
/// currently fires only because the double discount pushes the deep-ITM call
/// below parity.
#[test]
fn hscm_model_pricer_matches_pre_refactor_goldens() {
  let m = paper_model();
  let (s, k, r, q, tau) = GOLDEN_QUERY;

  // q = 0, the shape the pre-query struct defaulted to.
  let (c0, p0) = m.call_put(s, k, r, 0.0, tau);
  assert!((c0 - 4.650325431397971).abs() < TOL, "q=0 call {c0}");
  assert!((p0 - 5.7857392920842585).abs() < TOL, "q=0 put {p0}");

  let (call, put) = m.call_put(s, k, r, q, tau);
  assert!((call - 3.9323706322344987).abs() < TOL, "call {call}");
  assert!((put - 6.556590532614521).abs() < TOL, "put {put}");
  assert_eq!(m.price_call(s, k, r, q, tau), call);
  assert_eq!(m.price_put(s, k, r, q, tau), put);

  let iv = m.implied_volatility(4.0, s, k, r, q, tau, OptionType::Call);
  assert!((iv - 0.15110131862455398).abs() < TOL, "iv {iv}");

  // The former `price_call_at_strike(110.0)`, which cloned the pricer with
  // a new strike; a strike is now just a different argument.
  let at_110 = m.price_call_carr_madan(s, 110.0, r, q, tau);
  assert!((at_110 - 2.278126162705615).abs() < TOL, "K=110 {at_110}");
}

/// This model's carry factor really is `e^{-qτ}`, so the trait's vanilla
/// put-call parity is mathematically right here. The override exists to
/// keep the `max(0)` floor the pre-query `calculate_call_put` applied to
/// both legs, which the default does not have.
#[test]
fn hscm_price_put_matches_parity_but_is_floored() {
  let m = paper_model();
  let (s, k, r, q, tau) = GOLDEN_QUERY;
  let (call, put) = m.call_put(s, k, r, q, tau);
  let parity = call - s * (-q * tau).exp() + k * (-r * tau).exp();
  assert!((put - parity).abs() < TOL, "put {put} vs parity {parity}");

  // Deep in the money for the call is deep out of the money for the put:
  // the unfloored parity value goes negative, and the floor must catch it.
  let deep = m.price_put(s, 1.0, r, q, tau);
  assert_eq!(deep, 0.0, "deep-OTM put must be floored, got {deep}");
}

/// The capability the reshape exists for: one model, a whole grid.
#[test]
fn hscm_one_model_prices_a_grid() {
  let m = paper_model();
  for &tau in &[0.25, 0.5, 1.0] {
    let mut prev = f64::INFINITY;
    for &k in &[90.0, 100.0, 110.0] {
      let c = m.price_call(100.0, k, 0.05, 0.02, tau);
      assert!(c.is_finite() && c < prev, "call must fall in strike");
      prev = c;
    }
  }
}
