use num_complex::Complex64;

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
/// disagreed by ~18%. Both are now integrated to convergence and the residual
/// is the affine approximation alone — 0.18% at τ=0.02, falling to 0.018% at
/// τ=0.002, so the 0.3% band here has ~1.7× headroom on its worst point
/// rather than the 5× slack the old 1% carried.
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
      reldiff < 0.003,
      "HSCM(σ_ρ→0) must match Heston at τ={tau}: Heston={heston_call:.6}, HSCM={hscm_call:.6}, reldiff={reldiff:.4}"
    );
  }
}

/// φ(0) = E[1] = 1 exactly, for every rate and dividend yield.
///
/// Checked at **non-zero** `r`, which is the whole point: this held only at
/// `r = 0` while `char_func_complex` folded a `-rτ` discount into its
/// exponential, and the paper's own query point is `r = 0`, so a version of
/// this test pinned to `PAPER_QUERY` could not see it. At `u = 0` the three
/// Riccati equations have identically-zero solutions, so the assertion is
/// exact rather than approximate — hence 1e-14 and not the old 1e-2.
#[test]
fn char_func_at_zero_is_one() {
  let (s, _k, _r, _q, tau) = PAPER_QUERY;
  let model = paper_model();
  for r in [0.0, 0.05, 0.12] {
    for q in [0.0, 0.03] {
      let phi0 = model.char_func(0.0, s, r, q, tau);
      assert!(
        (phi0 - Complex64::new(1.0, 0.0)).norm() < 1e-14,
        "φ(0) = {phi0} at r={r}, q={q}, expected exactly 1"
      );
    }
  }
}

/// φ(−i) = E\[S_τ\] = S·e^{(r−q)τ}: the risk-neutral martingale condition,
/// with a closed form on the right-hand side and no free parameters.
///
/// This is the sharp guard the suite was missing. A stray `e^{-rτ}` inside
/// the characteristic function scales φ(−i) by exactly that factor, so this
/// assertion fails by `1 − e^{-rτ}` — 3.7% at `r = 0.05, τ = 0.75` and 8.6%
/// at `r = 0.12, τ = 0.75` — against a 1e-12 band. It also pins `q` into the
/// drift, which is what the two former put-call-parity tests were reaching
/// for and could not reach: they derived the put *from* the call by parity,
/// so parity held by construction whatever the call was worth.
#[test]
fn char_func_reproduces_the_forward() {
  let model = paper_model();
  let s = 100.0;
  for r in [0.0, 0.05, 0.12] {
    for q in [0.0, 0.03] {
      for tau in [0.25, 0.75, 1.5] {
        let phi = model.char_func_complex(Complex64::new(0.0, -1.0), s, r, q, tau);
        let forward = s * ((r - q) * tau).exp();
        assert!(
          (phi.re - forward).abs() / forward < 1e-12 && phi.im.abs() < 1e-9,
          "φ(−i) = {phi} must equal the forward {forward} at r={r}, q={q}, τ={tau}"
        );
      }
    }
  }
}

/// |φ(u)| ≤ 1 for real `u`, and — the sharper half — |φ(u)| does not depend
/// on `r` at all. The rate enters only through `iu·(r−q)` in the `dA` ODE,
/// which for real `u` is purely imaginary and so rotates φ's phase without
/// touching its modulus. A discount factor folded into φ would instead scale
/// the modulus by `e^{-rτ}`, breaking this by 4.2e-3 at `r = 0.05, τ = 1/12`.
///
/// The modulus bound is tightened from the old `1 + 0.02` to `1 + 1e-12`:
/// the observed maximum over `u ∈ (0, 50]` is 0.99940, so the affine
/// approximation is not straining the bound and 2% of slack bought nothing.
#[test]
fn char_func_is_finite_and_bounded() {
  let (s, _k, _r, q, tau) = PAPER_QUERY;
  let model = paper_model();
  for u in [0.1, 1.0, 5.0, 10.0, 20.0] {
    let reference = model.char_func(u, s, 0.0, q, tau).norm();
    for r in [0.0, 0.05, 0.12] {
      let phi = model.char_func(u, s, r, q, tau);
      assert!(phi.re.is_finite() && phi.im.is_finite(), "φ({u}) = {phi}");
      assert!(
        phi.norm() <= 1.0 + 1e-12,
        "φ({u}) norm > 1 at r={r}: {}",
        phi.norm()
      );
      assert!(
        (phi.norm() - reference).abs() < 1e-12,
        "|φ({u})| must not depend on r: {} at r={r} vs {reference} at r=0",
        phi.norm()
      );
    }
  }
}

#[test]
fn carr_madan_price_is_positive() {
  let (s, k, r, q, tau) = PAPER_QUERY;
  let call = paper_model().price_call_carr_madan(s, k, r, q, tau);
  assert!(call > 0.0, "call price must be positive, got {call}");
  assert!(call < s, "call price must be below spot, got {call}");
}

/// `(S·e^{−qτ} − K·e^{−rτ})⁺ ≤ C ≤ S·e^{−qτ}`: the model-free no-arbitrage
/// band. This replaces the two former `put_call_parity` tests, which were
/// structurally vacuous — [`HestonStochCorrPricer::call_put`] *derives* the
/// put from the call by parity, so `C − P = S·e^{−qτ} − K·e^{−rτ}` held by
/// construction no matter what the call was worth, and a 0.5 absolute band
/// on an identity is not a measurement.
///
/// The lower bound is a real constraint on the call and is what a double
/// discount breaks: scaling `C` by `e^{−rτ}` pushes it under the intrinsic
/// forward by `1 − e^{−rτ}` wherever that bound is tight, which is 1.2% at
/// τ=0.25 and 4.9% at τ=1 — 12× to 49× past the 1e-3 band asserted here.
/// The worst violation actually observed on this grid is 7.1e-5.
///
/// The grid's deepest strike is `K = 20`. Below that the α=1.25 damping
/// multiplies the inversion by `K^{−α}` — 316× at `K = 0.01` — and the raw
/// call leaves the band in both directions, so a bound this tight would be
/// asserting the quadrature rather than the model. See
/// [`hscm_put_is_parity_and_is_floored_at_zero`].
#[test]
fn call_respects_no_arbitrage_bounds() {
  let m = paper_model();
  let (s, r, q) = (100.0, 0.05, 0.02);
  for tau in [0.25, 0.75, 1.0] {
    for k in [20.0, 50.0, 80.0, 95.0] {
      let call = m.price_call_carr_madan(s, k, r, q, tau);
      let lower = (s * (-q * tau).exp() - k * (-r * tau).exp()).max(0.0);
      let upper = s * (-q * tau).exp();
      assert!(
        call >= lower - 1e-3 * lower.max(1.0),
        "call {call} below intrinsic forward {lower} at K={k}, τ={tau}"
      );
      assert!(
        call <= upper + 1e-3 * upper,
        "call {call} above discounted spot {upper} at K={k}, τ={tau}"
      );
    }
  }
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

/// HSCM with the correlation process frozen against standard Heston at one
/// ATM point. The residual is the paper's affine approximation — Lemma 3.1
/// linearises √v around `m = √(θ_v − σ_v²/(8κ_v))`, so the two models do not
/// coincide even at σ_ρ → 0 — and it grows with τ and |moneyness|.
///
/// The band was 15%, which is not a cross-implementation check so much as a
/// promise that nothing is infinite. At this query the approximation gap is
/// 2.47%, so 3% is asserted instead: ~20% headroom on a residual that is
/// deterministic, not stochastic.
///
/// 2.47% is *larger* than the 0.95% this test saw before the double-discount
/// fix, and that is the expected direction. The discount error ran at
/// −1.49% here and the approximation gap at +2.47%; they partially cancelled
/// to +0.98%. The old number was two errors agreeing to disagree, which is
/// precisely how a 15% band lets a systematic 1.5% bias live indefinitely.
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

  assert!(
    (h_call - hscm_call).abs() / h_call < 0.03,
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

/// Prices at the paper's parameter set and `(s, k, r, q, tau) = (100, 105,
/// 0.05, 0.02, 0.75)`.
///
/// These goldens moved once, deliberately, when the double discount came
/// out of `char_func_complex`. Until then `exp(-r * tau)` was applied both
/// inside the characteristic function and again by the Carr-Madan transform
/// in `price_call_carr_madan`, so every price was low by exactly
/// `1 - exp(-r * tau)` — 3.68% here, and identically zero at the source
/// paper's `r = 0`, which is how it survived. Each call below is its
/// predecessor times `e^{rτ}` to within 1 ulp, except `K = 110` which
/// differs by 2.1e-12 because the quadrature's stopping rule is relative to
/// a running total that the rescaling perturbs.
///
/// Verified against an independent reimplementation — DOP853 for the Riccati
/// system in place of fixed-step RK4, adaptive Gauss-Kronrod for the
/// inversion in place of tanh-sinh — which agrees to 1.2e-4 relative, the
/// discretisation floor between the two schemes. The tighter evidence is
/// structural and lives in [`char_func_reproduces_the_forward`]: φ(−i)
/// reproduces `S·e^{(r−q)τ}` to 1e-15, which the pre-fix function missed by
/// the full 3.68%.
#[test]
fn hscm_model_pricer_goldens() {
  let m = paper_model();
  let (s, k, r, q, tau) = GOLDEN_QUERY;

  // q = 0, the shape the pre-query struct defaulted to.
  let (c0, p0) = m.call_put(s, k, r, 0.0, tau);
  assert!((c0 - 4.82802365321209).abs() < TOL, "q=0 call {c0}");
  assert!((p0 - 5.96343751389837).abs() < TOL, "q=0 put {p0}");

  let (call, put) = m.call_put(s, k, r, q, tau);
  assert!((call - 4.082634367358097).abs() < TOL, "call {call}");
  assert!((put - 6.706854267738123).abs() < TOL, "put {put}");
  assert_eq!(m.price_call(s, k, r, q, tau), call);
  assert_eq!(m.price_put(s, k, r, q, tau), put);

  // Inverts a given price for a vol and reads none of the model's own
  // parameters, so the discount fix leaves it where it was.
  let iv = m.implied_volatility(4.0, s, k, r, q, tau, OptionType::Call);
  assert!((iv - 0.15110131862455398).abs() < TOL, "iv {iv}");

  // The former `price_call_at_strike(110.0)`, which cloned the pricer with
  // a new strike; a strike is now just a different argument.
  let at_110 = m.price_call_carr_madan(s, 110.0, r, q, tau);
  assert!((at_110 - 2.365177912984835).abs() < TOL, "K=110 {at_110}");
}

/// This model's carry factor really is `e^{-qτ}`, so the trait's vanilla
/// put-call parity is mathematically right here. The override exists to
/// keep the `max(0)` floor the pre-query `calculate_call_put` applied to
/// both legs, which the default does not have.
///
/// The floor cannot be demonstrated by pinning one strike, and the previous
/// `assert_eq!(m.price_put(s, 1.0, r, q, tau), 0.0)` was not evidence that
/// it worked. That assertion passed only because the double discount dragged
/// the deep-ITM call 3.7% under `S·e^{−qτ} − K·e^{−rτ}`, making the
/// unfloored parity ≈ −3.6; with the discount corrected the same call comes
/// out at 97.526 against a bound of 97.548, so the residual is −0.022 and
/// the floor fires on a quadrature artifact two orders of magnitude smaller.
///
/// There is no strike where the floor fires for a *model* reason: a European
/// put is worth a strictly positive amount at every finite strike, so the
/// exact unfloored parity is never negative. The floor exists for a
/// numerical reason instead. `price_call_carr_madan` divides the inversion
/// by `K^α` with α = 1.25, so as `K → 0` absolute quadrature error is
/// amplified — 316× at `K = 0.01` — and the raw call wanders off the
/// intrinsic bound in *both* directions (at τ=0.5, K=0.01 it overshoots to
/// 103.26, above spot). Picking whichever strike happens to land negative
/// would be pinning that artifact, not testing the floor.
///
/// So this asserts the floor's actual contract — no leg of `call_put` is
/// ever negative — across a grid that spans the region where the raw
/// inversion is known to break, and *counts* the points the floor rescued so
/// the test cannot quietly go vacuous. The count is checked against 3, which
/// the three structural rescues at `K = 0.01` supply on their own
/// (unfloored −20.95, −5.98, −75.37 at τ = 0.25, 0.75, 1.0); those are
/// K^{−α} amplification, orders of magnitude above cross-arch noise, so
/// their sign is stable in a way the marginal −1e-4 cases are not.
#[test]
fn hscm_put_is_parity_and_is_floored_at_zero() {
  let m = paper_model();
  let (s, k, r, q, tau) = GOLDEN_QUERY;
  let (call, put) = m.call_put(s, k, r, q, tau);
  let parity = call - s * (-q * tau).exp() + k * (-r * tau).exp();
  assert!((put - parity).abs() < TOL, "put {put} vs parity {parity}");

  let mut rescued = 0usize;
  for t in [0.25, 0.75, 1.0] {
    for kk in [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 200.0, 400.0] {
      let (c, p) = m.call_put(s, kk, r, q, t);
      assert!(
        c >= 0.0 && p >= 0.0,
        "negative price at K={kk}, τ={t}: call={c}, put={p}"
      );

      // `c` is already the floored call, so this is exactly the value the
      // trait's unfloored parity default would have returned for the put.
      let unfloored = c - s * (-q * t).exp() + kk * (-r * t).exp();
      if unfloored < 0.0 {
        rescued += 1;
        assert_eq!(
          p, 0.0,
          "floor must fire at K={kk}, τ={t}: unfloored parity {unfloored:e}"
        );
      } else {
        assert!(
          (p - unfloored).abs() < TOL,
          "unfloored put must pass through at K={kk}, τ={t}: {p} vs {unfloored}"
        );
      }
    }
  }
  assert!(
    rescued >= 3,
    "floor never fired, so the override is untested: {rescued} rescues"
  );
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
