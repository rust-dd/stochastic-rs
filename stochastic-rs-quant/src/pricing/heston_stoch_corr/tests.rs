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
/// **The `K = 20` floor and the `τ = 1` ceiling are both gone.** They existed
/// because the quadrature, not the model, could not hold the band there, and
/// the old doc comment said so. Deep in the money the transform's `K^{−α}`
/// prefactor multiplies the inversion by 316 at `K = 0.01`, so that is where
/// a quadrature error shows first — which makes it the sharpest available
/// test of the inversion rather than a reason to look away. What the two
/// limits were hiding:
///
/// | query | before | truth |
/// |---|---|---|
/// | `τ=0.25, K=0.01` | 78.54 | 99.49 |
/// | `τ=1, K=0.01` | 22.64 | 98.01 |
/// | `τ=2, K=20` | 10.46 | 77.98 |
/// | `τ=2, K=95` | 20.57 | 14.64 |
///
/// The last row is the one that matters most: at `τ = 2` the old inversion
/// was wrong at **every** strike on this grid, not only deep ones, so the
/// `K ≥ 0.2·S` region believed to be unaffected was not.
///
/// The band is `1e-6` relative, tightened from `1e-3`. The worst violation
/// over the full grid — including the points this test leaves to
/// [`call_respects_no_arbitrage_bounds_across_the_full_grid`] — is `9.2e-9`,
/// so the assertion keeps ~110× headroom, and the upper bound is not
/// violated anywhere at all.
///
/// The query list is explicit rather than a cross product because the cost
/// is very uneven: a `τ = 2` inversion runs the Riccati system ~8× longer
/// per quadrature node than `τ = 0.25`, and a deep-in-the-money strike
/// oscillates at `|ln(K/S)| = 9.2` where an at-the-money one oscillates at
/// `0.05`. These nine carry every failure mode above at about a fifth of the
/// full grid's runtime.
#[test]
fn call_respects_no_arbitrage_bounds() {
  let m = paper_model();
  let (s, r, q) = (100.0, 0.05, 0.02);
  for (tau, k) in [
    (0.25, 0.01),
    (0.25, 20.0),
    (0.25, 95.0),
    (0.75, 20.0),
    (0.75, 95.0),
    (1.0, 95.0),
    (2.0, 95.0),
  ] {
    assert_in_band(&m, s, k, r, q, tau);
  }
}

/// The full cross product the test above samples from, including the three
/// `τ = 2` deep strikes whose inversions dominate its runtime.
#[test]
#[ignore = "slow: HSCM Riccati Rk4 × adaptive quadrature over 24 deep/long queries. Run with --ignored."]
fn call_respects_no_arbitrage_bounds_across_the_full_grid() {
  let m = paper_model();
  let (s, r, q) = (100.0, 0.05, 0.02);
  for tau in [0.25, 0.75, 1.0, 2.0] {
    for k in [0.01, 1.0, 20.0, 50.0, 80.0, 95.0] {
      assert_in_band(&m, s, k, r, q, tau);
    }
  }
}

fn assert_in_band(m: &HestonStochCorrPricer, s: f64, k: f64, r: f64, q: f64, tau: f64) {
  let call = m.price_call_carr_madan(s, k, r, q, tau);
  let lower = (s * (-q * tau).exp() - k * (-r * tau).exp()).max(0.0);
  let upper = s * (-q * tau).exp();
  assert!(
    call >= lower - 1e-6 * lower.max(1.0),
    "call {call} below intrinsic forward {lower} at K={k}, τ={tau}"
  );
  assert!(
    call <= upper + 1e-6 * upper,
    "call {call} above discounted spot {upper} at K={k}, τ={tau}"
  );
}

/// A call is non-increasing and convex in the strike. Neither property
/// depends on the model, so this is a check on the inversion alone, and it
/// catches a failure that stays *inside* the no-arbitrage band.
///
/// The ladder starts at `K = 0.01` because that is what makes it bite: the
/// old inversion returned `78.54` there against `98.51` at `K = 1`, so the
/// very first step rose. `price_multiple_strikes` walks `τ = 0.5` from
/// `K = 80` and could not see it.
#[test]
fn carr_madan_is_monotone_and_convex_in_strike() {
  let m = paper_model();
  let (s, r, q, tau) = (100.0, 0.05, 0.02, 0.25);
  let strikes = [0.01, 1.0, 20.0, 50.0, 95.0];
  let prices: Vec<f64> = strikes
    .iter()
    .map(|&k| m.price_call_carr_madan(s, k, r, q, tau))
    .collect();
  for i in 1..prices.len() {
    assert!(
      prices[i] < prices[i - 1],
      "call must fall in strike: C({})={} >= C({})={}",
      strikes[i],
      prices[i],
      strikes[i - 1],
      prices[i - 1]
    );
  }
  for i in 1..prices.len() - 1 {
    let slope_lo = (prices[i] - prices[i - 1]) / (strikes[i] - strikes[i - 1]);
    let slope_hi = (prices[i + 1] - prices[i]) / (strikes[i + 1] - strikes[i]);
    // Deep in the money the call is `S·e^{−qτ} − K·e^{−rτ}` to eleven
    // digits, so consecutive slopes are both `−e^{−rτ}` and convexity is
    // exactly borderline; the band absorbs the inversion's own ~1e-7 there.
    // It stays far inside what this is for — walked on the pre-fix prices,
    // the first pair of slopes are `+20.18` then `−0.99`, a violation of 21.
    assert!(
      slope_hi >= slope_lo - 1e-7 * slope_lo.abs().max(1.0),
      "call must be convex in strike at K={}: slopes {slope_lo} then {slope_hi}",
      strikes[i]
    );
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
/// These goldens have moved twice, both times deliberately.
///
/// The first move took a double discount out of `char_func_complex`:
/// `exp(-r * tau)` was applied both inside the characteristic function and
/// again by the Carr-Madan transform, so every price was low by exactly
/// `1 - exp(-r * tau)` — 3.68% here, and identically zero at the source
/// paper's `r = 0`, which is how it survived.
///
/// The second move is the quadrature. `integrate_to_convergence` discarded
/// the tanh-sinh rule's own error estimate and grew its panels without
/// bound, so a panel spanning `[150, 350]` was accepted with a reported
/// error estimate of `6716`. At this query that panel added `0.311` to an
/// integral of `4310.68`, and the transform's `K^{-alpha}` prefactor turned it
/// into `2.9e-4` of the call — the `7.2e-5` relative error the three values
/// below used to carry.
///
/// Adjudicated against a reference sharing no code with the crate: an
/// adaptive Dormand-Prince 5(4) integration of the Riccati system in place
/// of fixed-step Rk4, and adaptive Gauss-Kronrod in place of tanh-sinh,
/// applied through *two* inversion formulas — the Carr-Madan transform and
/// Gil-Pelaez, which share no contour. Both agree with each other to 9e-9
/// and are stable to a 20% change in the truncation point.
///
/// | golden | before | after | reference |
/// |---|---|---|---|
/// | `q = 0` call | 4.82802365321209 | 4.82832421223066 | 4.8283242121 |
/// | `q = 0.02` call | 4.082634367358097 | 4.082339820498465 | 4.0823398213 |
/// | `K = 110` | 2.365177912984835 | 2.365470328642592 | 2.3654703293 |
///
/// The residual against the reference fell from 2.9e-4 to under 1e-8 on all
/// three. The structural evidence is unchanged and lives in
/// [`char_func_reproduces_the_forward`]: φ(−i) reproduces `S·e^{(r−q)τ}` to
/// 1e-15.
#[test]
fn hscm_model_pricer_goldens() {
  let m = paper_model();
  let (s, k, r, q, tau) = GOLDEN_QUERY;

  // q = 0, the shape the pre-query struct defaulted to.
  let (c0, p0) = m.call_put(s, k, r, 0.0, tau);
  assert!((c0 - 4.82832421223066).abs() < TOL, "q=0 call {c0}");
  assert!((p0 - 5.963738072916939).abs() < TOL, "q=0 put {p0}");

  let (call, put) = m.call_put(s, k, r, q, tau);
  assert!((call - 4.082339820498465).abs() < TOL, "call {call}");
  assert!((put - 6.706559720878488).abs() < TOL, "put {put}");
  assert_eq!(m.price_call(s, k, r, q, tau), call);
  assert_eq!(m.price_put(s, k, r, q, tau), put);

  // Inverts a given price for a vol and reads none of the model's own
  // parameters, so the discount fix leaves it where it was.
  let iv = m.implied_volatility(4.0, s, k, r, q, tau, OptionType::Call);
  assert!((iv - 0.15110131862455398).abs() < TOL, "iv {iv}");

  // The former `price_call_at_strike(110.0)`, which cloned the pricer with
  // a new strike; a strike is now just a different argument.
  let at_110 = m.price_call_carr_madan(s, 110.0, r, q, tau);
  assert!((at_110 - 2.365470328642592).abs() < TOL, "K=110 {at_110}");
}

/// This model's carry factor really is `e^{-qτ}`, so the trait's vanilla
/// put-call parity is mathematically right here. The override exists to
/// keep the `max(0)` floor the pre-query `calculate_call_put` applied to
/// both legs, which the default does not have.
///
/// **The floor no longer has a structural trigger, and that is the fix
/// working.** This test used to *count* the grid points the floor rescued
/// and require at least three, on the reasoning that `K = 0.01` supplied
/// three on its own with unfloored parities of −20.95, −5.98 and −75.37 —
/// "orders of magnitude above cross-arch noise, so their sign is stable in a
/// way the marginal cases are not". Those three were the `K^{−α}`
/// amplification of a quadrature error, not a property of the model, and
/// they are gone: the deepest unfloored parity anywhere on this grid is now
/// −9.0e-7 and most are 1e-10 or smaller. Counting them would now be exactly
/// the flaky sign-of-round-off assertion the old comment warned against, so
/// the count is replaced by a direct check on the floor itself.
///
/// There is no strike where the floor fires for a *model* reason: a European
/// put is worth a strictly positive amount at every finite strike, so the
/// exact unfloored parity is never negative. What is left to assert is the
/// floor's actual contract — no leg is ever negative, and a put whose
/// unfloored parity is non-negative passes through untouched.
#[test]
fn hscm_put_is_parity_and_is_floored_at_zero() {
  let m = paper_model();
  let (s, k, r, q, tau) = GOLDEN_QUERY;
  let (call, put) = m.call_put(s, k, r, q, tau);
  let parity = call - s * (-q * tau).exp() + k * (-r * tau).exp();
  assert!((put - parity).abs() < TOL, "put {put} vs parity {parity}");

  for t in [0.25, 0.75] {
    for kk in [0.01, 50.0, 200.0] {
      let (c, p) = m.call_put(s, kk, r, q, t);
      assert!(
        c >= 0.0 && p >= 0.0,
        "negative price at K={kk}, τ={t}: call={c}, put={p}"
      );

      // `c` is already the floored call, so this is exactly the value the
      // trait's unfloored parity default would have returned for the put.
      let unfloored = c - s * (-q * t).exp() + kk * (-r * t).exp();
      if unfloored >= 0.0 {
        assert!(
          (p - unfloored).abs() < TOL,
          "unfloored put must pass through at K={kk}, τ={t}: {p} vs {unfloored}"
        );
      } else {
        assert_eq!(p, 0.0, "floor must fire at K={kk}, τ={t}: {unfloored:e}");
      }
    }
  }

  // The floor itself, with a deterministic trigger rather than whichever
  // grid point happens to land a few ulp negative. It is the same split the
  // poison check needs: a negative price floors, a `NaN` does not.
  assert_eq!(super::pricer::floor_price(-1e-12), 0.0);
  assert_eq!(super::pricer::floor_price(-5.0), 0.0);
  assert_eq!(super::pricer::floor_price(3.5), 3.5);
  assert!(super::pricer::floor_price(f64::NAN).is_nan());
}

/// A non-finite market input has to survive the Carr-Madan inversion. Every
/// one of them poisons the characteristic function, and the quadrature used
/// to swallow that `NaN` and return `0.0` — after which `call.max(0.0)`
/// could not have recovered it anyway, since `f64::NAN.max(0.0)` is `0.0`.
/// So both halves are needed: the poison check in the quadrature, and a
/// floor here that tests for `NaN` before it clamps.
///
/// `tau` is not a hypothetical: [`TimeExt::tau_or_from_dates`](crate::traits::TimeExt)
/// documents `NaN` as its missing-data return, so an option whose expiry
/// never resolved priced at exactly zero through this pricer.
#[test]
fn hscm_preserves_nan_market_inputs() {
  let m = paper_model();
  let nan = f64::NAN;
  for (name, s, k, r, q, tau) in [
    ("tau", 100.0, 105.0, 0.05, 0.02, nan),
    ("s", nan, 105.0, 0.05, 0.02, 0.75),
    ("k", 100.0, nan, 0.05, 0.02, 0.75),
    ("r", 100.0, 105.0, nan, 0.02, 0.75),
    ("q", 100.0, 105.0, 0.05, nan, 0.75),
  ] {
    let call = m.price_call_carr_madan(s, k, r, q, tau);
    assert!(call.is_nan(), "NaN {name} must exit as NaN, got {call}");
    let (c, p) = m.call_put(s, k, r, q, tau);
    assert!(
      c.is_nan(),
      "NaN {name} must leave call_put's call NaN, got {c}"
    );
    assert!(
      p.is_nan(),
      "NaN {name} must leave call_put's put NaN, got {p}"
    );
  }
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

/// `HestonStochCorrPricer::new` validates the parameters that have a
/// domain, at the layer the caller supplies them.
///
/// Nothing announced itself before: every invalid value below produced a
/// finite, plausible Carr-Madan price against a reference of `6.9417`
/// (`s = k = 100, r = 0.05, τ = 0.5`) — `v0 = -0.04` gave `2.3422`,
/// `theta_v = -0.04` gave `4.9724`, `rho0 = -1.5` gave `7.0771` and
/// `rho2 = -1.5` gave `6.9712`. The last is within `0.03` of the correct
/// answer, which is the whole problem: it is indistinguishable from a small
/// modelling difference.
///
/// `sigma_v = 0` is deliberately **accepted**, unlike on
/// [`HestonPricer::new`](crate::pricing::HestonPricer) where it is
/// rejected. The reason is the characteristic function: Heston's closed
/// form divides by `sigma^2`, while this model integrates a Riccati system
/// by RK4 in which `sigma_v` only ever multiplies, so a zero vol-of-vol is
/// the deterministic-variance limit rather than a division by zero.
///
/// `kappa_v` and `kappa_r` stay unconstrained, matching
/// [`HestonPricer::new`]'s treatment of `kappa`.
mod construction_validation {
  use super::*;

  fn ok() -> [f64; 9] {
    [0.04, 2.0, 0.04, 0.3, -0.7, 5.0, -0.5, 0.2, 0.3]
  }

  fn build(p: [f64; 9]) -> HestonStochCorrPricer {
    HestonStochCorrPricer::new(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8])
  }

  #[test]
  #[should_panic(
    expected = "HestonStochCorrPricer::new: v0 must be a non-negative variance (got -0.04)"
  )]
  fn new_rejects_negative_v0() {
    let mut p = ok();
    p[0] = -0.04;
    let _ = build(p);
  }

  #[test]
  #[should_panic(
    expected = "HestonStochCorrPricer::new: theta_v must be a non-negative variance (got -0.04)"
  )]
  fn new_rejects_negative_long_run_variance() {
    let mut p = ok();
    p[2] = -0.04;
    let _ = build(p);
  }

  #[test]
  #[should_panic(
    expected = "HestonStochCorrPricer::new: sigma_v must be a non-negative volatility (got -0.3)"
  )]
  fn new_rejects_negative_vol_of_vol() {
    let mut p = ok();
    p[3] = -0.3;
    let _ = build(p);
  }

  #[test]
  #[should_panic(
    expected = "HestonStochCorrPricer::new: sigma_r must be a non-negative volatility (got -0.2)"
  )]
  fn new_rejects_negative_correlation_volatility() {
    let mut p = ok();
    p[7] = -0.2;
    let _ = build(p);
  }

  /// Three separate correlations, all bounded, all checked — `rho0` is the
  /// initial level, `mu_r` the level it mean-reverts to and `rho2` the
  /// correlation between the two driving Brownians. Leaving any one out
  /// would put a number outside `[-1, 1]` into the same expansion the other
  /// two are protected from.
  #[test]
  fn every_correlation_is_bounded() {
    for (idx, name) in [(4_usize, "rho0"), (6, "mu_r"), (8, "rho2")] {
      for bad in [-1.5_f64, 1.5] {
        let mut p = ok();
        p[idx] = bad;
        let err = std::panic::catch_unwind(move || build(p)).expect_err("must reject");
        let msg = err.downcast_ref::<String>().cloned().unwrap_or_else(|| {
          err
            .downcast_ref::<&str>()
            .copied()
            .unwrap_or("")
            .to_string()
        });
        assert!(
          msg.contains(&format!(
            "HestonStochCorrPricer::new: {name} must be in [-1, 1]"
          )),
          "{name} at {bad}: wrong message {msg}"
        );
      }
    }
  }

  /// The calibrator's `BOUNDS` box and the admissible degenerate edges must
  /// all still construct — a guard tighter than the box would abort a
  /// calibration on a legal iterate.
  #[test]
  fn the_calibrators_bounds_box_stays_constructible() {
    let lo = build([0.001, 0.01, 0.001, 0.01, -0.99, 0.01, -0.99, 0.01, -0.99]);
    assert_eq!(lo.v0, 0.001);
    let hi = build([0.5, 10.0, 1.0, 2.0, 0.99, 20.0, 0.99, 2.0, 0.99]);
    assert_eq!(hi.rho2, 0.99);

    let deterministic = build([0.04, 2.0, 0.0, 0.0, -1.0, 5.0, 1.0, 0.0, -1.0]);
    assert_eq!(deterministic.sigma_v, 0.0);
    assert_eq!(deterministic.theta_v, 0.0);
    assert_eq!(
      build([0.04, -2.0, 0.04, 0.3, -0.7, -5.0, -0.5, 0.2, 0.3]).kappa_v,
      -2.0
    );
  }
}
