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

/// With the correlation frozen (σ_ρ → 0) the model collapses to Heston at ATM; the residual is the
/// affine approximation alone, 0.18 % at τ = 0.02 against a 0.3 % band.
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

/// φ(0) = 1 exactly at a non-zero rate: the three Riccati solutions vanish at `u = 0`, so the
/// tolerance is 1e-14.
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

/// |φ(u)| ≤ 1 for real `u` and does not depend on `r`, which only rotates φ's phase through
/// `iu(r − q)`; a discount folded into φ would scale the modulus by `e^{−rτ}`.
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

/// The no-arbitrage band `(S e^{−qτ} − K e^{−rτ})⁺ ≤ C ≤ S e^{−qτ}` to 1e-6 relative on nine
/// queries down to `K = 0.01`, where the `K^{−α}` prefactor amplifies any inversion error 316-fold.
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

/// Calls are non-increasing and convex in the strike on a ladder from `K = 0.01`, which catches an
/// inversion error that stays inside the no-arbitrage band.
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
    // Deep in the money the call is `S e^{−qτ} − K e^{−rτ}` to eleven digits, so convexity is
    // borderline; the band absorbs the inversion's own ~1e-7.
    assert!(
      slope_hi >= slope_lo - 1e-7 * slope_lo.abs().max(1.0),
      "call must be convex in strike at K={}: slopes {slope_lo} then {slope_hi}",
      strikes[i]
    );
  }
}

/// `price_call` passes `q` through to the Carr-Madan inversion.
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

/// Frozen-correlation HSCM against Heston at one ATM point: the 2.47 % gap is Lemma 3.1's affine
/// approximation (√v linearised at `m = √(θ_v − σ_v²/(8κ_v))`), asserted within 3 %.
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

/// Goldens at the paper's parameters and `(100, 105, 0.05, 0.02, 0.75)`, within 1e-8 of a
/// Dormand–Prince Riccati solve with Gauss–Kronrod through both Carr-Madan and Gil-Pelaez.
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

/// The put is the call's parity floored at zero: no leg is ever negative, and a put whose
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

/// A non-finite market input, `tau` included (NaN is its missing-data value), prices to NaN: the
/// quadrature keeps the NaN and the floor tests for it before clamping.
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
