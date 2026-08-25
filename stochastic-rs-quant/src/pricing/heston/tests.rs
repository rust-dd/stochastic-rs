use super::*;
use crate::instruments::equity::EuropeanOption;
use crate::pricing::engines::AnalyticHestonEngine;
use crate::pricing::engines::HestonStaticParams;
use crate::traits::Greeks;
use crate::traits::PricingEngine;
use crate::traits::PricingResult;

fn price(v0: f64, k: f64, sigma: f64, tau: f64) -> f64 {
  // Long-run variance θ = v0 for these references.
  HestonPricer::new(v0, -0.7, 1.5, v0, sigma, Some(0.0)).price_call(100.0, k, 0.05, 0.0, tau)
}

/// Short-dated / low-variance options must match the converged Fourier
/// integral. The former fixed `φ_max = 50` truncated a tail that only
/// decays past `φ ~ 1/√(vτ)`, under-pricing these by 15-35%. Converged
/// references are from a `scipy.integrate.quad` inversion to `∞`, validated
/// against the repo's own long-dated `HESTON_REF`. The τ=1 case pins that
/// the already-accurate long-dated regime is unchanged.
#[test]
fn short_dated_matches_converged_reference() {
  // (v0, K, σ, τ, converged call)
  let cases = [
    (0.04, 100.0, 0.30, 0.02, 1.177515),
    (0.01, 100.0, 0.20, 0.03, 0.768268),
    (0.04, 100.0, 0.30, 1.00, 10.361856),
  ];
  for (v0, k, sigma, tau, expected) in cases {
    let c = price(v0, k, sigma, tau);
    assert!(
      (c - expected).abs() < 2e-3,
      "Heston call at v0={v0}, K={k}, σ={sigma}, τ={tau}: got {c}, converged {expected}"
    );
  }
}

/// Deep-OTM short-dated calls must be non-negative and ~0, not the negative
/// (arbitrage-violating) or spuriously-positive values the fixed integration
/// bound produced. Pre-fix: τ=0.1/K=150 → −0.0347, τ=0.01/K=110 → +0.062.
/// This exercises `HestonPricer` directly (no `.max(0.0)` clamp), so it pins
/// the integral itself, not a downstream floor — the root cause behind the
/// negative model prices in calibration issue #14.
#[test]
fn deep_otm_short_dated_non_negative() {
  for (v0, k, sigma, tau) in [(0.04, 150.0, 0.50, 0.10), (0.04, 110.0, 0.30, 0.01)] {
    let c = price(v0, k, sigma, tau);
    assert!(
      c > -1e-3 && c < 1e-2,
      "deep-OTM call at K={k}, τ={tau} must be non-negative and ~0, got {c}"
    );
  }
}

#[test]
fn heston_single_price() {
  let heston = HestonPricer::new(0.05, -0.8, 5.0, 0.05, 0.5, Some(0.0));
  let (call, put) = heston.call_put(100.0, 90.0, 0.03, 0.02, 0.5);
  assert!(call.is_finite() && call > 0.0);
  assert!(put.is_finite() && put > 0.0);
}

#[test]
fn analytic_initial_variance_vega_matches_a_resolved_centered_difference() {
  let pricer = HestonPricer::new(0.05, -0.8, 5.0, 0.1, 0.5, Some(0.0));
  let (s, k, r, q, tau) = (100.0, 90.0, 0.03, 0.02, 0.5);
  let bump = 1e-3;
  let mut up = pricer;
  up.v0 += bump;
  let mut down = pricer;
  down.v0 -= bump;
  let finite_difference =
    (up.price_call(s, k, r, q, tau) - down.price_call(s, k, r, q, tau)) / (2.0 * bump);
  let (call, put) = pricer.call_put_initial_variance_vega(s, k, r, q, tau);

  assert!((call - finite_difference).abs() < 1e-4);
  assert_eq!(call, put);
}

#[test]
fn heston_implied_volatility() {
  let heston = HestonPricer::new(0.05, -0.8, 5.0, 0.05, 0.5, Some(0.0));
  let (s, k, r, q, tau) = (100.0, 90.0, 0.03, 0.02, 1.0);

  let call = heston.price_call(s, k, r, q, tau);
  let iv = heston.implied_volatility(call, s, k, r, q, tau, OptionType::Call);
  assert!(iv.is_finite() && iv > 0.0, "implied vol {iv}");
}

/// Long-maturity / high-|ρ| regression: the Albrecher-Mayer-Schoutens-Tistaert
/// (2007) "Little Heston Trap" form must keep the principal-branch logarithm
/// stable for T = 5y, ρ = -0.9. Original Heston (1993) form develops a
/// branch-cut discontinuity in this regime; the Trap form does not.
#[test]
fn heston_little_trap_long_maturity_high_rho() {
  let heston = HestonPricer::new(0.04, -0.9, 2.0, 0.04, 0.3, Some(0.0));

  let (call, put) = heston.call_put(100.0, 100.0, 0.05, 0.0, 5.0);
  assert!(
    call.is_finite() && call > 0.0,
    "Heston Trap form should give finite positive call at T=5y, ρ=-0.9: {call}"
  );
  assert!(
    put.is_finite() && put > 0.0,
    "Heston Trap form should give finite positive put at T=5y, ρ=-0.9: {put}"
  );

  // Sanity check: put-call parity.
  let parity = call - put;
  let expected = 100.0 * 1.0 - 100.0 * (-0.05_f64 * 5.0).exp();
  assert!(
    (parity - expected).abs() < 0.5,
    "Put-call parity violated at T=5y: C-P={parity}, expected≈{expected}"
  );
}

const ATM_TAU: f64 = 0.1;
const ATM_QUERY: (f64, f64, f64, f64, f64) = (100.0, 100.0, 0.05, 0.0, ATM_TAU);

fn atm_pricer() -> HestonPricer {
  HestonPricer::new(0.04, -0.7, 1.5, 0.04, 0.3, None)
}

/// `HestonPricer`'s Greeks vs. `AnalyticHestonEngine::finite_diff_greeks`
/// on the identical parameter set.
///
/// `delta`/`gamma`/`vega` are checked tight (`rel < 1e-6`) at `bump = 1e-4`,
/// which makes the engine's `h_S = S·bump` and `h_v0 = v0·bump` coincide
/// exactly with this crate's own `h_S = S·1e-4` / `h_v0 = v0·1e-4` steps —
/// an apples-to-apples comparison of the same central-difference formula at
/// the same step, rather than two independently-tuned schemes that merely
/// converge to the same limit.
///
/// `theta` needs two separate checks, because `finite_diff_greeks`'s theta
/// is a *one-sided backward* difference (`price_at(τ)` and
/// `price_at(τ - h_τ)` only, no `τ + h_τ` term) — an asymmetric (`O(bump)`,
/// not this crate's `O(bump²)`) truncation error — even though both paths
/// now agree on the calendar `-∂P/∂τ` sign convention
/// [`GreeksExt::theta`](crate::traits::GreeksExt::theta)'s own doc mandates
/// ([`HestonPricer::theta`] and
/// [`AnalyticHestonEngine::finite_diff_greeks`] used to disagree in sign;
/// the engine's raw output has since been flipped to match), so no single
/// engine `bump` makes `direct.theta() == engine_theta` to a tight
/// tolerance while *also* reflecting the engine's real default:
/// - `theta_tight` (`bump = 1e-7`) verifies `theta` converges to the
///   analytic `-∂P/∂τ` limit as `bump → 0` — not that it agrees with the
///   engine's *actual configured* precision. (`bump` below ~`1e-8` stops
///   helping and then hurts: the underlying `p_j` characteristic-function
///   integral is itself only converged to a `1e-8` relative tolerance, so a
///   too-small bump starts differencing quadrature noise instead of
///   signal.)
/// - `theta_default` (`bump = 1e-3`, `AnalyticHestonEngine::new`'s own
///   out-of-the-box default) pins the real-world gap a caller hitting the
///   engine with its default settings would actually see. Tolerance is
///   `O(bump)`-derived: a one-sided backward difference has leading error
///   `≈ (h_τ/2)·|∂²P/∂τ²|`, i.e. relative error `≈ (h_τ/2)·|∂²P/∂τ² / ∂P/∂τ|`.
///   The curvature ratio `|∂²P/∂τ² / ∂P/∂τ|` is read off empirically from
///   the `bump = 1e-4` case (h_τ = τ·1e-4 = 1e-5), whose still-`O(h_τ)`
///   residual against `theta_tight` was ≈ 2.1e-5 relative, giving
///   `|∂²P/∂τ²/∂P/∂τ| ≈ 2·2.1e-5/1e-5 ≈ 4.2`; at the default `bump = 1e-3`
///   (`h_τ = τ·1e-3 = 1e-4`, 10× larger), the same-order error should scale
///   ≈ linearly to ≈ 2.1e-4 — `5e-4` below leaves headroom without hiding a
///   regression.
#[test]
fn heston_greeks_match_engine_bumps() {
  let direct = atm_pricer();
  let (s, k, r, q, tau) = ATM_QUERY;
  let ot = OptionType::Call;
  let opt = EuropeanOption::new_tau(100.0, ot, ATM_TAU);
  let params = HestonStaticParams::new(0.04, 1.5, 0.04, 0.3, -0.7);
  let mut engine = AnalyticHestonEngine::with_constants(100.0, 0.05, 0.0, params);

  engine.bump = 1e-4;
  let central_diff_greeks = engine.calculate(&opt).greeks().unwrap();

  let cases = [
    (
      "delta",
      direct.delta(s, k, r, q, tau, ot),
      central_diff_greeks.delta,
    ),
    (
      "gamma",
      direct.gamma(s, k, r, q, tau),
      central_diff_greeks.gamma,
    ),
    (
      "vega",
      direct.vega(s, k, r, q, tau),
      central_diff_greeks.vega,
    ),
  ];
  for (name, mine, engine_val) in cases {
    let rel = (mine - engine_val).abs() / engine_val.abs().max(1e-8);
    assert!(
      rel < 1e-6,
      "{name}: mine={mine}, engine={engine_val}, rel={rel}"
    );
  }

  engine.bump = 1e-7;
  let theta_tight = engine.calculate(&opt).greeks().unwrap().theta;
  let rel_tight =
    (direct.theta(s, k, r, q, tau, ot) - theta_tight).abs() / theta_tight.abs().max(1e-8);
  assert!(
    rel_tight < 1e-6,
    "theta (tight-bump engine): mine={}, engine_tight={theta_tight}, rel={rel_tight}",
    direct.theta(s, k, r, q, tau, ot)
  );

  engine.bump = 1e-3;
  let theta_default = engine.calculate(&opt).greeks().unwrap().theta;
  let rel_default =
    (direct.theta(s, k, r, q, tau, ot) - theta_default).abs() / theta_default.abs().max(1e-8);
  assert!(
    rel_default < 5e-4,
    "theta (engine default bump=1e-3): mine={}, engine_default={theta_default}, rel={rel_default}",
    direct.theta(s, k, r, q, tau, ot)
  );
}

/// Every Greek is finite; gamma is positive (convexity); call delta sits
/// strictly inside (0, 1).
#[test]
fn heston_greeks_all_finite() {
  let (s, k, r, q, tau) = ATM_QUERY;
  let g = atm_pricer().greeks(s, k, r, q, tau, OptionType::Call);
  for (name, v) in Greeks::COMPONENT_NAMES.iter().zip(g.as_array()) {
    assert!(v.is_finite(), "{name} is not finite: {v}");
  }
  assert!(g.gamma > 0.0, "gamma should be positive: {}", g.gamma);
  assert!(
    g.delta > 0.0 && g.delta < 1.0,
    "call delta should be in (0,1): {}",
    g.delta
  );
}

/// `vega`/`vanna`/`volga`/`veta` guard `v0 <= 0` (the `σ = √v0` chain rule
/// they use is undefined there); `theta`/`charm`/`veta` separately guard a
/// `tau` that is non-finite or not safely larger than `H_TAU`. Both guards
/// return `NaN` rather than a wrong finite number — this pins that the
/// guards actually fire, backing the Greeks module doc's claim.
#[test]
fn heston_greeks_nan_at_degenerate_inputs() {
  let degenerate_v0 = HestonPricer::new(0.0, -0.7, 1.5, 0.04, 0.3, Some(0.0));
  let (s, k, r, q) = (100.0, 100.0, 0.05, 0.0);
  assert!(
    degenerate_v0.vega(s, k, r, q, 1.0).is_nan(),
    "vega must be NaN at v0 = 0"
  );
  assert!(
    degenerate_v0.vanna(s, k, r, q, 1.0).is_nan(),
    "vanna must be NaN at v0 = 0"
  );
  assert!(
    degenerate_v0.volga(s, k, r, q, 1.0).is_nan(),
    "volga must be NaN at v0 = 0"
  );
  assert!(
    degenerate_v0.veta(s, k, r, q, 1.0).is_nan(),
    "veta must be NaN at v0 = 0"
  );

  let ok_v0 = HestonPricer::new(0.04, -0.7, 1.5, 0.04, 0.3, Some(0.0));
  let ot = OptionType::Call;
  assert!(
    ok_v0.theta(s, k, r, q, 0.0, ot).is_nan(),
    "theta must be NaN at tau = 0"
  );
  assert!(
    ok_v0.charm(s, k, r, q, 0.0, ot).is_nan(),
    "charm must be NaN at tau = 0"
  );
  assert!(
    ok_v0.veta(s, k, r, q, 0.0).is_nan(),
    "veta must be NaN at tau = 0"
  );
}

/// A *negative* `v0` is invalid input, not a degenerate-but-admissible
/// state, so the four volatility-space Greeks panic on it rather than
/// returning the `NaN` that `v0 == 0` earns. Pinned per accessor because each
/// guards independently; the `expected` anchor is on the parameter name and
/// value, so an unrelated panic (an index slip, an arithmetic overflow) fails
/// the test instead of silently satisfying it.
///
/// `heston_greeks_nan_at_degenerate_inputs` pins the `v0 == 0` half and is
/// deliberately unchanged by this split.
mod negative_v0_panics {
  use super::*;

  fn negative_v0() -> HestonPricer {
    HestonPricer::new(-0.01, -0.7, 1.5, 0.04, 0.3, Some(0.0))
  }

  #[test]
  #[should_panic(expected = "v0 must be non-negative (got -0.01)")]
  fn vega_rejects_negative_v0() {
    negative_v0().vega(100.0, 100.0, 0.05, 0.0, 1.0);
  }

  #[test]
  #[should_panic(expected = "v0 must be non-negative (got -0.01)")]
  fn vanna_rejects_negative_v0() {
    negative_v0().vanna(100.0, 100.0, 0.05, 0.0, 1.0);
  }

  #[test]
  #[should_panic(expected = "v0 must be non-negative (got -0.01)")]
  fn volga_rejects_negative_v0() {
    negative_v0().volga(100.0, 100.0, 0.05, 0.0, 1.0);
  }

  #[test]
  #[should_panic(expected = "v0 must be non-negative (got -0.01)")]
  fn veta_rejects_negative_v0() {
    negative_v0().veta(100.0, 100.0, 0.05, 0.0, 1.0);
  }

  /// The `v0` guard runs before the `tau` guard, so a negative `v0` panics
  /// even at a `tau` that would independently have returned `NaN`. Without
  /// this, a caller could mask invalid input behind a degenerate maturity.
  #[test]
  #[should_panic(expected = "v0 must be non-negative (got -0.01)")]
  fn veta_rejects_negative_v0_before_degenerate_tau() {
    negative_v0().veta(100.0, 100.0, 0.05, 0.0, 0.0);
  }

  /// Greeks that do not route through the `σ = √v0` chain rule keep working
  /// at a negative `v0` exactly as before — the split must not widen into a
  /// blanket model-parameter check.
  #[test]
  fn delta_and_rho_are_unaffected_by_the_guard() {
    let m = negative_v0();
    let ot = OptionType::Call;
    assert!(m.delta(100.0, 100.0, 0.05, 0.0, 1.0, ot).is_finite());
    assert!(m.rho(100.0, 100.0, 0.05, 0.0, 1.0, ot).is_finite());
  }
}

const TOL: f64 = 1e-12;

/// Captured from `PricerExt::calculate_call_put()`, `implied_volatility`,
/// `calculate_call_put_initial_variance_vega` and the `GreeksExt`
/// aggregate **before** the `ModelPricer` reshape, at
/// `(s, k, r, q, tau) = (100, 105, 0.05, 0.02, 0.75)` and
/// `(v0, rho, kappa, theta, sigma, lambda) = (0.04, -0.7, 2, 0.04, 0.3,
/// None)`. The reshape is an API change only, so none of these move.
#[test]
fn heston_model_pricer_matches_pre_refactor_goldens() {
  let m = HestonPricer::new(0.04, -0.7, 2.0, 0.04, 0.3, None);
  let (s, k, r, q, tau) = (100.0, 105.0, 0.05, 0.02, 0.75);
  let ot = OptionType::Call;

  let (call, put) = m.call_put(s, k, r, q, tau);
  assert!((call - 5.229917528523217).abs() < TOL, "call {call}");
  assert!((put - 7.854137428903243).abs() < TOL, "put {put}");
  assert_eq!(m.price_call(s, k, r, q, tau), call);
  assert_eq!(m.price_put(s, k, r, q, tau), put);

  let iv = m.implied_volatility(8.0, s, k, r, q, tau, ot);
  assert!((iv - 0.26891374755614555).abs() < TOL, "iv {iv}");
  let v0_vega = m.call_put_initial_variance_vega(s, k, r, q, tau).0;
  assert!(
    (v0_vega - 44.210172705008645).abs() < TOL,
    "v0 vega {v0_vega}"
  );

  let want = [
    0.5179826481178651,
    0.0253545923811771,
    17.684069082003457,
    -5.490187207612961,
    34.926261530188185,
    0.18739775236397804,
    -0.13335162307726023,
    45.78493484996492,
    2.6285888232280286,
  ];
  let got = m.greeks(s, k, r, q, tau, ot).as_array();
  for (i, name) in Greeks::COMPONENT_NAMES.iter().enumerate() {
    assert!(
      (got[i] - want[i]).abs() < TOL,
      "{name}: got {}, want {}",
      got[i],
      want[i]
    );
  }
}

/// The aggregate must be the nine accessors and nothing else — in
/// particular `volga`/`veta` must not be transposed, which is a mapping
/// that has no other pin.
#[test]
fn heston_greeks_aggregate_matches_accessors() {
  let m = atm_pricer();
  let (s, k, r, q, tau) = ATM_QUERY;
  for ot in [OptionType::Call, OptionType::Put] {
    let g = m.greeks(s, k, r, q, tau, ot);
    assert_eq!(g.delta, m.delta(s, k, r, q, tau, ot), "delta");
    assert_eq!(g.gamma, m.gamma(s, k, r, q, tau), "gamma");
    assert_eq!(g.vega, m.vega(s, k, r, q, tau), "vega");
    assert_eq!(g.theta, m.theta(s, k, r, q, tau, ot), "theta");
    assert_eq!(g.rho, m.rho(s, k, r, q, tau, ot), "rho");
    assert_eq!(g.vanna, m.vanna(s, k, r, q, tau), "vanna");
    assert_eq!(g.charm, m.charm(s, k, r, q, tau, ot), "charm");
    assert_eq!(g.volga, m.volga(s, k, r, q, tau), "volga is vomma");
    assert_eq!(g.veta, m.veta(s, k, r, q, tau), "veta is dvega_dtime");
    assert_ne!(g.volga, g.veta, "volga and veta must not be the same value");
  }
}

/// This model's carry factor really is `e^{-qτ}`, so the trait's vanilla
/// put-call parity is mathematically right here — the override exists only
/// so the returned put is bit-identical to the closed form the pre-query
/// `calculate_call_put().1` produced.
#[test]
fn heston_price_put_matches_parity_but_is_the_closed_form() {
  let m = HestonPricer::new(0.04, -0.7, 2.0, 0.04, 0.3, None);
  let (s, k, r, q, tau) = (100.0, 105.0, 0.05, 0.02, 0.75);
  let (call, put) = m.call_put(s, k, r, q, tau);
  let parity = call - s * (-q * tau).exp() + k * (-r * tau).exp();
  assert!((put - parity).abs() < TOL, "put {put} vs parity {parity}");
  assert_eq!(m.price_put(s, k, r, q, tau), put);
}

/// Put Greeks became expressible when `option_type` moved into the query.
/// The two must differ exactly by the derivative of the parity terms:
/// `delta_P = delta_C - e^{-qτ}` and `gamma_P = gamma_C`.
#[test]
fn heston_put_greeks_differ_from_call_by_the_parity_terms() {
  let m = HestonPricer::new(0.04, -0.7, 2.0, 0.04, 0.3, None);
  let (s, k, r, q, tau) = (100.0, 105.0, 0.05, 0.02, 0.75);
  let call_delta = m.delta(s, k, r, q, tau, OptionType::Call);
  let put_delta = m.delta(s, k, r, q, tau, OptionType::Put);
  assert!(
    (put_delta - (call_delta - (-q * tau).exp())).abs() < 1e-8,
    "put delta {put_delta} vs call delta {call_delta}"
  );
  assert!(put_delta < 0.0, "put delta must be negative: {put_delta}");
}

/// The capability the reshape exists for: one model, a whole grid.
#[test]
fn heston_one_model_prices_a_grid() {
  let m = HestonPricer::new(0.04, -0.7, 2.0, 0.04, 0.3, None);
  for &tau in &[0.25, 0.5, 1.0] {
    let mut prev = f64::INFINITY;
    for &k in &[90.0, 100.0, 110.0] {
      let c = m.price_call(100.0, k, 0.05, 0.02, tau);
      assert!(c.is_finite() && c < prev, "call must fall in strike");
      prev = c;
    }
  }
}
