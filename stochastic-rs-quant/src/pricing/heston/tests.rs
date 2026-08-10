use super::*;
use crate::instruments::equity::EuropeanOption;
use crate::pricing::engines::AnalyticHestonEngine;
use crate::pricing::engines::HestonStaticParams;
use crate::traits::Greeks;
use crate::traits::PricingEngine;
use crate::traits::PricingResult;

fn price(v0: f64, k: f64, sigma: f64, tau: f64) -> f64 {
  // Long-run variance θ = v0 for these references.
  HestonPricer::new(
    100.0,
    v0,
    k,
    0.05,
    Some(0.0),
    -0.7,
    1.5,
    v0,
    sigma,
    Some(0.0),
    Some(tau),
    None,
    None,
  )
  .calculate_call_put()
  .0
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
  let heston = HestonPricer::new(
    100.0,
    0.05,
    90.0,
    0.03,
    Some(0.02),
    -0.8,
    5.0,
    0.05,
    0.5,
    Some(0.0),
    Some(0.5),
    None,
    None,
  );

  let (call, put) = heston.calculate_call_put();
  println!("Call Price: {}, Put Price: {}", call, put);
}

#[test]
fn analytic_initial_variance_vega_matches_a_resolved_centered_difference() {
  let pricer = HestonPricer::new(
    100.0,
    0.05,
    90.0,
    0.03,
    Some(0.02),
    -0.8,
    5.0,
    0.1,
    0.5,
    Some(0.0),
    Some(0.5),
    None,
    None,
  );
  let bump = 1e-3;
  let mut up = pricer.clone();
  up.v0 += bump;
  let mut down = pricer.clone();
  down.v0 -= bump;
  let finite_difference = (up.calculate_call_put().0 - down.calculate_call_put().0) / (2.0 * bump);
  let (call, put) = pricer.calculate_call_put_initial_variance_vega();

  assert!((call - finite_difference).abs() < 1e-4);
  assert_eq!(call, put);
}

#[test]
fn heston_implied_volatility() {
  let heston = HestonPricer::new(
    100.0,
    0.05,
    90.0,
    0.03,
    Some(0.02),
    -0.8,
    5.0,
    0.05,
    0.5,
    Some(0.0),
    Some(1.0),
    None,
    None,
  );

  let (call, ..) = heston.calculate_call_put();
  let iv = heston.implied_volatility(call, OptionType::Call);
  println!("Implied Volatility: {}", iv);
}

/// Long-maturity / high-|ρ| regression: the Albrecher-Mayer-Schoutens-Tistaert
/// (2007) "Little Heston Trap" form must keep the principal-branch logarithm
/// stable for T = 5y, ρ = -0.9. Original Heston (1993) form develops a
/// branch-cut discontinuity in this regime; the Trap form does not.
#[test]
fn heston_little_trap_long_maturity_high_rho() {
  let heston = HestonPricer::new(
    100.0,
    0.04,
    100.0,
    0.05,
    Some(0.0),
    -0.9, // high-|ρ|
    2.0,
    0.04,
    0.3,
    Some(0.0),
    Some(5.0), // T = 5y
    None,
    None,
  );

  let (call, put) = heston.calculate_call_put();
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

fn atm_pricer() -> HestonPricer {
  HestonPricer::new(
    100.0,
    0.04,
    100.0,
    0.05,
    Some(0.0),
    -0.7,
    1.5,
    0.04,
    0.3,
    None,
    Some(ATM_TAU),
    None,
    None,
  )
}

/// `HestonPricer`'s `GreeksExt` impl vs. `AnalyticHestonEngine::finite_diff_greeks`
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
/// [`GreeksExt::theta`]'s own doc mandates ([`HestonPricer::theta`] and
/// [`AnalyticHestonEngine::finite_diff_greeks`] used to disagree in sign;
/// the engine's raw output has since been flipped to match), so no single
/// engine `bump` makes `direct.theta() == engine_theta` to a tight
/// tolerance while *also* reflecting the engine's real default:
/// - `theta_tight` (`bump = 1e-7`) verifies `direct.theta()` converges to
///   the analytic `-∂P/∂τ` limit as `bump → 0` — not that it agrees with
///   the engine's *actual configured* precision. (`bump` below ~`1e-8` stops
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
  let opt = EuropeanOption::new_tau(100.0, OptionType::Call, ATM_TAU);
  let params = HestonStaticParams::new(0.04, 1.5, 0.04, 0.3, -0.7);
  let mut engine = AnalyticHestonEngine::with_constants(100.0, 0.05, 0.0, params);

  engine.bump = 1e-4;
  let central_diff_greeks = engine.calculate(&opt).greeks().unwrap();

  let cases = [
    ("delta", direct.delta(), central_diff_greeks.delta),
    ("gamma", direct.gamma(), central_diff_greeks.gamma),
    ("vega", direct.vega(), central_diff_greeks.vega),
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
  let rel_tight = (direct.theta() - theta_tight).abs() / theta_tight.abs().max(1e-8);
  assert!(
    rel_tight < 1e-6,
    "theta (tight-bump engine): mine={}, engine_tight={theta_tight}, rel={rel_tight}",
    direct.theta()
  );

  engine.bump = 1e-3;
  let theta_default = engine.calculate(&opt).greeks().unwrap().theta;
  let rel_default = (direct.theta() - theta_default).abs() / theta_default.abs().max(1e-8);
  assert!(
    rel_default < 5e-4,
    "theta (engine default bump=1e-3): mine={}, engine_default={theta_default}, rel={rel_default}",
    direct.theta()
  );
}

/// Every Greek is finite; gamma is positive (convexity); call delta sits
/// strictly inside (0, 1).
#[test]
fn heston_greeks_all_finite() {
  let g = atm_pricer().greeks();
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
