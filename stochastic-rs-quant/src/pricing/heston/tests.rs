use super::*;
use crate::calendar::Maturity;
use crate::instruments::equity::EuropeanOption;
use crate::pricing::engines::AnalyticHestonEngine;
use crate::traits::Greeks;
use crate::traits::PricingEngine;
use crate::traits::PricingResult;

const P: HestonParams = HestonParams {
  v0: 0.04,
  kappa: 1.5,
  theta: 0.04,
  sigma: 0.3,
  rho: -0.7,
};

fn price(v0: f64, k: f64, sigma: f64, tau: f64) -> f64 {
  // Long-run variance θ = v0 for these references.
  HestonPricer::new(HestonParams::new(v0, 1.5, v0, sigma, -0.7))
    .price_call(100.0, k, 0.05, 0.0, tau)
}

/// 30-digit mpmath inversions of the same integrand to `∞`; the tail only decays past
/// `φ ~ 1/√(vτ)`, so a fixed upper limit under-prices the short-dated rows by 15-35%.
#[test]
fn short_dated_matches_converged_reference() {
  // (v0, K, σ, τ, converged call)
  let cases = [
    (0.04, 100.0, 0.30, 0.02, 1.1775156238),
    (0.01, 100.0, 0.20, 0.03, 0.7682676476),
    (0.04, 100.0, 0.30, 1.00, 10.361869021),
  ];
  for (v0, k, sigma, tau, expected) in cases {
    let c = price(v0, k, sigma, tau);
    assert!(
      (c - expected).abs() < 1e-9,
      "Heston call at v0={v0}, K={k}, σ={sigma}, τ={tau}: got {c}, converged {expected}"
    );
  }
}

/// `HestonPricer` applies no floor, so a truncated integral shows as a negative price here; the
/// exact calls are `8.2e-14` and `9.89e-9`.
#[test]
fn deep_otm_short_dated_stays_within_the_quadrature_tolerance_of_zero() {
  for (v0, k, sigma, tau) in [(0.04, 150.0, 0.50, 0.10), (0.04, 110.0, 0.30, 0.01)] {
    let c = price(v0, k, sigma, tau);
    assert!(
      c > -1e-8 && c < 1e-7,
      "deep-OTM call at K={k}, τ={tau}: {c}"
    );
  }
}

#[test]
fn heston_single_price() {
  let heston = HestonPricer::new(HestonParams::new(0.05, 5.0, 0.05, 0.5, -0.8));
  let CallPut { call, put } = heston.price_call_put(100.0, 90.0, 0.03, 0.02, 0.5);
  assert!(call.is_finite() && call > 0.0);
  assert!(put.is_finite() && put > 0.0);
}

#[test]
fn analytic_initial_variance_vega_matches_a_resolved_centered_difference() {
  let pricer = HestonPricer::new(HestonParams::new(0.05, 5.0, 0.1, 0.5, -0.8));
  let (s, k, r, q, tau) = (100.0, 90.0, 0.03, 0.02, 0.5);
  let bump = 1e-3;
  let mut up = pricer;
  up.params.v0 += bump;
  let mut down = pricer;
  down.params.v0 -= bump;
  let finite_difference =
    (up.price_call(s, k, r, q, tau) - down.price_call(s, k, r, q, tau)) / (2.0 * bump);
  let vega = pricer.initial_variance_vega(s, k, r, q, tau);

  assert!((vega - finite_difference).abs() < 1e-4);
}

#[test]
fn heston_implied_volatility() {
  let heston = HestonPricer::new(HestonParams::new(0.05, 5.0, 0.05, 0.5, -0.8));
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
  let heston = HestonPricer::new(HestonParams::new(0.04, 2.0, 0.04, 0.3, -0.9));

  let CallPut { call, put } = heston.price_call_put(100.0, 100.0, 0.05, 0.0, 5.0);
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
  HestonPricer::new(P)
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
/// agree on the calendar `-∂P/∂τ` sign convention
/// [`GreeksExt::theta`](crate::traits::GreeksExt::theta)'s own doc mandates,
/// so no single
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
  let opt = EuropeanOption::new(100.0, ot, Maturity::Tau(ATM_TAU));
  let params = P;
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
  let degenerate_v0 = HestonPricer::new(HestonParams::new(0.0, 1.5, 0.04, 0.3, -0.7));
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

  let ok_v0 = HestonPricer::new(P);
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

/// `HestonPricer::new` rejects invalid parameters but admits `v0 == 0`, `theta == 0`, `|rho| == 1`
/// and a violated Feller condition.
mod construction_validation;

/// `E₂[∫₀^τ v] = τ(v0·ℓ + κθτ·s)` against mpmath at 40 digits across `|x| = 1e-3`, `x = κτ`:
/// `ℓ = (1 − e⁻ˣ)/x` alone at `θ = 0`, `s = (x − 1 + e⁻ˣ)/x²` alone at `v0 = 0`.
#[test]
fn the_integrated_variance_mean_holds_its_digits_at_any_b_tau() {
  // (x, ℓ(x), s(x))
  let cases = [
    (1e-6, 0.9999995000001667, 0.499999833333375),
    (9.9e-5, 0.9999505016334596, 0.4999835004083669),
    (1.0225e-3, 0.9994889242065078, 0.4998296268871867),
    (-1.1e-3, 1.0005502017221373, 0.5001833837610937),
    (1e-2, 0.9950166250831947, 0.49833749168053576),
    (-1e-5, 1.0000050000166667, 0.5000016666708333),
    (-0.5, 1.2974425414002564, 0.5948850828005126),
    (2.0, 0.43233235838169365, 0.28383382080915315),
    // The scan's worst: 4.23e-13 with macOS's expm1, 1.6e-14 with glibc's. Past the switch an ulp
    // of expm1 moves `s` by 4.2e-13 relative, so the band admits any expm1 within two ulps.
    (-1.0114e-3, 1.0005058705314434, 0.5001686092973714),
  ];
  for (x, level, slope) in cases {
    // `b₂ = κ + λ` takes the sign of `x`; `κ` stays positive, so `κθ ≥ 0`.
    let b = 1.5_f64.copysign(x);
    let tau = x / b;
    let mean = |v0: f64, theta: f64| {
      HestonPricer::new(HestonParams::new(v0, 1.5, theta, 0.3, -0.7))
        .with_vol_risk_premium(b - 1.5)
        .mean_integrated_variance(2, tau)
    };
    let got_level = mean(1.0, 0.0) / tau;
    let got_slope = mean(0.0, 1.0) / (1.5 * tau * tau);
    assert!(
      (got_level / level - 1.0).abs() < 1e-12,
      "x = {x}: level {got_level} vs {level}"
    );
    assert!(
      (got_slope / slope - 1.0).abs() < 1e-12,
      "x = {x}: slope {got_slope} vs {slope}"
    );
  }
}

/// The control variance is the model's own `−2c₁`, one function for both, bit for bit.
#[test]
fn the_control_variance_is_minus_twice_the_first_cumulant() {
  for (kappa, tau) in [
    (1.5, 1e-6),
    (1.5, 0.3),
    (1.5, 2.0),
    (0.5, 1e-3),
    (-0.4, 3.0),
  ] {
    let theta = if kappa < 0.0 { 0.0 } else { 0.09 };
    let params = HestonParams::new(0.04, kappa, theta, 0.3, -0.7);
    let control = HestonPricer::new(params).mean_integrated_variance(2, tau);
    let c1 = crate::pricing::fourier::heston_cumulants(&params, tau).c1;
    assert_eq!(control, -2.0 * c1, "kappa = {kappa}, tau = {tau}");
  }
}

/// `P₂`'s `D` is the chf's `factor_cd` at `φ` bit for bit, and `P₁`'s at `φ − i` to rounding: one
/// Riccati exponent behind both.
#[test]
fn the_integrand_shares_the_chf_riccati_exponent() {
  let params = HestonParams::new(0.04, 1.5, 0.04, 0.3, -0.7);
  let pricer = HestonPricer::new(params);
  let factor_d = |xi: Complex64| crate::pricing::fourier::factor_cd(&params, 0.75, xi).1;
  for phi in [1e-8, 0.3, 7.0, 150.0] {
    let (_, d2) = pricer.f_and_d(2, phi, 0.75, 100.0, 0.05, 0.02);
    assert_eq!(d2, factor_d(Complex64::new(phi, 0.0)), "phi = {phi}");
  }
  for phi in [0.3, 7.0, 150.0] {
    let (_, d1) = pricer.f_and_d(1, phi, 0.75, 100.0, 0.05, 0.02);
    let want = factor_d(Complex64::new(phi, -1.0));
    assert!(
      (d1 - want).norm() <= 1e-14 * want.norm(),
      "phi = {phi}: {d1} vs {want}"
    );
  }
}

const TOL: f64 = 1e-12;

/// Goldens at `(s, k, r, q, τ) = (100, 105, 0.05, 0.02, 0.75)` and
/// `HestonParams::new(0.04, 2.0, 0.04, 0.3, -0.7)`.
#[test]
fn heston_model_pricer_matches_pre_refactor_goldens() {
  let m = HestonPricer::new(HestonParams::new(0.04, 2.0, 0.04, 0.3, -0.7));
  let (s, k, r, q, tau) = (100.0, 105.0, 0.05, 0.02, 0.75);
  let ot = OptionType::Call;

  let CallPut { call, put } = m.price_call_put(s, k, r, q, tau);
  assert!((call - 5.229927051858397).abs() < TOL, "call {call}");
  assert!((put - 7.854146952238423).abs() < TOL, "put {put}");
  assert_eq!(m.price_call(s, k, r, q, tau), call);
  assert_eq!(m.price_put(s, k, r, q, tau), put);

  let iv = m.implied_volatility(8.0, s, k, r, q, tau, ot);
  assert!((iv - 0.26891374755614555).abs() < TOL, "iv {iv}");
  // The vega integral starts at `phi = 1e-8`; from 0, 30 digits by mpmath, it is `44.2102926490`.
  let v0_vega = m.initial_variance_vega(s, k, r, q, tau);
  assert!(
    (v0_vega - 44.21029252905521).abs() < TOL,
    "v0 vega {v0_vega}"
  );

  let want = [
    0.5179825268957217,
    0.025354655903697676,
    17.684117011622085,
    -5.490198593349759,
    34.92624529677357,
    0.18739798181670153,
    -0.13335185400364935,
    45.78517450630034,
    2.628564235038766,
  ];
  // Finite-difference stencils amplify the price's last bits past `TOL`, and an independent
  // reference moves as much, so every Greek but the analytic `vega` gets the wider band.
  let got = m.greeks(s, k, r, q, tau, ot).as_array();
  for (i, name) in Greeks::COMPONENT_NAMES.iter().enumerate() {
    let tol = if *name == "vega" { TOL } else { 1e-7 };
    assert!(
      (got[i] - want[i]).abs() < tol,
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
    assert_eq!(g.volga, m.volga(s, k, r, q, tau), "volga");
    assert_eq!(g.veta, m.veta(s, k, r, q, tau), "veta");
    assert_ne!(g.volga, g.veta, "volga and veta must not be the same value");
  }
}

/// This model's carry factor really is `e^{-qτ}`, so the trait's vanilla
/// put-call parity is mathematically right here — the override exists only
/// so the returned put is bit-identical to the closed form the pre-query
/// `calculate_call_put().1` produced.
#[test]
fn heston_price_put_matches_parity_but_is_the_closed_form() {
  let m = HestonPricer::new(HestonParams::new(0.04, 2.0, 0.04, 0.3, -0.7));
  let (s, k, r, q, tau) = (100.0, 105.0, 0.05, 0.02, 0.75);
  let CallPut { call, put } = m.price_call_put(s, k, r, q, tau);
  let parity = call - s * (-q * tau).exp() + k * (-r * tau).exp();
  assert!((put - parity).abs() < TOL, "put {put} vs parity {parity}");
  assert_eq!(m.price_put(s, k, r, q, tau), put);
}

/// Put Greeks became expressible when `option_type` moved into the query.
/// The two must differ exactly by the derivative of the parity terms:
/// `delta_P = delta_C - e^{-qτ}` and `gamma_P = gamma_C`.
#[test]
fn heston_put_greeks_differ_from_call_by_the_parity_terms() {
  let m = HestonPricer::new(HestonParams::new(0.04, 2.0, 0.04, 0.3, -0.7));
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
  let m = HestonPricer::new(HestonParams::new(0.04, 2.0, 0.04, 0.3, -0.7));
  for &tau in &[0.25, 0.5, 1.0] {
    let mut prev = f64::INFINITY;
    for &k in &[90.0, 100.0, 110.0] {
      let c = m.price_call(100.0, k, 0.05, 0.02, tau);
      assert!(c.is_finite() && c < prev, "call must fall in strike");
      prev = c;
    }
  }
}
