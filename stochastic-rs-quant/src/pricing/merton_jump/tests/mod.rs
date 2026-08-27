use super::*;
use crate::OptionType;
use crate::pricing::bsm::BSMPricer;
use crate::traits::Greeks;

mod construction;
mod degenerate;
mod formula;
mod lambda_zero;

/// `m` (Poisson-series term count) is capped at 20 in these tests:
/// the pre-refactor term loop computed `n!` as a `usize` product, which
/// overflows past `n = 20` — a pre-existing limitation of the price series
/// itself, unrelated to convergence (at `λτ ≤ 0.25` the series is
/// converged to double precision well before `n = 20`).
///
/// `k = 105` is no longer *forced* to be off the money. It was, while the
/// `n = 0` term priced at zero volatility: that made the term a
/// deterministic payoff of `S` with a kink exactly at the forward, which
/// `bumped_price`'s finite difference would have straddled at `K = S`.
/// `σ_0 = d > 0` removes the kink, so every strike is now a smooth region;
/// the value is kept so the goldens below stay comparable across that
/// change.
const S: f64 = 100.0;
const K: f64 = 105.0;
const R: f64 = 0.05;
const Q: f64 = 0.0;
const TAU: f64 = 0.5;
const OT: OptionType = OptionType::Call;

fn merton(lambda: f64, gamma: f64, m: usize) -> Merton1976Pricer {
  Merton1976Pricer::new(0.2, lambda, gamma, m, BSMCoc::Bsm1973)
}

fn bumped_price(m: &Merton1976Pricer, ds: f64, dv: f64, dtau: f64) -> f64 {
  let mut p = *m;
  p.v += dv;
  p.price_option(S + ds, K, R, Q, TAU + dtau, OT)
}

/// λ=0 collapses the Poisson sum to its single `n=0` term (weight 1),
/// degenerating Merton (1976) to plain Black-Scholes at the input
/// volatility — see `Σ_{n=0}^∞ e^{-λT}(λT)^n/n!` at `λ=0`.
#[test]
fn merton_greeks_lambda_zero_equals_bs() {
  let m = merton(0.0, 0.4, 20);
  let bs = BSMPricer::new(m.v, m.b);

  let cases = [
    (
      "delta",
      m.delta(S, K, R, Q, TAU, OT),
      bs.delta(S, K, R, Q, TAU, OT),
    ),
    ("gamma", m.gamma(S, K, R, Q, TAU), bs.gamma(S, K, R, Q, TAU)),
    (
      "vega",
      m.vega(S, K, R, Q, TAU, OT),
      bs.vega(S, K, R, Q, TAU),
    ),
    (
      "theta",
      m.theta(S, K, R, Q, TAU, OT),
      bs.theta(S, K, R, Q, TAU, OT),
    ),
    (
      "rho",
      m.rho(S, K, R, Q, TAU, OT),
      bs.rho(S, K, R, Q, TAU, OT),
    ),
    (
      "vanna",
      m.vanna(S, K, R, Q, TAU, OT),
      bs.vanna(S, K, R, Q, TAU),
    ),
    (
      "charm",
      m.charm(S, K, R, Q, TAU, OT),
      bs.charm(S, K, R, Q, TAU, OT),
    ),
    (
      "volga",
      m.volga(S, K, R, Q, TAU, OT),
      bs.vomma(S, K, R, Q, TAU),
    ),
    (
      "veta",
      m.veta(S, K, R, Q, TAU, OT),
      bs.dvega_dtime(S, K, R, Q, TAU),
    ),
  ];
  for (name, got, want) in cases {
    assert!((got - want).abs() < 1e-10, "{name}: got {got}, want {want}");
  }
}

/// Merton delta/gamma/vega/theta/rho vs. central finite differences of
/// the price itself, independent of whether a given Greek is internally a
/// closed-form series or a finite difference.
///
/// `gamma` is checked at a looser `1e-3` (not `1e-4`): its BSM closed form
/// (`norm_pdf(d1)/(S·v·√τ)`) and a central second-difference of
/// `BSMPricer::call_put` already disagree by several `1e-4`
/// *at a single, non-Merton `BSMPricer` in isolation* — confirmed by
/// checking one term on its own — and the gap does not keep shrinking as
/// the bump size shrinks over several orders of magnitude the way a
/// normal `O(h²)` truncation error would. Second-derivative finite
/// differences are inherently more sensitive to a special function's
/// numerical approximation error than the function value itself, so this
/// is read as a pre-existing `norm_cdf`/`norm_pdf` precision
/// characteristic rather than a defect in either `gamma()`.
#[test]
fn merton_greeks_match_finite_difference() {
  let m = merton(0.5, 0.3, 20);

  let h_s = S * 1e-4;
  let delta_fd = (bumped_price(&m, h_s, 0.0, 0.0) - bumped_price(&m, -h_s, 0.0, 0.0)) / (2.0 * h_s);
  let gamma_fd = (bumped_price(&m, h_s, 0.0, 0.0) - 2.0 * m.price_option(S, K, R, Q, TAU, OT)
    + bumped_price(&m, -h_s, 0.0, 0.0))
    / (h_s * h_s);

  let h_v = m.v * 1e-4;
  let vega_fd = (bumped_price(&m, 0.0, h_v, 0.0) - bumped_price(&m, 0.0, -h_v, 0.0)) / (2.0 * h_v);

  let h_t = 1e-5;
  let theta_fd =
    -(bumped_price(&m, 0.0, 0.0, h_t) - bumped_price(&m, 0.0, 0.0, -h_t)) / (2.0 * h_t);

  let h_r = 1e-5;
  let rho_fd = (m.price_option(S, K, R + h_r, Q, TAU, OT)
    - m.price_option(S, K, R - h_r, Q, TAU, OT))
    / (2.0 * h_r);

  let cases = [
    ("delta", m.delta(S, K, R, Q, TAU, OT), delta_fd, 1e-4),
    ("gamma", m.gamma(S, K, R, Q, TAU), gamma_fd, 1e-3),
    ("vega", m.vega(S, K, R, Q, TAU, OT), vega_fd, 1e-4),
    ("theta", m.theta(S, K, R, Q, TAU, OT), theta_fd, 1e-4),
    ("rho", m.rho(S, K, R, Q, TAU, OT), rho_fd, 1e-4),
  ];
  for (name, analytic, fd, tol) in cases {
    let rel = (analytic - fd).abs() / fd.abs().max(1e-8);
    assert!(rel < tol, "{name}: analytic={analytic}, fd={fd}, rel={rel}");
  }
}

/// All 9 Greeks are finite for a representative `λ > 0` configuration —
/// exercises the finite-difference path (vega/theta/vanna/charm/volga/veta)
/// that the λ=0 test above cannot reach.
#[test]
fn merton_greeks_all_finite() {
  let m = merton(0.5, 0.3, 20);
  let g = m.greeks(S, K, R, Q, TAU, OT);
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

/// All 9 Greeks stay finite at `m = 50` — the crate's Python binding's
/// documented default, and well past the pre-refactor `usize`-factorial
/// overflow threshold (`m ≈ 21`; `(1..=21).product::<usize>()` panics in
/// debug builds, silently wraps in release). Every Greek routes through
/// [`Merton1976Pricer::series_price`], which has no such ceiling — this
/// test is the direct regression check for that.
#[test]
fn merton_greeks_finite_at_m50() {
  let m = merton(0.5, 0.3, 50);
  let g = m.greeks(S, K, R, Q, TAU, OT);
  for (name, v) in Greeks::COMPONENT_NAMES.iter().zip(g.as_array()) {
    assert!(v.is_finite(), "{name} is not finite at m=50: {v}");
  }
  // `call_put` itself (not just the Greeks path) must also survive m=50
  // now that it is routed through `poisson_weight` instead of an integer
  // `n!` — regression check for the overflow this refactor fixed
  // (`(1..=i).product::<usize>()` panics in debug / wraps in release past
  // `i ≈ 21`).
  let (call, put) = m.call_put(S, K, R, Q, TAU);
  assert!(call.is_finite(), "call not finite at m=50: {call}");
  assert!(put.is_finite(), "put not finite at m=50: {put}");
}

/// A second `(lambda, gamma)` pin, guarding
/// [`Merton1976Pricer::poisson_weight`]'s running product against the
/// integer `n!` it replaced (which overflows past `n ≈ 20`): the two
/// accumulation orders are mathematically identical but not guaranteed
/// bit-for-bit equal, so a `1e-12` absolute tolerance leaves headroom for
/// the operation order while still catching a real regression.
///
/// Both values moved with the `σ_n` correction — they were `1.883107` and
/// `4.290648`, computed with the diffusive variance scaled by `n/τ`. The
/// call is adjudicated to `4.401577` by Gil-Pelaez inversion of the Merton
/// characteristic function, which shares no code with the Poisson series;
/// the `8.3e-6` gap is `norm_cdf`'s Abramowitz-Stegun 7.1.26 error, not a
/// pricing difference. The put follows from it by the generalised parity
/// `C - P = S - Ke^{-rτ}` asserted below, so the pair cannot drift apart.
#[test]
fn merton_price_m10_matches_the_reference_value() {
  let m = merton(0.5, 0.3, 10);
  let (call, put) = m.call_put(S, K, R, Q, TAU);
  let want_call = 4.401_569_155_621_004;
  let want_put = 6.809_109_918_595_418;
  assert!(
    (call - want_call).abs() < 1e-12,
    "call regressed: got {call}, want {want_call}"
  );
  assert!(
    (put - want_put).abs() < 1e-12,
    "put regressed: got {put}, want {want_put}"
  );
  let parity = call - S + K * (-R * TAU).exp();
  assert!((put - parity).abs() < TOL, "put {put} vs parity {parity}");
}

/// At `τ` below the finite-difference step `H_TAU = 1e-5`, the `λ > 0`
/// path's down-bump would evaluate the price series at a negative
/// time-to-maturity. `theta`/`charm`/`veta` must return `NaN` there
/// instead of the large finite garbage a silently-zeroed `NaN` term in the
/// Poisson sum would otherwise produce — mirrors `HestonPricer::theta`'s
/// identical near-expiry guard (`pricing::heston`).
#[test]
fn merton_greeks_theta_charm_veta_nan_near_expiry() {
  let m = merton(0.5, 0.3, 20);
  let tiny = 1e-6;
  assert!(
    m.theta(S, K, R, Q, tiny, OT).is_nan(),
    "theta should be NaN at tau=1e-6"
  );
  assert!(
    m.charm(S, K, R, Q, tiny, OT).is_nan(),
    "charm should be NaN at tau=1e-6"
  );
  assert!(
    m.veta(S, K, R, Q, tiny, OT).is_nan(),
    "veta should be NaN at tau=1e-6"
  );
}

/// Under [`BSMCoc::GarmanKohlhagen1983`] a caller wanting to carry at
/// `r_d - r_f` while discounting at a *separate* rate `r` passes the query
/// `(r, r - r_d + r_f)`. That is an identity, not an approximation: GK's
/// `b(r, q) = r - q`, so solving `r - q = r_d - r_f` for `q` gives exactly
/// that, and it collapses to `r_f` in the ordinary case `r == r_d`.
///
/// Before this task the mapping lived inside the pricer as
/// `Merton1976Pricer::query_rates`, reading `self.r`, `self.r_d` and
/// `self.r_f`; those three fields are query data, so they moved out with
/// the rest of the query and the caller now supplies the pair. The
/// *property* is unchanged and is still pinned here — at `r != r_d`, the
/// only configuration where carry and discount come apart — against a
/// **hand-written** closed form rather than against anything routed through
/// the pricer, so it cannot go stale with the mapping.
///
/// Task 5a shipped a silent regression on exactly this property (it
/// discounted at `r_d` instead of `r`) and it was caught by this test.
#[test]
fn merton_gk_carries_at_rd_minus_rf_and_discounts_at_r() {
  use stochastic_rs_distributions::special::norm_cdf;

  let (s, k, v, tau) = (100.0_f64, 105.0_f64, 0.25_f64, 0.75_f64);
  let (r, r_d, r_f) = (0.06_f64, 0.05_f64, 0.02_f64);
  // The query that reproduces discount `r` and carry `r_d - r_f`.
  let q = r - r_d + r_f;
  let m = Merton1976Pricer::new(v, 0.0, 0.4, 20, BSMCoc::GarmanKohlhagen1983);

  let b = r_d - r_f;
  let sqrt_tau = tau.sqrt();
  let d1 = ((s / k).ln() + (b + 0.5 * v * v) * tau) / (v * sqrt_tau);
  let d2 = d1 - v * sqrt_tau;

  // delta pins the carry factor exp((b - r) * tau) — wrong on either leg
  // if `b` came from a different pair or the discount came from r_d.
  let want_delta = ((b - r) * tau).exp() * norm_cdf(d1);
  let got_delta = m.delta(s, k, r, q, tau, OptionType::Call);
  assert!(
    (got_delta - want_delta).abs() < 1e-12,
    "GK delta: got {got_delta}, want {want_delta}"
  );

  // rho pins the discount factor exp(-r * tau) on its own.
  let want_rho = k * tau * (-r * tau).exp() * norm_cdf(d2);
  let got_rho = m.rho(s, k, r, q, tau, OptionType::Call);
  assert!(
    (got_rho - want_rho).abs() < 1e-12,
    "GK rho must discount at r ({r}), not r_d ({r_d}): got {got_rho}, want {want_rho}"
  );
}

/// Cross-arch tolerance: the goldens route through `norm_cdf`.
const TOL: f64 = 1e-12;

/// The reference price and Greeks at
/// `(s, k, r, q, tau) = (100, 105, 0.05, 0, 0.5)` and
/// `(v, lambda, gamma, m, coc) = (0.2, 0.5, 0.4, 10, Bsm1973)`.
///
/// Every value here moved when `σ_n` was corrected from
/// `√((d² + z²)·n/τ)` to `√(d² + z²·n/τ)`; each is adjudicated against a
/// reference sharing no code with the pricer. The call was `1.963018` and
/// is now `4.276112`, against `4.276118` from Gil-Pelaez inversion of the
/// Merton characteristic function and `4.2717 ± 0.0061` from an 8M-path
/// Monte Carlo — the old value sat **760 standard errors** below that
/// interval. The `6.6e-6` residual is `norm_cdf`'s Abramowitz-Stegun
/// 7.1.26 error.
///
/// The nine Greeks are adjudicated as numerical derivatives of that same
/// characteristic-function price: the five first-order ones agree to
/// between `3e-9` and `2e-6` relative, the four second-order ones to
/// `4e-4` (both sides being finite differences there). Under the old `σ_n`
/// the same comparison was off by 21 % to 111 %, and `volga` had the wrong
/// sign.
#[test]
fn merton_pins_the_reference_price_and_greeks() {
  let m = merton(0.5, 0.4, 10);
  let (call, put) = m.call_put(S, K, R, Q, TAU);
  assert!((call - 4.276111556095045).abs() < TOL, "call {call}");
  assert!((put - 6.683652319069461).abs() < TOL, "put {put}");
  assert_eq!(m.price_call(S, K, R, Q, TAU), call);
  assert_eq!(m.price_put(S, K, R, Q, TAU), put);

  let want = [
    0.4496816609264091,
    0.032071500866967965,
    26.41969293408763,
    -7.717747123159312,
    20.346027268272934,
    0.5136783187698057,
    -0.27511605438235165,
    3.6927971791556042,
    -30.537329331892234,
  ];
  let got = m.greeks(S, K, R, Q, TAU, OT).as_array();
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
fn merton_greeks_aggregate_matches_accessors() {
  let m = merton(0.5, 0.3, 20);
  for ot in [OptionType::Call, OptionType::Put] {
    let g = m.greeks(S, K, R, Q, TAU, ot);
    assert_eq!(g.delta, m.delta(S, K, R, Q, TAU, ot), "delta");
    assert_eq!(g.gamma, m.gamma(S, K, R, Q, TAU), "gamma");
    assert_eq!(g.vega, m.vega(S, K, R, Q, TAU, ot), "vega");
    assert_eq!(g.theta, m.theta(S, K, R, Q, TAU, ot), "theta");
    assert_eq!(g.rho, m.rho(S, K, R, Q, TAU, ot), "rho");
    assert_eq!(g.vanna, m.vanna(S, K, R, Q, TAU, ot), "vanna");
    assert_eq!(g.charm, m.charm(S, K, R, Q, TAU, ot), "charm");
    assert_eq!(g.volga, m.volga(S, K, R, Q, TAU, ot), "volga is vomma");
    assert_eq!(g.veta, m.veta(S, K, R, Q, TAU, ot), "veta is dvega_dtime");
    assert_ne!(g.volga, g.veta, "volga and veta must not be the same value");
  }
}

/// Each term of the Poisson series carries at `exp((b - r) * tau)`, which
/// equals the trait default's `exp(-q * tau)` only when `b = r - q` —
/// false for `Bsm1973` at `q != 0`, which is this test's configuration.
#[test]
fn merton_price_put_overrides_vanilla_parity() {
  let m = merton(0.5, 0.4, 10);
  let q = 0.03;
  let (call, put) = m.call_put(S, K, R, q, TAU);
  let vanilla = call - S * (-q * TAU).exp() + K * (-R * TAU).exp();
  assert!(
    (put - vanilla).abs() > 1e-3,
    "the default would be a silent mispricing: put {put}, default {vanilla}"
  );
  // `Bsm1973`'s carry is `b = r`, so the generalised parity reduces to
  // `C - P = S - K e^{-r tau}` and must hold exactly.
  let generalised = call - S + K * (-R * TAU).exp();
  assert!(
    (put - generalised).abs() < TOL,
    "put {put} vs {generalised}"
  );
}

/// The capability the reshape exists for: one model, a whole grid.
#[test]
fn merton_one_model_prices_a_grid() {
  let m = merton(0.5, 0.4, 10);
  for &tau in &[0.25, 0.5, 1.0] {
    let mut prev = f64::INFINITY;
    for &k in &[90.0, 100.0, 110.0] {
      let c = m.price_call(S, k, R, Q, tau);
      assert!(c.is_finite() && c < prev, "call must fall in strike");
      prev = c;
    }
  }
}
