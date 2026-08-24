use super::*;
use crate::pricing::bsm::BSMPricer;
use crate::traits::Greeks;
use crate::traits::GreeksExt;
use crate::traits::PricerExt;
use crate::traits::TimeExt;

/// `m` (Poisson-series term count) is capped at 20 in these tests:
/// [`PricerExt::calculate_call_put`]'s own term loop computes `n!` as a
/// `usize` product, which overflows past `n = 20` — a pre-existing
/// limitation of the price series itself (not introduced by `GreeksExt`),
/// unrelated to convergence (at `λτ ≤ 0.25` the series is converged to
/// double precision well before `n = 20`).
///
/// `k = 105` (not `100`, i.e. not ATM): the `n = 0` term is always priced
/// at zero volatility (see [`Merton1976Pricer::greek_series`]'s doc), which
/// turns it into a deterministic, piecewise-*linear* payoff of `S` with a
/// kink exactly at `S = K` (since `Bsm1973`'s cost of carry is `b = r`, the
/// forward is `S` itself) — `bumped_price`'s finite difference would
/// straddle that kink at `K = S`, so an ATM strike is *not* a
/// representative test of a smooth region.
fn merton(lambda: f64, gamma: f64, m: usize) -> Merton1976Pricer {
  Merton1976Pricer::new(
    100.0,
    0.2,
    105.0,
    0.05,
    None,
    None,
    None,
    lambda,
    gamma,
    m,
    Some(0.5),
    None,
    None,
    OptionType::Call,
    BSMCoc::Bsm1973,
  )
}

fn bumped_price(m: &Merton1976Pricer, ds: f64, dv: f64, dtau: f64) -> f64 {
  let mut p = m.clone();
  p.s += ds;
  p.v += dv;
  p.tau = Some(p.tau_or_from_dates() + dtau);
  p.calculate_price()
}

/// λ=0 collapses the Poisson sum to its single `n=0` term (weight 1),
/// degenerating Merton (1976) to plain Black-Scholes at the input
/// volatility — see `Σ_{n=0}^∞ e^{-λT}(λT)^n/n!` at `λ=0`.
#[test]
fn merton_greeks_lambda_zero_equals_bs() {
  let m = merton(0.0, 0.4, 20);
  let bs = BSMPricer::new(m.v, m.b);
  let (s, k, ot) = (m.s, m.k, m.option_type);
  let (r, q) = m.query_rates();
  let tau = m.tau_or_from_dates();

  let cases = [
    ("delta", m.delta(), bs.delta(s, k, r, q, tau, ot)),
    ("gamma", m.gamma(), bs.gamma(s, k, r, q, tau)),
    ("vega", m.vega(), bs.vega(s, k, r, q, tau)),
    ("theta", m.theta(), bs.theta(s, k, r, q, tau, ot)),
    ("rho", m.rho(), bs.rho(s, k, r, q, tau, ot)),
    ("vanna", m.vanna(), bs.vanna(s, k, r, q, tau)),
    ("charm", m.charm(), bs.charm(s, k, r, q, tau, ot)),
    ("volga", m.volga(), bs.vomma(s, k, r, q, tau)),
    ("veta", m.veta(), bs.dvega_dtime(s, k, r, q, tau)),
  ];
  for (name, got, want) in cases {
    assert!((got - want).abs() < 1e-10, "{name}: got {got}, want {want}");
  }
}

/// Merton delta/gamma/vega/theta/rho vs. central finite differences of
/// `calculate_price()` itself, independent of whether a given Greek is
/// internally a closed-form series or a finite difference.
///
/// `gamma` is checked at a looser `1e-3` (not `1e-4`): its BSM closed form
/// (`norm_pdf(d1)/(S·v·√τ)`) and a central second-difference of
/// `BSMPricer::calculate_call_put` already disagree by several `1e-4`
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

  let h_s = m.s * 1e-4;
  let delta_fd = (bumped_price(&m, h_s, 0.0, 0.0) - bumped_price(&m, -h_s, 0.0, 0.0)) / (2.0 * h_s);
  let gamma_fd = (bumped_price(&m, h_s, 0.0, 0.0) - 2.0 * m.calculate_price()
    + bumped_price(&m, -h_s, 0.0, 0.0))
    / (h_s * h_s);

  let h_v = m.v * 1e-4;
  let vega_fd = (bumped_price(&m, 0.0, h_v, 0.0) - bumped_price(&m, 0.0, -h_v, 0.0)) / (2.0 * h_v);

  let h_t = 1e-5;
  let theta_fd =
    -(bumped_price(&m, 0.0, 0.0, h_t) - bumped_price(&m, 0.0, 0.0, -h_t)) / (2.0 * h_t);

  let h_r = 1e-5;
  let mut r_up = m.clone();
  r_up.r += h_r;
  let mut r_dn = m.clone();
  r_dn.r -= h_r;
  let rho_fd = (r_up.calculate_price() - r_dn.calculate_price()) / (2.0 * h_r);

  let cases = [
    ("delta", m.delta(), delta_fd, 1e-4),
    ("gamma", m.gamma(), gamma_fd, 1e-3),
    ("vega", m.vega(), vega_fd, 1e-4),
    ("theta", m.theta(), theta_fd, 1e-4),
    ("rho", m.rho(), rho_fd, 1e-4),
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
  let g = m.greeks();
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
/// documented default, and well past `PricerExt::calculate_call_put`'s own
/// `usize`-factorial overflow threshold (`m ≈ 21`; `(1..=21).product::<usize>()`
/// panics in debug builds, silently wraps in release). `GreeksExt` routes
/// every Greek through [`Merton1976Pricer::series_price`] instead, which
/// has no such ceiling — this test is the direct regression check for that.
#[test]
fn merton_greeks_finite_at_m50() {
  let m = merton(0.5, 0.3, 50);
  let g = m.greeks();
  for (name, v) in Greeks::COMPONENT_NAMES.iter().zip(g.as_array()) {
    assert!(v.is_finite(), "{name} is not finite at m=50: {v}");
  }
  // `calculate_call_put` itself (not just the `GreeksExt` path) must also
  // survive m=50 now that it is routed through `poisson_weight` instead of
  // an integer `n!` — regression check for the overflow this refactor
  // fixes (`(1..=i).product::<usize>()` panics in debug / wraps in release
  // past `i ≈ 21`).
  let (call, put) = m.calculate_call_put();
  assert!(call.is_finite(), "call not finite at m=50: {call}");
  assert!(put.is_finite(), "put not finite at m=50: {put}");
}

/// `calculate_call_put` was refactored to accumulate the Poisson weight
/// `e^{-λτ}(λτ)^n/n!` via [`Merton1976Pricer::poisson_weight`]'s running
/// product instead of an integer `n!` (which overflows past `n ≈ 20`). The
/// two accumulation orders are mathematically identical but not guaranteed
/// bit-for-bit equal, so this pins the m=10 call price to the value the
/// *pre-refactor* factorial-based loop actually produced — computed by
/// temporarily instrumenting the old code path with a `println!` before
/// making any change, not derived from the new code. A tolerance of `1e-12`
/// absolute leaves headroom for the differing floating-point operation
/// order while still catching a real regression.
#[test]
fn merton_price_m10_matches_pre_refactor_value() {
  let m = merton(0.5, 0.3, 10);
  let (call, put) = m.calculate_call_put();
  // Captured from the factorial-based `(1..=i).product::<usize>()` loop
  // prior to the `poisson_weight` refactor.
  let old_call = 1.883_106_823_679_627_8;
  let old_put = 4.290_647_586_654_042;
  assert!(
    (call - old_call).abs() < 1e-12,
    "call regressed: got {call}, pre-refactor {old_call}"
  );
  assert!(
    (put - old_put).abs() < 1e-12,
    "put regressed: got {put}, pre-refactor {old_put}"
  );
}

/// At `τ` below the finite-difference step `H_TAU = 1e-5`, the `λ > 0`
/// path's down-bump would evaluate the price series at a negative
/// time-to-maturity. `theta`/`charm`/`veta` must return `NaN` there
/// instead of the large finite garbage a silently-zeroed `NaN` term in the
/// Poisson sum would otherwise produce — mirrors `HestonPricer::theta`'s
/// identical near-expiry guard (`pricing::heston::tests`).
#[test]
fn merton_greeks_theta_charm_veta_nan_near_expiry() {
  let mut m = merton(0.5, 0.3, 20);
  m.tau = Some(1e-6);
  assert!(m.theta().is_nan(), "theta should be NaN at tau=1e-6");
  assert!(m.charm().is_nan(), "charm should be NaN at tau=1e-6");
  assert!(m.veta().is_nan(), "veta should be NaN at tau=1e-6");
}

/// Under `BSMCoc::GarmanKohlhagen1983` this pricer carries at `r_d - r_f`
/// but discounts at its own `r` field, which is separate from `r_d`. That
/// split only shows up at `r != r_d`, so it is pinned here against a
/// hand-written closed form rather than against anything routed through
/// [`Merton1976Pricer::query_rates`]. Textbook GK would discount at `r_d`;
/// whichever task migrates this pricer decides whether to change that.
#[test]
fn merton_gk_carries_at_rd_minus_rf_and_discounts_at_r() {
  use stochastic_rs_distributions::special::norm_cdf;

  let (s, k, v, tau) = (100.0_f64, 105.0_f64, 0.25_f64, 0.75_f64);
  let (r, r_d, r_f) = (0.06_f64, 0.05_f64, 0.02_f64);
  let m = Merton1976Pricer::builder(s, v, k, r, 0.0, 0.4, 20)
    .r_d(r_d)
    .r_f(r_f)
    .tau(tau)
    .coc(BSMCoc::GarmanKohlhagen1983)
    .build();

  let b = r_d - r_f;
  let sqrt_tau = tau.sqrt();
  let d1 = ((s / k).ln() + (b + 0.5 * v * v) * tau) / (v * sqrt_tau);
  let d2 = d1 - v * sqrt_tau;

  // delta pins the carry factor exp((b - r) * tau) — wrong on either leg
  // if `b` came from (r, q) or the discount came from r_d.
  let want_delta = ((b - r) * tau).exp() * norm_cdf(d1);
  assert!(
    (m.delta() - want_delta).abs() < 1e-12,
    "GK delta: got {}, want {want_delta}",
    m.delta()
  );

  // rho pins the discount factor exp(-r * tau) on its own.
  let want_rho = k * tau * (-r * tau).exp() * norm_cdf(d2);
  assert!(
    (m.rho() - want_rho).abs() < 1e-12,
    "GK rho must discount at r ({r}), not r_d ({r_d}): got {}, want {want_rho}",
    m.rho()
  );
}
