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
  let bs = BSMPricer::new(
    m.s,
    m.v,
    m.k,
    m.r,
    m.r_d,
    m.r_f,
    m.q,
    m.tau,
    m.eval,
    m.expiration,
    m.option_type,
    m.b,
  );

  let cases = [
    ("delta", m.delta(), bs.delta()),
    ("gamma", m.gamma(), bs.gamma()),
    ("vega", m.vega(), bs.vega()),
    ("theta", m.theta(), bs.theta()),
    ("rho", m.rho(), bs.rho()),
    ("vanna", m.vanna(), bs.vanna()),
    ("charm", m.charm(), bs.charm()),
    ("volga", m.volga(), bs.vomma()),
    ("veta", m.veta(), bs.dvega_dtime()),
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
