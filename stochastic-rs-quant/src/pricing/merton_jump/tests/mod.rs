use super::*;
use crate::OptionType;
use crate::pricing::bsm::BsmPricer;
use crate::traits::Greeks;

mod construction;
mod degenerate;
mod formula;
mod lambda_zero;
mod option_type;
mod poison;
mod volatility;

/// The goldens below are pinned at this spot with `K = 105` and `m ≤ 20`; at `λτ ≤ 0.25` the series
/// has converged to double precision well before 20 terms.
const S: f64 = 100.0;
const K: f64 = 105.0;
const R: f64 = 0.05;
const Q: f64 = 0.0;
const TAU: f64 = 0.5;
const OT: OptionType = OptionType::Call;

fn merton(lambda: f64, gamma: f64, m: usize) -> Merton1976Pricer {
  Merton1976Pricer::new(0.2, lambda, gamma, m, BsmCoc::Bsm1973)
}

fn merton_at(v: f64, lambda: f64, gamma: f64, m: usize) -> Merton1976Pricer {
  Merton1976Pricer::new(v, lambda, gamma, m, BsmCoc::Bsm1973)
}

fn bumped_price(m: &Merton1976Pricer, ds: f64, dv: f64, dtau: f64) -> f64 {
  let mut p = *m;
  p.sigma += dv;
  p.price_option(S + ds, K, R, Q, TAU + dtau, OT)
}

/// At `λ = 0` the series is its `n = 0` term, Black-Scholes at `sigma`.
#[test]
fn merton_greeks_lambda_zero_equals_bs() {
  let m = merton(0.0, 0.4, 20);
  let bs = BsmPricer::new(m.sigma, m.b);

  let cases = [
    (
      "delta",
      m.delta(S, K, R, Q, TAU, OT),
      bs.delta(S, K, R, Q, TAU, OT),
    ),
    ("gamma", m.gamma(S, K, R, Q, TAU), bs.gamma(S, K, R, Q, TAU)),
    ("vega", m.vega(S, K, R, Q, TAU), bs.vega(S, K, R, Q, TAU)),
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
    ("vanna", m.vanna(S, K, R, Q, TAU), bs.vanna(S, K, R, Q, TAU)),
    (
      "charm",
      m.charm(S, K, R, Q, TAU, OT),
      bs.charm(S, K, R, Q, TAU, OT),
    ),
    ("volga", m.volga(S, K, R, Q, TAU), bs.volga(S, K, R, Q, TAU)),
    ("veta", m.veta(S, K, R, Q, TAU), bs.veta(S, K, R, Q, TAU)),
  ];
  for (name, got, want) in cases {
    assert!((got - want).abs() < 1e-10, "{name}: got {got}, want {want}");
  }
}

/// The Greeks against central differences of the price; `gamma` at `1e-3`, since a second
/// difference of one Black-Scholes price is already several `1e-4` from its closed form.
#[test]
fn merton_greeks_match_finite_difference() {
  let m = merton(0.5, 0.3, 20);

  let h_s = S * 1e-4;
  let delta_fd = (bumped_price(&m, h_s, 0.0, 0.0) - bumped_price(&m, -h_s, 0.0, 0.0)) / (2.0 * h_s);
  let gamma_fd = (bumped_price(&m, h_s, 0.0, 0.0) - 2.0 * m.price_option(S, K, R, Q, TAU, OT)
    + bumped_price(&m, -h_s, 0.0, 0.0))
    / (h_s * h_s);

  let h_v = m.sigma * 1e-4;
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
    ("vega", m.vega(S, K, R, Q, TAU), vega_fd, 1e-4),
    ("theta", m.theta(S, K, R, Q, TAU, OT), theta_fd, 1e-4),
    ("rho", m.rho(S, K, R, Q, TAU, OT), rho_fd, 1e-4),
  ];
  for (name, analytic, fd, tol) in cases {
    let rel = (analytic - fd).abs() / fd.abs().max(1e-8);
    assert!(rel < tol, "{name}: analytic={analytic}, fd={fd}, rel={rel}");
  }
}

/// All nine Greeks are finite with both diffusive and jump variance.
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

/// All nine Greeks stay finite at `m = 50`, the Python binding's default and past the ≈ 21 terms an
/// integer `n!` would survive.
#[test]
fn merton_greeks_finite_at_m50() {
  let m = merton(0.5, 0.3, 50);
  let g = m.greeks(S, K, R, Q, TAU, OT);
  for (name, v) in Greeks::COMPONENT_NAMES.iter().zip(g.as_array()) {
    assert!(v.is_finite(), "{name} is not finite at m=50: {v}");
  }
  // The pair must survive `m = 50` as well.
  let CallPut { call, put } = m.price_call_put(S, K, R, Q, TAU);
  assert!(call.is_finite(), "call not finite at m=50: {call}");
  assert!(put.is_finite(), "put not finite at m=50: {put}");
}

/// The `m = 10` pair against `4.401577`, the call a Gil-Pelaez inversion of Merton's chf confirms;
/// the put follows by `C − P = S − Ke^{−rτ}`.
#[test]
fn merton_price_m10_matches_the_reference_value() {
  let m = merton(0.5, 0.3, 10);
  let CallPut { call, put } = m.price_call_put(S, K, R, Q, TAU);
  let want_call = 4.401_577_413_196_381;
  let want_put = 6.809_118_176_170_799_5;
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

/// At `λ > 0` and `τ` under the `1e-5` bump, `theta`, `charm` and `veta` are `NaN`: the down-bump
/// would price at a negative maturity.
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
    m.veta(S, K, R, Q, tiny).is_nan(),
    "veta should be NaN at tau=1e-6"
  );
}

/// An ordinary model off the strike and a frozen one at its forward: expired, each is worth its
/// payoff and has no Greek.
#[test]
fn zero_tau_prices_the_payoff_and_no_greek() {
  let frozen = Merton1976Pricer::new(0.0, 0.5, 0.4, 20, BsmCoc::Black1976);
  for (m, k) in [(merton(0.5, 0.4, 10), K), (frozen, S)] {
    assert_eq!(
      m.price_call_put(S, k, R, Q, 0.0),
      CallPut {
        call: (S - k).max(0.0),
        put: (k - S).max(0.0)
      }
    );
    let greeks = m.greeks(S, k, R, Q, 0.0, OT).as_array();
    for (greek, v) in Greeks::COMPONENT_NAMES.iter().zip(greeks) {
      assert!(v.is_nan(), "k = {k}: {greek} must be NaN, got {v}");
    }
  }
}

/// Under [`BsmCoc::GarmanKohlhagen1983`] the query `(r, r − r_d + r_f)` carries at `r_d − r_f` and
/// discounts at `r`, checked at `r ≠ r_d` against a hand-written closed form.
#[test]
fn merton_gk_carries_at_rd_minus_rf_and_discounts_at_r() {
  use stochastic_rs_distributions::special::norm_cdf;

  let (s, k, v, tau) = (100.0_f64, 105.0_f64, 0.25_f64, 0.75_f64);
  let (r, r_d, r_f) = (0.06_f64, 0.05_f64, 0.02_f64);
  // The query that reproduces discount `r` and carry `r_d - r_f`.
  let q = r - r_d + r_f;
  let m = Merton1976Pricer::new(v, 0.0, 0.4, 20, BsmCoc::GarmanKohlhagen1983);

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

/// Price and Greeks at `(100, 105, 0.05, 0, 0.5)`, `(σ, λ, γ, m) = (0.2, 0.5, 0.4, 10)`: the call
/// is the Gil-Pelaez value `4.276118`, the vol Greeks 60-digit mpmath derivatives.
#[test]
fn merton_pins_the_reference_price_and_greeks() {
  let m = merton(0.5, 0.4, 10);
  let CallPut { call, put } = m.price_call_put(S, K, R, Q, TAU);
  assert!((call - 4.276118145681529).abs() < TOL, "call {call}");
  assert!((put - 6.68365890865595).abs() < TOL, "put {put}");
  assert_eq!(m.price_call(S, K, R, Q, TAU), call);
  assert_eq!(m.price_put(S, K, R, Q, TAU), put);

  let want = [
    0.4496816679513633,
    0.032071500866967965,
    26.419722807093013,
    -7.7177657908489286,
    20.3460243247274,
    0.5135917494280899,
    -0.27506665389864793,
    3.6913241421551275,
    -30.536697719973912,
  ];
  let got = m.greeks(S, K, R, Q, TAU, OT).as_array();
  for (i, name) in Greeks::COMPONENT_NAMES.iter().enumerate() {
    let tolerance = if *name == "veta" { 1e-8 } else { TOL };
    assert!(
      (got[i] - want[i]).abs() < tolerance,
      "{name}: got {}, want {}",
      got[i],
      want[i]
    );
  }
}

/// The aggregate is the nine accessors, with `volga` and `veta` not transposed.
#[test]
fn merton_greeks_aggregate_matches_accessors() {
  let m = merton(0.5, 0.3, 20);
  for ot in [OptionType::Call, OptionType::Put] {
    let g = m.greeks(S, K, R, Q, TAU, ot);
    assert_eq!(g.delta, m.delta(S, K, R, Q, TAU, ot), "delta");
    assert_eq!(g.gamma, m.gamma(S, K, R, Q, TAU), "gamma");
    assert_eq!(g.vega, m.vega(S, K, R, Q, TAU), "vega");
    assert_eq!(g.theta, m.theta(S, K, R, Q, TAU, ot), "theta");
    assert_eq!(g.rho, m.rho(S, K, R, Q, TAU, ot), "rho");
    assert_eq!(g.vanna, m.vanna(S, K, R, Q, TAU), "vanna");
    assert_eq!(g.charm, m.charm(S, K, R, Q, TAU, ot), "charm");
    assert_eq!(g.volga, m.volga(S, K, R, Q, TAU), "volga");
    assert_eq!(g.veta, m.veta(S, K, R, Q, TAU), "veta");
    assert_ne!(g.volga, g.veta, "volga and veta must not be the same value");
  }
}

/// Under `Bsm1973` at `q ≠ 0` each term carries at `e^{(b−r)τ} ≠ e^{−qτ}`, so the put is not the
/// parity default.
#[test]
fn merton_price_put_overrides_vanilla_parity() {
  let m = merton(0.5, 0.4, 10);
  let q = 0.03;
  let CallPut { call, put } = m.price_call_put(S, K, R, q, TAU);
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
