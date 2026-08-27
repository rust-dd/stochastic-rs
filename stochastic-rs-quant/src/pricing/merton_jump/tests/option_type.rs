//! Why four of the nine Greeks take no `option_type`, and the one place
//! that decision is visible in a number.
//!
//! `vega`, `vanna`, `volga` and `veta` each differentiate the price in
//! `sigma` at least once. Generalised put-call parity's spread,
//! `C - P = S e^{(b-r)tau} - K e^{-r tau}`, carries no `sigma`, so that
//! derivative annihilates it and the call and the put share one answer —
//! which is why the accessors difference the call series and hand the
//! result to both. `delta`, `theta`, `rho` and `charm` leave a surviving
//! spread term and keep their parameter; `gamma` loses its own for the
//! other reason the spread admits, being linear in `S`.
//!
//! The second test is the exception that proves it is a real argument
//! rather than a hopeful one: on a 1800-point sweep exactly one
//! configuration made the two legs disagree, and it is not a parity
//! failure at all but a `2e-9` jump in `erf` across the origin.

use super::*;

/// `vega`/`vanna`/`volga`/`veta` take no `option_type`, and this is the
/// property that licenses it: generalised put-call parity's spread
/// `C - P = S e^{(b-r)tau} - K e^{-r tau}` carries no `sigma`, so a
/// derivative that touches `sigma` even once annihilates it and the call
/// and the put have the *same* number. The four accessors difference the
/// call series; this test differences the **put** and gets the same
/// answers back.
///
/// Two halves, because the second does not imply the first. The price
/// level pins that the spread really is `sigma`-free — bump `v` and
/// `call - put` must not move, which is the parity statement itself. The
/// Greek level then pins that the finite differences agree.
///
/// **The tolerance is a round-off budget, not a fitted number.** Each
/// accessor divides a difference of prices by its own step, so a fixed
/// error in the prices shows up divided by that step: `2h_v` for `vega`,
/// `4 h_s h_v` for `vanna`, `h_v^2` for `volga`, `4 h_v h_tau` for `veta`.
/// The budget below allows 64 ulps of the largest quantity a Black-Scholes
/// leg evaluates (`max(S, K e^{-r tau})`, which is what the put's own
/// `-S N(-d1) + K e^{-r tau} N(-d2)` cancels between) over that step. The
/// observed differences use between 0.49 % and 0.98 % of it — `3.55e-10`
/// of `3.64e-8` for `vega`, `3.55e-5` of `3.64e-3` for `volga` — so the
/// four are equal to well under a single ulp of the differenced prices
/// rather than merely inside a loose band.
///
/// The steps are re-derived from the same expressions the implementation
/// uses (`|v|.max(0.01) * 1e-4`, `|s| * 1e-4`, `H_TAU = 1e-5`) so the
/// truncation error is identical on both sides and cancels, leaving only
/// round-off. `price_put` is deliberately the *other* accumulation route
/// (`call_put`'s own loop rather than `series_price`), so the budget also
/// absorbs the two orders of summation.
#[test]
fn the_volatility_greeks_are_the_same_for_a_put() {
  let m = merton(0.5, 0.4, 10);
  let h_v = m.v.abs().max(0.01) * 1e-4;
  let h_s = S.abs() * 1e-4;
  let h_tau = 1e-5;

  // The parity spread carries no `sigma`: bumping `v` moves the call and
  // the put by the same amount, so their difference is invariant.
  let spread_at = |dv: f64| {
    let mut b = m;
    b.v += dv;
    let (c, p) = b.call_put(S, K, R, Q, TAU);
    c - p
  };
  let spread = spread_at(0.0);
  for dv in [-h_v, h_v] {
    let moved = spread_at(dv);
    assert!(
      (moved - spread).abs() < 1e-12,
      "the parity spread moved with v: {spread} -> {moved}"
    );
  }

  let put = |dv: f64, ds: f64, dtau: f64| {
    let mut b = m;
    b.v += dv;
    b.price_put(S + ds, K, R, Q, TAU + dtau)
  };
  let vega_put = (put(h_v, 0.0, 0.0) - put(-h_v, 0.0, 0.0)) / (2.0 * h_v);
  let vanna_put = (put(h_v, h_s, 0.0) - put(-h_v, h_s, 0.0) - put(h_v, -h_s, 0.0)
    + put(-h_v, -h_s, 0.0))
    / (4.0 * h_s * h_v);
  let volga_put =
    (put(h_v, 0.0, 0.0) - 2.0 * put(0.0, 0.0, 0.0) + put(-h_v, 0.0, 0.0)) / (h_v * h_v);
  let veta_put = -(put(h_v, 0.0, h_tau) - put(h_v, 0.0, -h_tau) - put(-h_v, 0.0, h_tau)
    + put(-h_v, 0.0, -h_tau))
    / (4.0 * h_v * h_tau);

  let leg_scale = S.max(K * (-R * TAU).exp());
  let budget = |step: f64| 64.0 * f64::EPSILON * leg_scale / step;
  let cases = [
    ("vega", m.vega(S, K, R, Q, TAU), vega_put, budget(2.0 * h_v)),
    (
      "vanna",
      m.vanna(S, K, R, Q, TAU),
      vanna_put,
      budget(4.0 * h_s * h_v),
    ),
    (
      "volga",
      m.volga(S, K, R, Q, TAU),
      volga_put,
      budget(h_v * h_v),
    ),
    (
      "veta",
      m.veta(S, K, R, Q, TAU),
      veta_put,
      budget(4.0 * h_v * h_tau),
    ),
  ];
  for (name, from_call, from_put, tol) in cases {
    let d = (from_call - from_put).abs();
    assert!(
      d < tol,
      "{name}: call-side {from_call}, put-side {from_put}, |diff| {d:.3e} exceeds the {tol:.3e} round-off budget"
    );
  }
}

/// The one query family where [`Merton1976Pricer::volga`] is not the
/// number it claims, pinned with both the wrong value and the right one so
/// neither can drift. Found while removing `option_type` from the four
/// volatility Greeks: it is the only configuration on a 1800-point sweep
/// where the call-side and put-side answers came apart by more than a few
/// ulps, and it turned out not to be about the option type at all.
///
/// **The cause is `erf`, and it is two distinct faults.**
/// [`erf`](stochastic_rs_distributions::special::erf) is Abramowitz &
/// Stegun 7.1.26, whose five coefficients sum to `0.999999999` rather than
/// to `1`, and it is made odd by an explicit sign branch. So it has a
/// `2e-9` **jump** across the origin, `-1e-9` to `+1e-9`, where the true
/// `erf` passes continuously through `0`:
///
/// 1. The jump alone contaminates `volga` — and only `volga`, the one
///    Greek of the nine that evaluates the series at the **unbumped** `v`,
///    dividing by `h_v^2 ~ 4e-10`. It hits the call and the put *equally*,
///    because `norm_cdf(-x)` is `1 - norm_cdf(x)` exactly, so both legs
///    are wrong by the same `+-3.3` and no `option_type` question arises.
/// 2. At exactly `+-0.0` the sign branch fails to flip, because
///    `-0.0 < 0.0` is false in Rust. That is the *only* argument at which
///    `norm_cdf(x) + norm_cdf(-x)` is not exactly `1`, and it is the only
///    way the call and the put can disagree.
///
/// The configuration reaches both, and it is a genuine coincidence rather
/// than a constructed one: at `(v, lambda, gamma) = (0.2, 0.5, 0.3)` and
/// `tau = 1`, the `n = 3` term has `sigma_3^2 = 2.5 v^2 = 0.1`, which
/// `Bsm1973`'s `b = r = 0.05` makes exactly `2b` — and `d_2 = 0` at
/// `S = K` is precisely the condition `sigma^2 = 2b`.
///
/// **A failure here is most likely good news** — that `erf` gained a
/// modern rational-Chebyshev or `erfc`-based kernel and both faults went
/// with it. Re-derive rather than re-fit.
#[test]
fn a_poisson_term_on_erfs_origin_jump_wobbles_volga() {
  use stochastic_rs_distributions::special::erf;
  use stochastic_rs_distributions::special::norm_cdf;

  // Fault 1: a 2e-9 jump where the true erf is continuous through 0.
  let above = erf(1e-300);
  let below = erf(-1e-300);
  assert!((above - 1e-9).abs() < 1e-16, "erf just above 0: {above}");
  assert_eq!(below, -above, "the sign branch makes it odd off zero");

  // Fault 2: at exactly +-0.0 the branch does not flip, so this is the
  // one argument where the two tails do not sum to 1.
  assert_eq!(erf(-0.0), erf(0.0), "-0.0 < 0.0 is false");
  assert!((norm_cdf(0.0) + norm_cdf(-0.0) - 1.0 - 1e-9).abs() < 1e-16);
  assert_eq!(
    norm_cdf(1e-9) + norm_cdf(-1e-9) - 1.0,
    0.0,
    "exact at every other argument"
  );

  let m = merton_at(0.2, 0.5, 0.3, 10);
  let (s, k, tau) = (110.0_f64, 110.0_f64, 1.0_f64);
  let coarse = |strike: f64, ot: OptionType| {
    // 100x the production step divides the same 2e-9 by 1e4 more.
    let h = 2e-3;
    let mut up = m;
    up.v += h;
    let mut dn = m;
    dn.v -= h;
    (up.price_option(s, strike, R, Q, tau, ot) - 2.0 * m.price_option(s, strike, R, Q, tau, ot)
      + dn.price_option(s, strike, R, Q, tau, ot))
      / (h * h)
  };
  let truth = coarse(k, OptionType::Call);
  assert!(
    (truth - coarse(k, OptionType::Put)).abs() < 1e-2,
    "off the artefact the two legs agree: {truth} vs {}",
    coarse(k, OptionType::Put)
  );
  assert!(
    (truth - 11.33).abs() < 1e-2,
    "the reference volga moved: {truth}"
  );

  // Fault 1, in a band rather than at a point: nudging the strike by 1e-5
  // leaves d_2 inside the stencil's sign change, so the wobble survives.
  let put_volga = |strike: f64| {
    let h = m.v * 1e-4;
    let mut up = m;
    up.v += h;
    let mut dn = m;
    dn.v -= h;
    (up.price_put(s, strike, R, Q, tau) - 2.0 * m.price_put(s, strike, R, Q, tau)
      + dn.price_put(s, strike, R, Q, tau))
      / (h * h)
  };
  for strike in [k, k + 1e-5, k - 1e-5] {
    let got = m.volga(s, strike, R, Q, tau);
    assert!(
      (got - truth).abs() > 3.0,
      "K={strike}: volga {got} no longer wobbles against {truth}"
    );
  }

  // Fault 2, at the exact point only: this is the whole of the call/put
  // divergence, and one ulp of strike either side removes it.
  assert!(
    (m.volga(s, k, R, Q, tau) - put_volga(k)).abs() > 3.0,
    "the exact d2 = 0 point splits the legs"
  );
  for strike in [k + 1e-5, k - 1e-5] {
    let d = (m.volga(s, strike, R, Q, tau) - put_volga(strike)).abs();
    assert!(d < 1e-2, "K={strike}: legs should rejoin, apart by {d:.3e}");
  }
}
