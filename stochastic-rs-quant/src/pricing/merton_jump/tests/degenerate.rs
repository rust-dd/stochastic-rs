//! A zero-volatility term is a removable singularity at the forward, and
//! the price there is the limit — not `NaN`.
//!
//! A term priced at `term_vol(n, τ) = 0` is a zero-volatility Black-Scholes
//! call. Its `d₁` is `±∞` wherever `Se^{bτ} ≠ K` and `0/0` **at** the
//! forward, which poisons the whole Poisson sum. The limit is not
//! undefined: both normal CDFs converge to `½` as `σ → 0⁺` along
//! `Se^{bτ} = K`, so the term tends to `½(Se^{(b-r)τ} - Ke^{-rτ}) = 0`.
//!
//! `σ_n = √(d² + z²·n/τ)` is zero only where the *diffusive* volatility `d`
//! is, so what reaches this branch is a zero total volatility — `v = 0`,
//! which `new` accepts on purpose. The pure-jump corner `gamma = 1` should
//! reach it too, at the `n = 0` term alone, but only does so about half the
//! time: `diffusive_std` computes `v² − λz²` after round-tripping `z`
//! through a `sqrt`, so `(v, λ) = (0.5, 1)` lands on `d = 0` while
//! `(0.2, 0.5)` lands one ulp below and returns `NaN`. That is a separate
//! defect and is not pinned here.
//!
//! An ordinary configuration has `σ_0 = d > 0` and never touches the
//! branch; `an_ordinary_configuration_never_reaches_the_branch` is the pin
//! that keeps that claim from rotting.
//!
//! Which strike is singular is set by the cost of carry, not by the
//! volatility: `Black1976` and `Asay1982` have `b = 0`, so the hole lands on
//! the at-the-money strike — the most-quoted point on a futures-option
//! surface. The three carrying conventions put it at `Se^{bτ}`, a strike
//! nobody asks for exactly.

use super::*;

const FUT_TAU: f64 = 0.5;

/// Zero total volatility: no diffusion and no jump size, so every term of
/// the series is degenerate.
fn frozen(coc: BSMCoc) -> Merton1976Pricer {
  Merton1976Pricer::new(0.0, 0.5, 0.4, 20, coc)
}

/// The headline: an at-the-money futures option priced `NaN`.
///
/// The second assertion is what makes this a test of the interception
/// rather than of arithmetic — the `BSMPricer` the term is built from does
/// return `NaN` at exactly this query, so without the branch the sum is
/// `NaN` too.
#[test]
fn zero_total_volatility_at_the_forward_is_the_limit() {
  for coc in [BSMCoc::Black1976, BSMCoc::Asay1982] {
    let m = frozen(coc);
    let (call, put) = m.call_put(S, S, R, Q, FUT_TAU);
    assert_eq!(call, 0.0, "{coc:?} call");
    assert_eq!(put, 0.0, "{coc:?} put");

    let raw = BSMPricer::new(0.0, coc).call_put(S, S, R, Q, FUT_TAU);
    assert!(
      raw.0.is_nan() && raw.1.is_nan(),
      "{coc:?}: the underlying zero-vol term must be NaN, else this pins nothing: {raw:?}"
    );
  }
}

/// `0` is the value the *neighbourhood* tends to, not merely a constant
/// that happens to be finite: the surrounding price is `e^{-rτ}(F-K)⁺`, so
/// a strike `1e-6` either side of the forward is worth `9.75e-7` on the
/// in-the-money leg and exactly nothing on the other. Both one-sided limits
/// are `0`; only the slope jumps.
///
/// The pin that a wrong constant would survive `assert_eq!(call, 0.0)` in
/// isolation but not this.
#[test]
fn the_filled_in_value_is_what_its_neighbours_tend_to() {
  let m = frozen(BSMCoc::Black1976);
  let eps = 1e-6;
  let at = m.price_call(S, S, R, Q, FUT_TAU);
  let below = m.price_call(S, S - eps, R, Q, FUT_TAU);
  let above = m.price_call(S, S + eps, R, Q, FUT_TAU);

  assert_eq!(at, 0.0);
  assert_eq!(above, at, "the limit from above");
  assert!(below > at, "monotone in K: {below}, {at}, {above}");
  assert!(
    below - at < 1e-5,
    "the limit from below must close too: {below} vs {at}"
  );
  // and the kink is real — the slope, not the value, is what jumps.
  assert!(
    below - at > 9e-7,
    "the deterministic payoff's kink must survive: {below} vs {at}"
  );
}

/// Away from the forward there is no `0/0`: `d₁` saturates to `±∞`, both
/// CDFs saturate with it, and every term collapses to discounted intrinsic
/// value on the forward. That is the right answer for a frozen underlying,
/// and it is not the branch's doing.
#[test]
fn zero_total_volatility_away_from_the_forward_is_discounted_intrinsic() {
  let m = frozen(BSMCoc::Black1976);
  let disc = (-R * FUT_TAU).exp();
  for &k in &[90.0, 110.0] {
    let (call, put) = m.call_put(S, k, R, Q, FUT_TAU);
    // Black1976 carries at b = 0, so the forward is S itself.
    assert!(
      (call - disc * (S - k).max(0.0)).abs() < TOL,
      "K={k}: call {call}"
    );
    assert!(
      (put - disc * (k - S).max(0.0)).abs() < TOL,
      "K={k}: put {put}"
    );
  }
}

/// The singularity is at the *forward*, not at the money — `Bsm1973` hits
/// the identical `0/0` once `r = 0` moves its forward onto the strike.
/// This is the pin that stops the fix being read as a `BSMCoc` special
/// case.
#[test]
fn carrying_conventions_hit_the_same_hole_at_their_own_forward() {
  let m = frozen(BSMCoc::Bsm1973);
  assert_eq!(m.price_call(S, S, 0.0, Q, FUT_TAU), 0.0);
  assert!(
    BSMPricer::new(0.0, BSMCoc::Bsm1973)
      .call_put(S, S, 0.0, Q, FUT_TAU)
      .0
      .is_nan(),
    "Bsm1973 at r = 0 must reach the same 0/0"
  );
}

/// Generalised put-call parity still holds at the filled-in point:
/// `b = 0` under `Black1976`, so `C - P = (S - K)e^{-rτ}`, which is `0` at
/// the money. A limit taken on only one leg would break this.
#[test]
fn parity_holds_at_the_filled_in_point() {
  let m = frozen(BSMCoc::Black1976);
  let (call, put) = m.call_put(S, S, R, Q, FUT_TAU);
  assert!((call - put).abs() < TOL, "call {call} vs put {put}");
}

/// The reachability claim, made falsifiable: with `σ_0 = d > 0` an ordinary
/// configuration has no degenerate term at all, so `term_call_put` must
/// hand back exactly what `BSMPricer` produced for every `n`, at the very
/// query — an at-the-money futures option — that used to be `NaN`.
///
/// Without this, "the branch is now unreachable in practice" would be a
/// claim with nothing checking it, and the branch could quietly start
/// intercepting live prices again.
#[test]
fn an_ordinary_configuration_never_reaches_the_branch() {
  let m = Merton1976Pricer::new(0.2, 0.5, 0.4, 10, BSMCoc::Black1976);
  for n in 0..m.m {
    let sigma = m.term_vol(n, FUT_TAU);
    assert!(sigma > 0.0, "term {n} volatility {sigma} is not positive");
    assert_eq!(
      m.term_call_put(n, FUT_TAU, S, S, R, Q),
      m.term_bsm(n, FUT_TAU).call_put(S, S, R, Q, FUT_TAU),
      "term {n} was intercepted"
    );
  }
  let call = m.price_call(S, S, R, Q, FUT_TAU);
  assert!(call.is_finite() && call > 0.0, "ATM futures call {call}");
}
