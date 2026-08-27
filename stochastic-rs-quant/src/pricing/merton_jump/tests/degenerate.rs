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
//! which `new` accepts on purpose — and the pure-jump corner `gamma = 1`,
//! at the `n = 0` term alone. The corner used to reach it only about half
//! the time: `diffusive_std` computed `v² − λz²` after round-tripping `z`
//! through a `sqrt`, so `(v, λ) = (0.5, 1)` landed on `d = 0` while
//! `(0.2, 0.5)` landed one ulp below and returned `NaN`. `λz²` is `v²γ` by
//! construction, so it is now taken directly and the corner is `0` for
//! every intensity;
//! `the_pure_jump_corner_is_zero_for_every_intensity` pins that, and
//! `an_inadmissible_gamma_still_announces_itself` pins the `NaN` the
//! direct form must not silence along with it.
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

/// The pure-jump corner made deterministic: at `gamma = 1` the whole
/// variance is jump variance, so `d² = v² − λz²` is `0` exactly — for
/// every intensity, not for the half of them whose `sqrt` round-trip
/// happened to land on zero rather than one ulp below it.
///
/// The six `(v, λ)` pairs are the reported split: the first four returned
/// `0` before the fix and the last two returned `NaN`, from the same model
/// and the same `gamma`. The sweep is what makes "every intensity" a
/// falsifiable claim rather than six lucky draws.
#[test]
fn the_pure_jump_corner_is_zero_for_every_intensity() {
  for &(v, lambda) in &[
    (0.5, 1.0),
    (0.2, 1.0),
    (0.3, 0.5),
    (0.2, 0.25),
    (0.2, 0.5),
    (0.25, 2.0),
  ] {
    let m = Merton1976Pricer::new(v, lambda, 1.0, 20, BSMCoc::Bsm1973);
    assert_eq!(m.diffusive_std(), 0.0, "v={v} lambda={lambda}");
    let call = m.price_call(S, K, R, Q, TAU);
    assert!(call.is_finite(), "v={v} lambda={lambda} priced {call}");
  }

  for i in 1..=400 {
    let lambda = f64::from(i) * 0.025;
    for &v in &[0.05, 0.2, 0.35, 1.0] {
      let m = Merton1976Pricer::new(v, lambda, 1.0, 20, BSMCoc::Bsm1973);
      assert_eq!(m.diffusive_std(), 0.0, "v={v} lambda={lambda}");
    }
  }
}

/// The `NaN` the direct form would have silenced.
///
/// `gamma` and `lambda` of opposite sign make `z = √(v²γ/λ)` imaginary,
/// and `v²(1−γ)` is a perfectly finite number at exactly those points —
/// so without `diffusive_std`'s realness branch an inadmissible `gamma`
/// would price as confidently as an admissible one. `new` documents that
/// a `gamma` outside `[0, 1]` announces itself; this is the pin. The
/// third case is the other half of that claim, where the announcement
/// comes from the subtraction rather than from `z`.
#[test]
fn an_inadmissible_gamma_still_announces_itself() {
  for &(lambda, gamma) in &[(0.5, -0.25), (-0.5, 0.4), (0.5, 1.5)] {
    let m = Merton1976Pricer::new(0.2, lambda, gamma, 20, BSMCoc::Bsm1973);
    assert!(
      m.diffusive_std().is_nan(),
      "lambda={lambda} gamma={gamma} diffusive_std {}",
      m.diffusive_std()
    );
    let call = m.price_call(S, K, R, Q, TAU);
    assert!(call.is_nan(), "lambda={lambda} gamma={gamma} priced {call}");
  }
}
