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
//! The **Greeks** need the same treatment nine times over, and the limit is
//! not the same one twice. Away from the forward the `1/σ`-shaped closed
//! forms tend to `0` while the rest are already saturated; at the forward
//! `delta → ±½e^{(b−r)τ}`, `rho → ±½Kτe^{−rτ}` and `gamma → +∞`, and only
//! the price — which is what the six bump-based Greeks difference — tends
//! to `0` there. A single floor answered `0` for all of them, which was
//! right away from the forward, right for `theta` everywhere, and wrong for
//! `delta`, `gamma` and `rho` at the singular strike.
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
    let term = m.term_bsm(n, FUT_TAU);
    assert_eq!(
      Merton1976Pricer::term_call_put(&term, S, S, R, Q, FUT_TAU),
      term.call_put(S, S, R, Q, FUT_TAU),
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

/// The half of the degenerate term's Greeks that was always exact, and the
/// reason `greek_series` needed a per-Greek limit rather than a wider floor.
///
/// Away from the forward a degenerate term's `d₁` saturates to `±∞`, so the
/// `1/v`-shaped closed forms are `0/0` and their `σ → 0⁺` limits really are
/// `0`: `norm_pdf(d₁)` decays like `e^{-c²/2σ²}` and beats the linear
/// `1/v`. The Greeks that do not vanish there — `delta → e^{(b−r)τ}`, `rho`
/// — never went `NaN` in the first place, so they keep their closed forms
/// and are asserted against those rather than against `0`.
#[test]
fn a_degenerate_term_away_from_the_forward_is_its_saturated_limit() {
  let m = frozen(BSMCoc::Black1976);
  let carry = (-R * FUT_TAU).exp();
  for &k in &[90.0_f64, 110.0] {
    let in_the_money = if k < S { 1.0 } else { 0.0 };
    let g = m.greeks(S, k, R, Q, FUT_TAU, OT);
    assert_eq!(g.gamma, 0.0, "K={k} gamma");
    assert_eq!(g.vanna, 0.0, "K={k} vanna");
    assert_eq!(g.volga, 0.0, "K={k} volga");
    assert!(
      (g.delta - carry * in_the_money).abs() < TOL,
      "K={k}: delta {} must be the saturated closed form",
      g.delta
    );
    // And it is a limit, not a poison check: the underlying BSM term is
    // genuinely `NaN` here, so without it the whole sum would be.
    assert!(
      BSMPricer::new(0.0, BSMCoc::Black1976)
        .gamma(S, k, R, Q, FUT_TAU)
        .is_nan(),
      "K={k}: the zero-vol term's gamma must be NaN, else this pins nothing"
    );
    assert_eq!(
      Merton1976Pricer::term_regime(&m.term_bsm(0, FUT_TAU), S, k, R, Q, FUT_TAU),
      TermRegime::Saturated,
      "K={k} must reach the saturated arm, not the forward one"
    );
  }
}

/// The half that used to be **wrong**, now the limit it should always have
/// been. This replaces a test that asserted `0.0` for three Greeks whose
/// limits are not zero, alongside the right answers, so that the gap could
/// not drift while it went unfixed.
///
/// At the forward a degenerate term's `d₁` is `0/0`, so every closed form
/// is `NaN` and the retired floor answered `0.0` for all of them. Three of
/// those answers were wrong. Both CDFs converge to `½` as `σ → 0⁺` along
/// `Se^{bτ} = K` — `d₁ = σ√τ/2 → 0⁺`, `d₂ = -σ√τ/2 → 0⁻` — which gives
/// `delta → ±½e^{(b−r)τ}` and `rho → ±½Kτe^{−rτ}`, while
/// `gamma = φ(d₁)/(Sσ√τ)` holds a strictly positive numerator over a
/// vanishing `σ` and diverges like `1/σ`.
///
/// The closed forms below would only restate the implementation, so the
/// sweep is what adjudicates them: it reaches each value from models with
/// `v > 0`, which have no degenerate term and never enter the branch. The
/// gaps close **linearly in `σ`** — a decade of `v` buys a decade of
/// accuracy — which is a stronger statement than "close enough", and `σΓ`
/// is constant to ten figures across four decades, which is the signature
/// of a pole rather than of a merely large number.
///
/// `theta` is the one the floor got right, for a reason that does not
/// generalise: what the bump-based Greeks difference is a *price*, whose
/// forward limit genuinely is `½(Se^{(b−r)τ} − Ke^{−rτ}) = 0`.
#[test]
fn the_forward_point_greeks_of_a_degenerate_term_are_their_limits() {
  let m = frozen(BSMCoc::Black1976);
  // Black1976 carries at b = 0, so the forward is S itself and the carry
  // factor coincides with the discount factor.
  let disc = (-R * FUT_TAU).exp();
  let want_delta = 0.5 * disc;
  let want_rho = 0.5 * S * FUT_TAU * disc;

  let call = m.greeks(S, S, R, Q, FUT_TAU, OT);
  assert!(
    (call.delta - want_delta).abs() < TOL,
    "call delta {} must be the limit {want_delta}, not 0",
    call.delta
  );
  assert!(
    (call.rho - want_rho).abs() < TOL,
    "call rho {} must be the limit {want_rho}, not 0",
    call.rho
  );
  assert!(
    call.gamma.is_infinite() && call.gamma.is_sign_positive(),
    "gamma must be the +inf its limit is, not 0: {}",
    call.gamma
  );
  assert_eq!(call.theta, -0.0, "theta at the forward is the price limit");

  // `delta` and `rho` keep an `option_type`, so both limits flip sign;
  // `gamma` does not take one and must be the same divergence on both legs.
  let put = m.greeks(S, S, R, Q, FUT_TAU, OptionType::Put);
  assert!(
    (put.delta + want_delta).abs() < TOL,
    "put delta {} must be the negated limit",
    put.delta
  );
  assert!(
    (put.rho + want_rho).abs() < TOL,
    "put rho {} must be the negated limit",
    put.rho
  );
  assert_eq!(put.gamma, call.gamma, "gamma takes no option_type");

  // What they are the limits *of*, measured along `sigma -> 0+` from models
  // that never reach the degenerate branch.
  let mut prev_pole = f64::NAN;
  for (i, &v) in [1e-3_f64, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8].iter().enumerate() {
    let near = Merton1976Pricer::new(v, 0.5, 0.4, 20, BSMCoc::Black1976);
    let delta = near.delta(S, S, R, Q, FUT_TAU, OT);
    let rho = near.rho(S, S, R, Q, FUT_TAU, OT);
    let gamma = near.gamma(S, S, R, Q, FUT_TAU);
    assert!(
      (delta - want_delta).abs() < 0.2 * v,
      "v={v}: delta {delta} must close on {want_delta} linearly in v"
    );
    assert!(
      (rho - want_rho).abs() < 10.0 * v,
      "v={v}: rho {rho} must close on {want_rho} linearly in v"
    );
    // A `1/sigma` pole has a constant residue; a large finite number does
    // not. `sigma_n = v*sqrt(0.6 + 1.6n)` here, so `v` is the scale of the
    // whole series rather than of one term, and the product is constant to
    // the precision the pole is resolved at.
    let pole = gamma * v;
    if i > 2 {
      assert!(
        (pole - prev_pole).abs() < 1e-8 * pole,
        "v={v}: sigma*gamma must be constant, {prev_pole} -> {pole}"
      );
    }
    assert!(pole > 0.0, "v={v}: the residue must be positive: {pole}");
    prev_pole = pole;
  }
}

/// One degenerate term at the forward is enough to move a model that is
/// **otherwise ordinary**, which is what makes this a live defect rather
/// than a curiosity of a frozen underlying.
///
/// The pure-jump corner `gamma == 1` puts the whole variance in the jumps,
/// so `d = 0` and the `n = 0` term alone is degenerate; the other nineteen
/// have `σ_n = z√(n/τ) > 0` and price normally. Under `Black1976` the
/// forward is `S`, so the at-the-money strike is the singular one and the
/// option is one a desk actually quotes. The price there was already right;
/// `delta`, `gamma` and `rho` were each short one `w₀`-weighted term.
///
/// Adjudicated by letting `gamma → 1⁻`, which shrinks `σ_0 = v√(1-γ)`
/// towards zero through models with **no** degenerate term at all: every
/// value below is the limit of that family, and `gamma` diverges along it
/// with a constant `σ_0·Γ` residue.
#[test]
fn one_degenerate_term_at_the_forward_moves_an_otherwise_ordinary_model() {
  let m = Merton1976Pricer::new(0.2, 0.5, 1.0, 20, BSMCoc::Black1976);
  assert_eq!(
    Merton1976Pricer::term_regime(&m.term_bsm(0, FUT_TAU), S, S, R, Q, FUT_TAU),
    TermRegime::AtTheForward,
    "the n = 0 term must be the degenerate one"
  );
  for n in 1..m.m {
    assert!(
      m.term_vol(n, FUT_TAU) > 0.0,
      "term {n} must be ordinary, so this is not a frozen model"
    );
  }
  let price = m.price_call(S, S, R, Q, FUT_TAU);
  assert!(price.is_finite() && price > 0.0, "a live price: {price}");

  let g = m.greeks(S, S, R, Q, FUT_TAU, OT);
  assert!(
    g.gamma.is_infinite() && g.gamma.is_sign_positive(),
    "one degenerate term at the forward is enough to make gamma diverge: {}",
    g.gamma
  );

  // The limit of the non-degenerate family that approaches this model.
  let approaching = |one_minus_gamma: f64| {
    Merton1976Pricer::new(0.2, 0.5, 1.0 - one_minus_gamma, 20, BSMCoc::Black1976)
  };
  let near = approaching(1e-12);
  assert!(
    near.term_vol(0, FUT_TAU) > 0.0,
    "the approaching family must not itself be degenerate"
  );
  assert!(
    (near.delta(S, S, R, Q, FUT_TAU, OT) - g.delta).abs() < 1e-6,
    "delta {} is not the gamma -> 1- limit {}",
    g.delta,
    near.delta(S, S, R, Q, FUT_TAU, OT)
  );
  assert!(
    (near.rho(S, S, R, Q, FUT_TAU, OT) - g.rho).abs() < 1e-4,
    "rho {} is not the gamma -> 1- limit {}",
    g.rho,
    near.rho(S, S, R, Q, FUT_TAU, OT)
  );
  assert!(
    (near.price_call(S, S, R, Q, FUT_TAU) - price).abs() < 1e-4,
    "price {price} is not the gamma -> 1- limit {}",
    near.price_call(S, S, R, Q, FUT_TAU)
  );

  // And the divergence has a residue, so it is a pole and not an overflow.
  let mut prev_pole = f64::NAN;
  for (i, e) in [8, 9, 10, 11, 12].iter().enumerate() {
    let n = approaching(10f64.powi(-e));
    let pole = n.term_vol(0, FUT_TAU) * n.gamma(S, S, R, Q, FUT_TAU);
    if i > 0 {
      assert!(
        (pole - prev_pole).abs() < 1e-4 * pole,
        "1-gamma=1e-{e}: sigma_0*gamma must be constant, {prev_pole} -> {pole}"
      );
    }
    prev_pole = pole;
  }
}

/// The Greeks' idea of the price and the price are the same function.
///
/// Every bump-based Greek differences `series_price`, which prices each
/// term through `term_call_put` — the same call `call_put` makes — rather
/// than through a second copy of the degenerate term's forward limit. Two
/// copies of that expression could drift; this is what stops them being two.
#[test]
fn the_greeks_price_a_degenerate_term_exactly_as_the_price_does() {
  let corner = Merton1976Pricer::new(0.2, 0.5, 1.0, 20, BSMCoc::Black1976);
  for m in [frozen(BSMCoc::Black1976), corner] {
    for &k in &[S, 90.0, 110.0] {
      let (call, put) = m.call_put(S, k, R, Q, FUT_TAU);
      assert_eq!(
        m.series_price(S, k, R, Q, FUT_TAU, OptionType::Call),
        call,
        "K={k} call"
      );
      assert_eq!(
        m.series_price(S, k, R, Q, FUT_TAU, OptionType::Put),
        put,
        "K={k} put"
      );
    }
  }
}

/// The one place the per-Greek limits stop, and why they stop there.
///
/// At `λ == 0` `greek_series` hands the single surviving term to
/// `BSMPricer` directly, and the six bump-based Greeks skip the series
/// altogether for the same reason. A model that is *both* frozen and
/// jump-free therefore splits: `delta`, `gamma`, `rho` and the price go
/// through this crate's own code and get their limits, while the other six
/// get `BSMPricer`'s undefended `NaN`.
///
/// That asymmetry belongs to `BSMPricer`, not to this pricer — the same
/// query prices `NaN` on a bare one, which
/// `zero_total_volatility_at_the_forward_is_the_limit` already asserts.
/// Pinned rather than fixed: fixing it means giving `BSMPricer` a forward
/// limit of its own, which would move every model built on it.
#[test]
fn a_frozen_no_jump_model_inherits_black_scholess_own_gaps() {
  let m = Merton1976Pricer::new(0.0, 0.0, 0.4, 20, BSMCoc::Black1976);
  let disc = (-R * FUT_TAU).exp();
  assert_eq!(m.price_call(S, S, R, Q, FUT_TAU), 0.0, "price");
  let delta = m.delta(S, S, R, Q, FUT_TAU, OT);
  assert!((delta - 0.5 * disc).abs() < TOL, "delta {delta}");
  let rho = m.rho(S, S, R, Q, FUT_TAU, OT);
  assert!((rho - 0.5 * S * FUT_TAU * disc).abs() < TOL, "rho {rho}");
  assert!(m.gamma(S, S, R, Q, FUT_TAU).is_infinite(), "gamma");
  for (name, v) in [
    ("vega", m.vega(S, S, R, Q, FUT_TAU)),
    ("theta", m.theta(S, S, R, Q, FUT_TAU, OT)),
    ("vanna", m.vanna(S, S, R, Q, FUT_TAU)),
    ("charm", m.charm(S, S, R, Q, FUT_TAU, OT)),
    ("volga", m.volga(S, S, R, Q, FUT_TAU)),
    ("veta", m.veta(S, S, R, Q, FUT_TAU)),
  ] {
    assert!(v.is_nan(), "{name} must still be BSMPricer's NaN, got {v}");
  }
}
