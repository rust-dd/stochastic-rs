//! A `σ_n = 0` term (at `sigma = 0`, or `n = 0` at `gamma = 1`) is `0/0` at the forward, a
//! removable singularity: both CDFs tend to `½`, so its limit there is `0`, not `NaN`.

use super::*;

const FUT_TAU: f64 = 0.5;

/// Zero total volatility: no diffusion and no jump size, so every term of
/// the series is degenerate.
fn frozen(coc: BsmCoc) -> Merton1976Pricer {
  Merton1976Pricer::new(0.0, 0.5, 0.4, 20, coc)
}

/// An at-the-money futures option on a frozen underlying prices the limit where its bare
/// `BsmPricer` term is `NaN`.
#[test]
fn zero_total_volatility_at_the_forward_is_the_limit() {
  for coc in [BsmCoc::Black1976, BsmCoc::Asay1982] {
    let m = frozen(coc);
    let CallPut { call, put } = m.price_call_put(S, S, R, Q, FUT_TAU);
    assert_eq!(call, 0.0, "{coc:?} call");
    assert_eq!(put, 0.0, "{coc:?} put");

    let bsm = BsmPricer::new(0.0, coc);
    let raw = (
      bsm.price_call(S, S, R, Q, FUT_TAU),
      bsm.price_put(S, S, R, Q, FUT_TAU),
    );
    assert!(
      raw.0.is_nan() && raw.1.is_nan(),
      "{coc:?}: the underlying zero-vol term must be NaN, else this pins nothing: {raw:?}"
    );
  }
}

/// The filled-in `0` is the limit of its neighbours: strikes `1e-6` either side of the forward are
/// worth `9.75e-7` in the money and nothing out of it.
#[test]
fn the_filled_in_value_is_what_its_neighbours_tend_to() {
  let m = frozen(BsmCoc::Black1976);
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

/// Off the forward `d₁` saturates to `±∞`, so every term is discounted intrinsic value on the
/// forward without the branch.
#[test]
fn zero_total_volatility_away_from_the_forward_is_discounted_intrinsic() {
  let m = frozen(BsmCoc::Black1976);
  let disc = (-R * FUT_TAU).exp();
  for &k in &[90.0, 110.0] {
    let CallPut { call, put } = m.price_call_put(S, k, R, Q, FUT_TAU);
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
#[test]
fn carrying_conventions_hit_the_same_hole_at_their_own_forward() {
  let m = frozen(BsmCoc::Bsm1973);
  assert_eq!(m.price_call(S, S, 0.0, Q, FUT_TAU), 0.0);
  assert!(
    BsmPricer::new(0.0, BsmCoc::Bsm1973)
      .price_call(S, S, 0.0, Q, FUT_TAU)
      .is_nan(),
    "Bsm1973 at r = 0 must reach the same 0/0"
  );
}

/// Generalised parity holds at the filled-in point: `C − P = (S − K)e^{−rτ} = 0` at the money
/// under `Black1976`, which a limit on one leg would break.
#[test]
fn parity_holds_at_the_filled_in_point() {
  let m = frozen(BsmCoc::Black1976);
  let CallPut { call, put } = m.price_call_put(S, S, R, Q, FUT_TAU);
  assert!((call - put).abs() < TOL, "call {call} vs put {put}");
}

/// With `σ_0 = d > 0` no term is degenerate, so `term_call_put` returns exactly `BsmPricer`'s value
/// for every `n`, at an at-the-money futures option as well.
#[test]
fn an_ordinary_configuration_never_reaches_the_branch() {
  let m = Merton1976Pricer::new(0.2, 0.5, 0.4, 10, BsmCoc::Black1976);
  for n in 0..m.m {
    let sigma = m.term_vol(n, FUT_TAU);
    assert!(sigma > 0.0, "term {n} volatility {sigma} is not positive");
    let term = m.term_bsm(n, FUT_TAU);
    assert_eq!(
      Merton1976Pricer::term_call_put(&term, S, S, R, Q, FUT_TAU),
      CallPut {
        call: term.price_call(S, S, R, Q, FUT_TAU),
        put: term.price_put(S, S, R, Q, FUT_TAU)
      },
      "term {n} was intercepted"
    );
  }
  let call = m.price_call(S, S, R, Q, FUT_TAU);
  assert!(call.is_finite() && call > 0.0, "ATM futures call {call}");
}

/// At `gamma = 1` the whole variance is jump variance, so `d² = σ² − λz²` is `0` exactly for every
/// `(σ, λ)` in the sweep.
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
    let m = Merton1976Pricer::new(v, lambda, 1.0, 20, BsmCoc::Bsm1973);
    assert_eq!(m.diffusive_std(), 0.0, "v={v} lambda={lambda}");
    let call = m.price_call(S, K, R, Q, TAU);
    assert!(call.is_finite(), "v={v} lambda={lambda} priced {call}");
  }

  for i in 1..=400 {
    let lambda = f64::from(i) * 0.025;
    for &v in &[0.05, 0.2, 0.35, 1.0] {
      let m = Merton1976Pricer::new(v, lambda, 1.0, 20, BsmCoc::Bsm1973);
      assert_eq!(m.diffusive_std(), 0.0, "v={v} lambda={lambda}");
    }
  }
}

/// An inadmissible `(lambda, gamma)` in a literal that bypasses `new` prices NaN: `√(σ²γ/λ)` or
/// `√(σ² − σ²γ)` is not real there, where the direct `σ²(1 − γ)` gives a finite number.
#[test]
fn an_inadmissible_gamma_still_announces_itself() {
  for &(lambda, gamma) in &[(0.5, -0.25), (-0.5, 0.4), (0.5, 1.5)] {
    let m = Merton1976Pricer {
      sigma: 0.2,
      lambda,
      gamma,
      m: 20,
      b: BsmCoc::Bsm1973,
    };
    assert!(
      m.diffusive_std().is_nan(),
      "lambda={lambda} gamma={gamma} diffusive_std {}",
      m.diffusive_std()
    );
    let call = m.price_call(S, K, R, Q, TAU);
    assert!(call.is_nan(), "lambda={lambda} gamma={gamma} priced {call}");
  }
}

/// Off the forward a `σ_n = 0` term's `1/σ`-shaped Greeks tend to `0`, and `delta` and `rho` keep
/// their saturated closed forms (`delta → e^{(b−r)τ}`).
#[test]
fn a_degenerate_term_away_from_the_forward_is_its_saturated_limit() {
  let m = frozen(BsmCoc::Black1976);
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
      BsmPricer::new(0.0, BsmCoc::Black1976)
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

/// At the forward `delta → ±½e^{(b−r)τ}`, `rho → ±½Kτe^{−rτ}` and `gamma → +∞`, approached from
/// `sigma > 0` with gaps closing linearly in σ; `theta`, a price difference, tends to `0`.
#[test]
fn the_forward_point_greeks_of_a_degenerate_term_are_their_limits() {
  let m = frozen(BsmCoc::Black1976);
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
    let near = Merton1976Pricer::new(v, 0.5, 0.4, 20, BsmCoc::Black1976);
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
    // A `1/σ` pole has a constant residue and a large finite number does not; `v`, the model's
    // `sigma`, scales the whole series, so `gamma · v` is constant to the pole's resolution.
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

/// At `gamma = 1` only the `n = 0` term is degenerate, yet at the money under `Black1976` its limit
/// moves `delta`, `gamma` and `rho`; the values are the limits of `gamma → 1⁻`.
#[test]
fn one_degenerate_term_at_the_forward_moves_an_otherwise_ordinary_model() {
  let m = Merton1976Pricer::new(0.2, 0.5, 1.0, 20, BsmCoc::Black1976);
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
    Merton1976Pricer::new(0.2, 0.5, 1.0 - one_minus_gamma, 20, BsmCoc::Black1976)
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

/// The bump-based Greeks difference `series_price`, which prices each term through
/// `term_call_put` as the price does, so the degenerate limit has one copy.
#[test]
fn the_greeks_price_a_degenerate_term_exactly_as_the_price_does() {
  let corner = Merton1976Pricer::new(0.2, 0.5, 1.0, 20, BsmCoc::Black1976);
  for m in [frozen(BsmCoc::Black1976), corner] {
    for &k in &[S, 90.0, 110.0] {
      let CallPut { call, put } = m.price_call_put(S, k, R, Q, FUT_TAU);
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

/// At `λ = 0` a frozen model's `delta`, `gamma`, `rho` and price get their limits while the other
/// six Greeks keep `BsmPricer`'s `NaN`, pinned: a `BsmPricer` limit would move every model on it.
#[test]
fn a_frozen_no_jump_model_inherits_black_scholess_own_gaps() {
  let m = Merton1976Pricer::new(0.0, 0.0, 0.4, 20, BsmCoc::Black1976);
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
    assert!(v.is_nan(), "{name} must still be BsmPricer's NaN, got {v}");
  }
}
