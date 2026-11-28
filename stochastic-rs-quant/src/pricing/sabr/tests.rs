use super::*;
use crate::OptionType;
use crate::traits::ModelPricer;

#[test]
fn sabr_pricer_basic() {
  let s = 3.724;
  let k = 3.8;
  let r = 0.065;
  let q = 0.022;
  let tau = 30.0 / 365.0;
  let pr = SabrPricer::new(SabrParams::new(0.11, 1.0, 0.5, 0.6));
  let (c, p) = (
    pr.price_call(s, k, r, q, tau),
    pr.price_put(s, k, r, q, tau),
  );
  assert!(c >= 0.0 && p >= 0.0);
  let d = pr.sabr_fx_forward_delta(s, k, r, q, tau, 1.0);
  assert!(d.is_finite());
}

/// A non-positive spot (hence forward) or strike is a query outside the domain: NaN, never a panic.
#[test]
fn sabr_pricer_sigma_is_nan_for_a_nonpositive_spot_or_strike() {
  let pricer = SabrPricer::new(SabrParams::new(0.11, 1.0, 0.5, 0.6));
  assert!(pricer.sigma(-3.724, 3.8, 0.065, 0.022, 0.5).is_nan());
  assert!(pricer.sigma(3.724, -3.8, 0.065, 0.022, 0.5).is_nan());
}

#[test]
fn sabr_price_is_nan_for_a_nonpositive_spot_or_strike() {
  let model = SabrPricer::new(SabrParams::new(0.2, 1.0, -0.3, 0.6));
  for (s, k) in [(-100.0, 100.0), (100.0, -10.0)] {
    assert!(model.price_call(s, k, 0.02, 0.0, 1.0).is_nan());
    assert!(model.price_put(s, k, 0.02, 0.0, 1.0).is_nan());
  }
}

/// Legal parameters inside `SabrCalibrator`'s box push Hagan's bracket `1 + (a + b + c)τ` negative
/// (`sigma = -0.3925`); both legs are then NaN, never `(0, 0)`.
#[test]
fn sabr_degenerate_hagan_vol_is_nan_on_both_legs() {
  let m = SabrPricer::new(SabrParams::new(0.2, 1.0, -0.9, 3.0));
  let sigma = m.sigma(100.0, 100.0, 0.0, 0.0, 10.0);
  assert!(
    sigma < 0.0,
    "fixture drifted: expansion no longer degenerates, sigma={sigma}"
  );

  let (call, put) = (
    m.price_call(100.0, 100.0, 0.0, 0.0, 10.0),
    m.price_put(100.0, 100.0, 0.0, 0.0, 10.0),
  );
  assert!(call.is_nan(), "call must be NaN, got {call}");
  assert!(put.is_nan(), "put must be NaN, got {put}");
  assert!(m.price_call(100.0, 100.0, 0.0, 0.0, 10.0).is_nan());
  assert!(m.price_put(100.0, 100.0, 0.0, 0.0, 10.0).is_nan());
}

/// At these parameters the expansion fails on the whole strike row (all `NaN`), and a `τ` it is
/// valid at brings every point back finite.
#[test]
fn sabr_degeneracy_tracks_tau_not_the_strike_grid() {
  let m = SabrPricer::new(SabrParams::new(0.2, 1.0, -0.9, 3.0));
  for &k in &[80.0, 100.0, 125.0] {
    assert!(
      m.price_call(100.0, k, 0.0, 0.0, 10.0).is_nan(),
      "K={k} at tau=10 must be NaN"
    );
    let short = m.price_call(100.0, k, 0.0, 0.0, 0.25);
    assert!(
      short.is_finite(),
      "K={k} at tau=0.25 must price, got {short}"
    );
  }
}

const S: f64 = 100.0;
const K: f64 = 105.0;
const R: f64 = 0.05;
const Q: f64 = 0.02;
const TAU: f64 = 0.75;

/// Cross-arch tolerance: `norm_cdf` and the Hagan expansion differ in the last bits between
/// aarch64-darwin and x86_64 under FMA contraction.
const TOL: f64 = 1e-12;

fn golden_model() -> SabrPricer {
  SabrPricer::new(SabrParams::new(0.11, 1.0, 0.5, 0.6))
}

/// Goldens of the call, the put and the four inherent accessors.
#[test]
fn sabr_model_pricer_matches_pre_refactor_goldens() {
  let m = golden_model();
  let (call, put) = (m.price_call(S, K, R, Q, TAU), m.price_put(S, K, R, Q, TAU));
  assert!((call - 2.8384773477990137).abs() < TOL, "call {call}");
  assert!((put - 5.46269724817904).abs() < TOL, "put {put}");
  assert_eq!(m.price_call(S, K, R, Q, TAU), call);
  assert_eq!(m.price_put(S, K, R, Q, TAU), put);

  let fwd = m.forward(S, R, Q, TAU);
  assert!((fwd - 102.27550341644461).abs() < TOL, "forward {fwd}");
  let sigma = m.sigma(S, K, R, Q, TAU);
  assert!((sigma - 0.11646804397344582).abs() < TOL, "sigma {sigma}");
  let delta = m.sabr_fx_forward_delta(S, K, R, Q, TAU, 1.0);
  assert!((delta - 0.38215989178925835).abs() < TOL, "delta {delta}");
  let iv = m.implied_volatility(3.0, S, K, R, Q, TAU, OptionType::Call);
  assert!((iv - 0.12131352686290568).abs() < TOL, "iv {iv}");
}

/// The carry is `b = r − q`, where parity is exact: it holds to `1e-12`, and `price_put` is the
/// Black-Scholes closed form.
#[test]
fn sabr_price_put_matches_parity_but_is_the_closed_form() {
  let m = golden_model();
  let (call, put) = (m.price_call(S, K, R, Q, TAU), m.price_put(S, K, R, Q, TAU));
  let parity = call - S * (-Q * TAU).exp() + K * (-R * TAU).exp();
  assert!((put - parity).abs() < TOL, "put {put} vs parity {parity}");
  assert_eq!(m.price_put(S, K, R, Q, TAU), put);
}

/// A price is Hagan's implied vol, then Black-Scholes at Merton (1973) carry, against a reference
/// composed by hand rather than through `SabrPricer`.
#[test]
fn sabr_price_call_is_hagan_vol_through_black_scholes() {
  use crate::pricing::bsm::BsmCoc;
  use crate::pricing::bsm::BsmPricer;

  let m = golden_model();
  let fwd = S * ((R - Q) * TAU).exp();
  let sigma = hagan_implied_vol(K, fwd, TAU, &m.params());
  let want = BsmPricer::new(sigma, BsmCoc::Merton1973).price_call(S, K, R, Q, TAU);
  assert_eq!(m.price_call(S, K, R, Q, TAU), want);
}

/// One model prices a whole strike/maturity grid.
#[test]
fn sabr_one_model_prices_a_grid() {
  let m = golden_model();
  for &tau in &[0.25, 0.5, 1.0] {
    let mut prev = f64::INFINITY;
    for &k in &[90.0, 100.0, 110.0] {
      let c = m.price_call(S, k, R, Q, tau);
      assert!(c.is_finite() && c < prev, "call must fall in strike");
      prev = c;
    }
  }
}

/// `(K, F, tau, alpha, beta, rho, nu, implied vol)`.
type HaganCase = (f64, f64, f64, f64, f64, f64, f64, f64);

/// Hagan (2002, Eq. A.69a) general-β implied vol — reference values.
#[test]
fn hagan_implied_vol_reference() {
  let cases: &[HaganCase] = &[
    (
      100.0,
      100.0,
      1.0,
      0.2,
      1.0,
      -0.3,
      0.5,
      2.021041666666667e-01,
    ),
    (
      110.0,
      100.0,
      1.0,
      0.2,
      1.0,
      -0.3,
      0.5,
      1.966695601513802e-01,
    ),
    (90.0, 100.0, 1.0, 0.2, 1.0, -0.3, 0.5, 2.118933616456034e-01),
    (100.0, 100.0, 0.5, 0.15, 0.5, -0.2, 0.8, 1.5373767578125e-02),
    (
      110.0,
      100.0,
      0.5,
      0.15,
      0.5,
      -0.2,
      0.8,
      3.080869461133284e-02,
    ),
    (
      90.0,
      100.0,
      0.5,
      0.15,
      0.5,
      -0.2,
      0.8,
      3.832319931343581e-02,
    ),
    (
      3.8,
      3.724,
      30.0 / 365.0,
      0.14,
      1.0,
      0.33,
      1.6,
      1.486360336704149e-01,
    ),
    (
      3.6,
      3.724,
      30.0 / 365.0,
      0.14,
      1.0,
      0.33,
      1.6,
      1.365590050371177e-01,
    ),
    (
      105.0,
      100.0,
      0.25,
      0.3,
      0.7,
      0.1,
      0.4,
      7.683910485737674e-02,
    ),
  ];

  for (i, &(k, f, t, alpha, beta, rho, nu, expected)) in cases.iter().enumerate() {
    let got = hagan_implied_vol(k, f, t, &SabrParams::new(alpha, beta, rho, nu));
    let err = (got - expected).abs();
    assert!(
      err < 1e-12,
      "case {}: got {:.15e}, expected {:.15e}, err={:.2e}",
      i,
      got,
      expected,
      err
    );
  }
}

/// α-from-ATM-vol cubic solver (Hagan 2002, Eq. A.69b) — reference values + round-trip.
#[test]
fn alpha_from_atm_vol_reference() {
  let cases: &[(f64, f64, f64, f64, f64, f64, f64)] = &[
    (0.20, 100.0, 1.0, 1.0, -0.3, 0.5, 1.979023350370119e-01),
    (0.15, 100.0, 0.5, 0.5, -0.2, 0.8, 1.465254087095464e+00),
    (
      0.1424,
      3.724,
      30.0 / 365.0,
      1.0,
      0.33,
      1.6,
      1.401312256794535e-01,
    ),
    (0.30, 50.0, 2.0, 0.7, 0.1, 0.4, 9.40944265414271e-01),
    (0.25, 100.0, 1.0, 0.0, 0.0, 0.3, 2.475118650781591e+01),
  ];

  for (i, &(v_atm, f, t, beta, rho, nu, expected)) in cases.iter().enumerate() {
    let got = alpha_from_atm_vol(v_atm, f, t, beta, rho, nu);
    let err = (got - expected).abs() / expected.abs();
    assert!(
      err < 1e-8,
      "case {}: got {:.15e}, expected {:.15e}, rel_err={:.2e}",
      i,
      got,
      expected,
      err
    );
    let vol_rt = hagan_implied_vol(f, f, t, &SabrParams::new(got, beta, rho, nu));
    let rt_err = (vol_rt - v_atm).abs();
    assert!(
      rt_err < 1e-10,
      "round-trip case {}: vol={:.15e}, target={:.15e}, err={:.2e}",
      i,
      vol_rt,
      v_atm,
      rt_err
    );
  }
}

/// Here (`β = 1`) eq. A.69b's quadratic has positive roots `α ≈ 0.3657`, near `α₀ = 0.15`, and a
/// spurious 0.3838 within 5%; both round-trip, so Newton's exact root is pinned.
#[test]
fn alpha_from_atm_vol_close_roots_reference() {
  let (v_atm, f, tau, beta, rho, nu) = (0.15, 100.0, 3.0, 1.0, -0.95, 1.5);
  let correct_root = 3.656720905235972e-1;
  let spurious_root = 3.838162135699706e-1;
  assert!(
    spurious_root / correct_root < 1.05,
    "fixture drifted: roots are no longer a close-roots hazard"
  );

  let got = alpha_from_atm_vol(v_atm, f, tau, beta, rho, nu);
  let err = (got - correct_root).abs() / correct_root;
  assert!(
    err < 1e-8,
    "solver did not return the economically meaningful root: got {got:.15e}, expected {correct_root:.15e}, rel_err={err:.2e}"
  );

  let vol_at_correct = hagan_implied_vol(f, f, tau, &SabrParams::new(correct_root, beta, rho, nu));
  assert!(
    (vol_at_correct - v_atm).abs() < 1e-10,
    "correct root should round-trip: got {vol_at_correct:.15e}"
  );
  let vol_at_spurious =
    hagan_implied_vol(f, f, tau, &SabrParams::new(spurious_root, beta, rho, nu));
  assert!(
    (vol_at_spurious - v_atm).abs() < 1e-10,
    "spurious root should *also* round-trip, which is exactly the hazard: got {vol_at_spurious:.15e}"
  );
}

/// [`hagan_implied_vol`] against a 40-digit mpmath reimplementation of Hagan eq. A.69a:
///
/// ```text
/// from mpmath import mp, mpf, log, sqrt
/// mp.dps = 40
/// logfk = log(F/K); fk = (F*K)**(1-beta)
/// a = (1-beta)**2*alpha**2/(24*fk)
/// b = rho*beta*nu*alpha/(4*sqrt(fk))
/// c = (2-3*rho**2)*nu**2/24
/// z = nu*sqrt(fk)*logfk/alpha
/// x = log((sqrt(1-2*rho*z+z*z)+z-rho)/(1-rho))
/// sigma = alpha*(z/x)*(1+(a+b+c)*T) / (sqrt(fk)*(1 + (1-beta)**2*logfk**2/24
///                                                 + (1-beta)**4*logfk**4/1920))
/// ```
///
/// All five agree to `1e-15` relative, so the `1e-14` tolerance is not tuned to pass.
#[test]
fn hagan_matches_an_independent_high_precision_implementation() {
  // (K, F, tau, alpha, beta, rho, nu, mpmath value at 40 digits)
  let cases: &[HaganCase] = &[
    (100.0, 100.0, 1.0, 0.2, 1.0, -0.3, 0.5, 0.20210416666666668),
    (110.0, 100.0, 1.0, 0.2, 1.0, -0.3, 0.5, 0.19666956015138031),
    (90.0, 100.0, 1.0, 0.2, 1.0, -0.3, 0.5, 0.21189336164560343),
    (
      100.0,
      100.0,
      0.5,
      0.15,
      0.5,
      -0.2,
      0.8,
      // mpmath: 0.015373767578124999… (same f64 as the shortest form below)
      0.015373767578125,
    ),
    (110.0, 100.0, 0.5, 0.15, 0.5, -0.2, 0.8, 0.03080869461133287),
  ];
  for &(k, f, tau, alpha, beta, rho, nu, expected) in cases {
    let got = hagan_implied_vol(k, f, tau, &SabrParams::new(alpha, beta, rho, nu));
    let rel = (got - expected).abs() / expected.abs();
    assert!(
      rel < 1e-14,
      "K={k}: got {got:e}, mpmath {expected:e}, rel {rel:e}"
    );
  }
}

/// The ATM `beta = 1` correction by hand: `b = ρνα/4 = −0.0075`, `c = (2 − 3ρ²)ν²/24`, so
/// `σ = α(1 + (b + c)τ) = 0.2021041666…`.
#[test]
fn atm_beta_one_matches_hand_computation() {
  let (alpha, rho, nu, tau) = (0.2, -0.3, 0.5, 1.0);
  let b = 0.25 * rho * nu * alpha;
  let c = (2.0 - 3.0 * rho * rho) * nu * nu / 24.0;
  let expected = alpha * (1.0 + (b + c) * tau);
  let got = hagan_implied_vol(100.0, 100.0, tau, &SabrParams::new(alpha, 1.0, rho, nu));
  assert!((got - expected).abs() < 1e-15, "got {got}, hand {expected}");
}

#[test]
#[should_panic(expected = "rho must satisfy `rho > -1.0 && rho < 1.0`, got rho = 1.0")]
fn rejects_rho_at_one() {
  let _ = hagan_implied_vol(
    110.0,
    100.0,
    1.0,
    &SabrParams {
      alpha: 0.2,
      beta: 1.0,
      rho: 1.0,
      nu: 0.5,
    },
  );
}

#[test]
fn a_zero_strike_is_nan() {
  assert!(hagan_implied_vol(0.0, 100.0, 1.0, &SabrParams::new(0.2, 1.0, -0.3, 0.5)).is_nan());
}

/// The parameter asserts run first, so an invalid `alpha` panics whatever the query.
#[test]
#[should_panic(expected = "alpha must satisfy `alpha.is_finite() && alpha > 0.0`, got alpha = 0.0")]
fn an_invalid_alpha_panics_even_at_a_query_outside_the_domain() {
  let _ = hagan_implied_vol(
    -1.0,
    100.0,
    1.0,
    &SabrParams {
      alpha: 0.0,
      beta: 1.0,
      rho: -0.3,
      nu: 0.5,
    },
  );
}

#[test]
#[should_panic(expected = "alpha must satisfy `alpha.is_finite() && alpha > 0.0`, got alpha = 0.0")]
fn rejects_a_nonpositive_alpha() {
  let _ = hagan_implied_vol(
    110.0,
    100.0,
    1.0,
    &SabrParams {
      alpha: 0.0,
      beta: 1.0,
      rho: -0.3,
      nu: 0.5,
    },
  );
}

#[test]
fn alpha_from_atm_vol_is_nan_at_a_nonpositive_forward() {
  assert!(alpha_from_atm_vol(0.2, 0.0, 1.0, 1.0, -0.3, 0.5).is_nan());
}

#[test]
fn alpha_from_atm_vol_is_nan_at_a_nonpositive_v_atm() {
  assert!(alpha_from_atm_vol(0.0, 100.0, 1.0, 1.0, -0.3, 0.5).is_nan());
}

#[test]
#[should_panic(expected = "rho must satisfy `rho > -1.0 && rho < 1.0`, got rho = 1.0")]
fn alpha_from_atm_vol_rejects_rho_at_one() {
  let _ = alpha_from_atm_vol(0.2, 100.0, 1.0, 1.0, 1.0, 0.5);
}

/// `SabrPricer::new` validates all four parameters through `SabrParams::new`; an invalid `beta = 5`
/// would price at the spot.
mod construction_validation {
  use super::*;

  #[test]
  #[should_panic(
    expected = "alpha must satisfy `alpha.is_finite() && alpha > 0.0`, got alpha = -0.2"
  )]
  fn new_rejects_non_positive_alpha() {
    let _ = SabrPricer::new(SabrParams {
      alpha: -0.2,
      beta: 1.0,
      rho: -0.3,
      nu: 0.4,
    });
  }

  #[test]
  #[should_panic(expected = "beta must satisfy `(0.0..=1.0).contains(&beta)`, got beta = 5.0")]
  fn new_rejects_cev_exponent_above_one() {
    let _ = SabrPricer::new(SabrParams {
      alpha: 0.2,
      beta: 5.0,
      rho: -0.3,
      nu: 0.4,
    });
  }

  #[test]
  #[should_panic(expected = "beta must satisfy `(0.0..=1.0).contains(&beta)`, got beta = -1.0")]
  fn new_rejects_negative_cev_exponent() {
    let _ = SabrPricer::new(SabrParams {
      alpha: 0.2,
      beta: -1.0,
      rho: -0.3,
      nu: 0.4,
    });
  }

  #[test]
  #[should_panic(expected = "nu must satisfy `nu.is_finite() && nu >= 0.0`, got nu = -0.4")]
  fn new_rejects_negative_vol_of_vol() {
    let _ = SabrPricer::new(SabrParams {
      alpha: 0.2,
      beta: 1.0,
      rho: -0.3,
      nu: -0.4,
    });
  }

  #[test]
  #[should_panic(expected = "rho must satisfy `rho > -1.0 && rho < 1.0`, got rho = 5.0")]
  fn new_rejects_out_of_range_rho() {
    let _ = SabrPricer::new(SabrParams {
      alpha: 0.2,
      beta: 1.0,
      rho: 5.0,
      nu: 0.4,
    });
  }

  /// `rho = ±1` is refused at construction as well, the open bound `hagan_implied_vol`'s `1 − ρ`
  /// denominator sets.
  #[test]
  #[should_panic(expected = "rho must satisfy `rho > -1.0 && rho < 1.0`, got rho = 1.0")]
  fn new_rejects_unit_rho_like_the_expansion_does() {
    let _ = SabrPricer::new(SabrParams {
      alpha: 0.2,
      beta: 1.0,
      rho: 1.0,
      nu: 0.4,
    });
  }

  /// `SabrCalibrator`'s projection box constructs (`alpha`, `nu ≥ 1e-6`, `beta ∈ [0, 1]`,
  /// `rho = ±0.9999`), so no legal iterate aborts a calibration.
  #[test]
  fn the_calibrators_projection_box_stays_constructible() {
    for &(alpha, beta, rho, nu) in &[
      (1e-6, 0.0, -0.9999, 1e-6),
      (1e-6, 1.0, 0.9999, 1e-6),
      (0.2, 0.5, 0.0, 0.0),
    ] {
      let p = SabrParams {
        alpha,
        beta,
        rho,
        nu,
      };
      assert_eq!(SabrPricer::new(p).params(), p);
    }
  }
}

/// Hagan's volatility is negative at `α 0.2, β 1, ρ −0.9, ν 3` and `τ = 10`, so the price is NaN,
/// as `SabrPricer`'s is, where `BsmPricer::new` panicked; `bs_price_fx` reads `sigma` as BSM does.
#[test]
fn a_negative_hagan_volatility_prices_nan_not_a_panic() {
  let params = SabrParams {
    alpha: 0.2,
    beta: 1.0,
    rho: -0.9,
    nu: 3.0,
  };
  let (call, put) = model_price_hagan(100.0, 100.0, 0.05, 0.0, 10.0, &params);
  assert!(call.is_nan() && put.is_nan(), "({call}, {put})");
  let pricer = SabrPricer::new(params);
  assert!(pricer.price_call(100.0, 100.0, 0.05, 0.0, 10.0).is_nan());
  for sigma in [
    -1.3502217069673441,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
  ] {
    let (call, put) = bs_price_fx(100.0, 100.0, 0.05, 0.0, 10.0, sigma);
    assert!(call.is_nan() && put.is_nan(), "{sigma}: ({call}, {put})");
  }
  let (call, put) = bs_price_fx(100.0, 100.0, 0.05, 0.0, 10.0, 0.0);
  assert!((call - (100.0 - 100.0 * (-0.5_f64).exp())).abs() < 1e-12 && put == 0.0);
  let (call, put) = model_price_hagan(100.0, 100.0, 0.05, 0.0, 1.0, &params);
  assert!(call > 0.0 && put > 0.0, "({call}, {put})");
}
