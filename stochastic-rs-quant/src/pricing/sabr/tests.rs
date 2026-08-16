use super::*;
use crate::traits::PricerExt;

#[test]
fn sabr_pricer_basic() {
  let s = 3.724;
  let k = 3.8;
  let r = 0.065;
  let q = Some(0.022);
  let tau = 30.0 / 365.0;
  let pr = SabrPricer::new(s, k, r, q, 0.11, 1.0, 0.6, 0.5, Some(tau), None, None);
  let (c, p) = pr.calculate_call_put();
  println!("Call: {}, Put: {}", c, p);
  assert!(c >= 0.0 && p >= 0.0);
  let d = pr.sabr_fx_forward_delta(1.0);
  assert!(d.is_finite());
}

/// Hagan (2002, Eq. A.69a) general-β implied vol — reference values.
#[test]
fn hagan_implied_vol_reference() {
  let cases: &[(f64, f64, f64, f64, f64, f64, f64, f64)] = &[
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
    let got = hagan_implied_vol(k, f, t, alpha, beta, nu, rho);
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
    let vol_rt = hagan_implied_vol(f, f, t, got, beta, nu, rho);
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

/// A hazard case for [`alpha_from_atm_vol`]'s root selection: at these
/// parameters the Hagan Eq. A.69b quadratic (β = 1, so the cubic's α³ term
/// vanishes) has two positive real roots within 5% of each other —
/// α ≈ 0.3657 (economically meaningful, near the zeroth-order estimate
/// `α₀ = σ_ATM · F^{1−β} = 0.15`) and α ≈ 0.3838 (spurious). Found by a
/// grid search over (β, ρ, ν, τ, F, σ_ATM) for the closest positive-root
/// pair reachable within this crate's own SABR-smile calibrator bounds
/// (`ρ ∈ [-0.99, 0.99]`, `ν ∈ [0.01, 10]`, `vol_surface::sabr_smile`).
///
/// Both roots round-trip through [`hagan_implied_vol`] back to `v_atm` —
/// proving a round-trip check alone (as in `alpha_from_atm_vol_reference`
/// above) cannot distinguish "found *a* root" from "found the *correct*
/// root" — so this pins the exact value Newton must return, not just that
/// *some* value satisfies the round trip.
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

  let vol_at_correct = hagan_implied_vol(f, f, tau, correct_root, beta, nu, rho);
  assert!(
    (vol_at_correct - v_atm).abs() < 1e-10,
    "correct root should round-trip: got {vol_at_correct:.15e}"
  );
  let vol_at_spurious = hagan_implied_vol(f, f, tau, spurious_root, beta, nu, rho);
  assert!(
    (vol_at_spurious - v_atm).abs() < 1e-10,
    "spurious root should *also* round-trip, which is exactly the hazard: got {vol_at_spurious:.15e}"
  );
}

/// Independent cross-check of [`hagan_implied_vol`] against a 40-decimal-digit
/// reimplementation of Hagan Eq. A.69a, rather than against values this crate
/// produced itself.
///
/// The rest of this file's reference cases are regression pins — they catch
/// drift but would not catch a formula that was wrong from the start. These
/// five were recomputed with `mpmath` at `mp.dps = 40`:
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
/// Every case agreed to within 1e-15 relative, i.e. floating-point noise, so
/// the tolerance below is set at 1e-14 rather than tuned to pass.
#[test]
fn hagan_matches_an_independent_high_precision_implementation() {
  // (K, F, tau, alpha, beta, rho, nu, mpmath value at 40 digits)
  let cases: &[(f64, f64, f64, f64, f64, f64, f64, f64)] = &[
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
    let got = hagan_implied_vol(k, f, tau, alpha, beta, nu, rho);
    let rel = (got - expected).abs() / expected.abs();
    assert!(
      rel < 1e-14,
      "K={k}: got {got:e}, mpmath {expected:e}, rel {rel:e}"
    );
  }
}

/// The ATM `beta = 1` case is simple enough to verify by hand, which pins the
/// `(a + b + c) * tau` correction independently of any reference value:
/// `a = 0`, `b = rho*nu*alpha/4 = -0.0075`,
/// `c = (2 - 3*rho^2)*nu^2/24 = 0.01802083...`, so
/// `sigma = alpha * (1 + (b + c) * tau) = 0.2 * 1.01052083... = 0.2021041666...`
#[test]
fn atm_beta_one_matches_hand_computation() {
  let (alpha, rho, nu, tau) = (0.2, -0.3, 0.5, 1.0);
  let b = 0.25 * rho * nu * alpha;
  let c = (2.0 - 3.0 * rho * rho) * nu * nu / 24.0;
  let expected = alpha * (1.0 + (b + c) * tau);
  let got = hagan_implied_vol(100.0, 100.0, tau, alpha, 1.0, nu, rho);
  assert!((got - expected).abs() < 1e-15, "got {got}, hand {expected}");
}

#[test]
#[should_panic(expected = "rho must lie strictly inside (-1, 1)")]
fn rejects_rho_at_one() {
  let _ = hagan_implied_vol(110.0, 100.0, 1.0, 0.2, 1.0, 0.5, 1.0);
}

#[test]
#[should_panic(expected = "strike k must be strictly positive")]
fn rejects_a_nonpositive_strike() {
  let _ = hagan_implied_vol(0.0, 100.0, 1.0, 0.2, 1.0, 0.5, -0.3);
}

#[test]
#[should_panic(expected = "alpha must be strictly positive")]
fn rejects_a_nonpositive_alpha() {
  let _ = hagan_implied_vol(110.0, 100.0, 1.0, 0.0, 1.0, 0.5, -0.3);
}

#[test]
#[should_panic(expected = "forward f must be strictly positive")]
fn alpha_from_atm_vol_rejects_a_nonpositive_forward() {
  let _ = alpha_from_atm_vol(0.2, 0.0, 1.0, 1.0, -0.3, 0.5);
}

#[test]
#[should_panic(expected = "v_atm must be strictly positive")]
fn alpha_from_atm_vol_rejects_a_nonpositive_v_atm() {
  let _ = alpha_from_atm_vol(0.0, 100.0, 1.0, 1.0, -0.3, 0.5);
}

#[test]
#[should_panic(expected = "rho must lie strictly inside (-1, 1)")]
fn alpha_from_atm_vol_rejects_rho_at_one() {
  let _ = alpha_from_atm_vol(0.2, 100.0, 1.0, 1.0, 1.0, 0.5);
}
