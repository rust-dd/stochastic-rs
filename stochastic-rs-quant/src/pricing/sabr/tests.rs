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
