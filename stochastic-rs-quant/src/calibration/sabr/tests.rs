use super::*;
use crate::traits::Calibrator;

#[test]
fn sabr_params_array_round_trip_preserves_beta() {
  for &beta in &[0.0, 0.25, 0.5, 0.75, 1.0] {
    let p = SabrParams {
      alpha: 0.18,
      beta,
      nu: 0.62,
      rho: -0.31,
    };
    let v = Array1::from(p);
    let p2 = SabrParams::from(v);
    assert!((p.alpha - p2.alpha).abs() < 1e-15);
    assert!(
      (p.beta - p2.beta).abs() < 1e-15,
      "β must round-trip: input {beta}, got {}",
      p2.beta
    );
    assert!((p.nu - p2.nu).abs() < 1e-15);
    assert!((p.rho - p2.rho).abs() < 1e-15);
  }
}

#[test]
fn test_sabr_calibrate_price_based() {
  let s = vec![100.0; 8];
  let k = vec![80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0];
  let r = 0.02;
  let q = 0.01;
  let tau = 0.5;

  let true_p = SabrParams {
    alpha: 0.2,
    beta: 1.0,
    nu: 0.6,
    rho: -0.4,
  };

  let mut c_market = Vec::new();
  for &kk in &k {
    let pr = SabrPricer::new(true_p.alpha, true_p.beta, true_p.nu, true_p.rho);
    let (call, _) = pr.call_put(100.0, kk, r, q, tau);
    c_market.push(call);
  }

  let calibrator = SabrCalibrator::new(
    Some(SabrParams {
      alpha: 0.15,
      beta: 1.0,
      nu: 0.8,
      rho: 0.0,
    }),
    c_market.clone().into(),
    s.clone().into(),
    k.clone().into(),
    r,
    Some(q),
    tau,
    OptionType::Call,
    true,
  );

  calibrator.calibrate(None).unwrap();
}

fn calibrator(s: f64, k: f64) -> SabrCalibrator {
  SabrCalibrator::new(
    None,
    vec![1.0].into(),
    vec![s].into(),
    vec![k].into(),
    0.02,
    Some(0.01),
    0.5,
    OptionType::Call,
    false,
  )
}

/// `calibrate` must return `Err`, not panic, for a non-positive or `NaN`
/// spot/strike — it used to panic inside the Levenberg-Marquardt cost
/// callback because `s`/`k` fed `hagan_implied_vol` unchecked.
#[test]
fn sabr_calibrate_rejects_nonpositive_or_nan_spot_and_strike() {
  for bad_s in [0.0, -50.0, f64::NAN] {
    let err = calibrator(bad_s, 100.0).calibrate(None).unwrap_err();
    assert!(
      err.to_string().contains("s[0]"),
      "s = {bad_s}: unexpected message {err}"
    );
  }
  for bad_k in [0.0, -10.0, f64::NAN] {
    let err = calibrator(100.0, bad_k).calibrate(None).unwrap_err();
    assert!(
      err.to_string().contains("k[0]"),
      "k = {bad_k}: unexpected message {err}"
    );
  }
}
