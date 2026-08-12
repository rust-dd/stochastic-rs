use super::*;

#[test]
fn variance_stays_non_negative() {
  let p = BatesSvj::new(
    Some(0.05_f64),
    None,
    None,
    None,
    0.5,
    -0.1,
    0.2,
    0.04,
    1.5,
    0.3,
    -0.7,
    256,
    Some(100.0),
    Some(0.04),
    Some(1.0),
    Some(false),
    Unseeded,
  );
  let [_s, v] = p.sample();
  assert!(v.iter().all(|x| *x >= 0.0));
}

#[test]
fn price_stays_positive() {
  let p = BatesSvj::new(
    Some(0.05_f64),
    None,
    None,
    None,
    0.5,
    -0.1,
    0.2,
    0.04,
    1.5,
    0.3,
    -0.7,
    256,
    Some(100.0),
    Some(0.04),
    Some(1.0),
    Some(false),
    Unseeded,
  );
  let [s, _v] = p.sample();
  assert!(s.iter().all(|x| *x > 0.0));
}

#[test]
fn drift_prefers_r_minus_rf() {
  let p = BatesSvj::new(
    Some(0.9_f64),
    Some(0.7),
    Some(0.4),
    Some(0.1),
    0.5,
    0.0,
    0.1,
    0.04,
    1.5,
    0.3,
    -0.5,
    8,
    Some(1.0),
    Some(0.04),
    Some(1.0),
    None,
    Unseeded,
  );
  assert!((p.drift() - 0.3).abs() < 1e-12);
}

#[test]
fn drift_uses_b_if_rates_missing() {
  let p = BatesSvj::new(
    Some(0.9_f64),
    Some(0.7),
    None,
    None,
    0.5,
    0.0,
    0.1,
    0.04,
    1.5,
    0.3,
    -0.5,
    8,
    Some(1.0),
    Some(0.04),
    Some(1.0),
    None,
    Unseeded,
  );
  assert!((p.drift() - 0.7).abs() < 1e-12);
}

#[test]
fn drift_falls_back_to_mu() {
  let p = BatesSvj::new(
    Some(0.9_f64),
    None,
    None,
    None,
    0.5,
    0.0,
    0.1,
    0.04,
    1.5,
    0.3,
    -0.5,
    8,
    Some(1.0),
    Some(0.04),
    Some(1.0),
    None,
    Unseeded,
  );
  assert!((p.drift() - 0.9).abs() < 1e-12);
}
