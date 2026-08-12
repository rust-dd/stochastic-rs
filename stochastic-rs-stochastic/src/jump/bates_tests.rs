use stochastic_rs_distributions::scalar::ScalarNormal;

use super::*;

fn make_bates(
  mu: Option<f64>,
  b: Option<f64>,
  r: Option<f64>,
  r_f: Option<f64>,
) -> Bates1996<f64, ScalarNormal<f64>> {
  Bates1996::new(
    mu,
    b,
    r,
    r_f,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    ScalarNormal::new(0.0, 1.0),
    8,
    Some(1.0),
    Some(0.0),
    Some(1.0),
    Some(false),
    Unseeded,
  )
}

#[test]
fn effective_drift_prefers_r_minus_rf_when_present() {
  let p = make_bates(Some(0.9), Some(0.7), Some(0.4), Some(0.1));
  assert!((p.effective_drift() - 0.3).abs() < 1e-12);
}

#[test]
fn effective_drift_uses_b_if_rates_missing() {
  let p = make_bates(Some(0.9), Some(0.7), None, None);
  assert!((p.effective_drift() - 0.7).abs() < 1e-12);
}

#[test]
fn effective_drift_falls_back_to_mu() {
  let p = make_bates(Some(0.9), None, None, None);
  assert!((p.effective_drift() - 0.9).abs() < 1e-12);
}
