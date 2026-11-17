use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

/// Volatility stays non-negative across a 2-bucket term structure.
#[test]
fn dynamic_sabr_volatility_non_negative() {
  let p = MultifactorSabr::<f64, _>::new(
    Some(1.0),
    Some(0.2),
    vec![0.5],        // single knot → 2 buckets
    vec![0.3, 0.7],   // β: 0.3 short end, 0.7 long end
    vec![-0.5, -0.2], // ρ term structure
    vec![0.4, 0.3],   // ν term structure
    512,
    Some(1.0),
    Unseeded,
  );
  let [_f, v] = p.sample();
  assert!(v.iter().all(|x| *x >= 0.0));
}

/// Empty-knot dynamic SABR with a single bucket must reproduce the
/// classic static SABR dynamics (forward stays non-negative, vol path
/// non-negative).
#[test]
fn dynamic_sabr_single_bucket_reduces_to_static() {
  let p = MultifactorSabr::<f64, _>::new(
    Some(1.0),
    Some(0.2),
    vec![], // no knots → 1 bucket
    vec![0.5],
    vec![-0.3],
    vec![0.4],
    256,
    Some(1.0),
    Unseeded,
  );
  let [f, v] = p.sample();
  assert!(f.iter().all(|x| *x >= 0.0));
  assert!(v.iter().all(|x| *x >= 0.0));
  assert_eq!(f.len(), 256);
}

/// The bucket lookup picks the right segment across the knots.
#[test]
fn dynamic_sabr_bucket_lookup() {
  let p = MultifactorSabr::<f64, _>::new(
    Some(1.0),
    Some(0.2),
    vec![0.25, 0.5, 0.75],
    vec![0.1, 0.2, 0.3, 0.4],
    vec![-0.1, -0.2, -0.3, -0.4],
    vec![0.1, 0.2, 0.3, 0.4],
    128,
    Some(1.0),
    Unseeded,
  );
  assert_eq!(p.bucket_at(0.0), 0);
  assert_eq!(p.bucket_at(0.1), 0);
  assert_eq!(p.bucket_at(0.25), 1);
  assert_eq!(p.bucket_at(0.3), 1);
  assert_eq!(p.bucket_at(0.5), 2);
  assert_eq!(p.bucket_at(0.75), 3);
  assert_eq!(p.bucket_at(0.99), 3);
  // Beyond the last knot stays in the last bucket.
  assert_eq!(p.bucket_at(5.0), 3);
}

/// Different β across buckets must produce a forward path whose
/// short-end and long-end realised local-vol scale differently. We
/// check determinism reproduces under a fixed seed.
#[test]
fn dynamic_sabr_seed_determinism() {
  let make = || {
    MultifactorSabr::<f64, _>::new(
      Some(1.0),
      Some(0.25),
      vec![0.5],
      vec![0.2, 0.8],
      vec![-0.4, -0.1],
      vec![0.5, 0.2],
      256,
      Some(1.0),
      Deterministic::new(7),
    )
  };
  let [f1, v1] = make().sample();
  let [f2, v2] = make().sample();
  assert_eq!(f1, f2, "forward path must be seed-deterministic");
  assert_eq!(v1, v2, "vol path must be seed-deterministic");
}

#[test]
#[should_panic(expected = "rho[1] must satisfy `rho[k].abs() < 1.0`, got rho[1] = 1.0")]
fn a_unit_correlation_is_named_with_its_bucket() {
  let _ = MultifactorSabr::<f64, _>::new(
    Some(1.0),
    Some(0.2),
    vec![0.5],
    vec![0.5, 0.5],
    vec![-0.3, 1.0],
    vec![0.4, 0.3],
    64,
    Some(1.0),
    Unseeded,
  );
}

/// Coefficient-vector length mismatch is rejected.
#[test]
#[should_panic(
  expected = "beta.len() must satisfy `beta.len() == knots.len() + 1`, got beta.len() = 1"
)]
fn dynamic_sabr_length_mismatch_panics() {
  let _ = MultifactorSabr::<f64, _>::new(
    Some(1.0),
    Some(0.2),
    vec![0.5], // 2 buckets
    vec![0.5], // β has only 1 entry → mismatch
    vec![-0.3, -0.2],
    vec![0.4, 0.3],
    64,
    Some(1.0),
    Unseeded,
  );
}
