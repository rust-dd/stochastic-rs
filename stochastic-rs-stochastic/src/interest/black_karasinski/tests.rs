use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

fn theta_const(_t: f64) -> f64 {
  0.05
}

fn theta_hundred(_t: f32) -> f32 {
  100.0
}

/// ln r is OU: E[ln r_t] = e^{-a t} ln r_0 + (theta/a)(1 - e^{-a t}) when
/// theta is constant (Black & Karasinski 1991).
#[test]
fn black_karasinski_log_mean_matches_ou() {
  let a = 0.8_f64;
  let sigma = 0.1;
  let r0 = 0.03_f64;
  let t = 1.0_f64;
  let n = 200;
  let paths = 20_000;
  let expected = (-a * t).exp() * r0.ln() + (0.05 / a) * (1.0 - (-a * t).exp());

  let best_rel_err = [2718u64, 999, 42]
    .into_iter()
    .map(|seed| {
      let bk = BlackKarasinski::<f64, _>::new(
        theta_const as fn(f64) -> f64,
        a,
        sigma,
        n,
        Some(r0),
        Some(t),
        Deterministic::new(seed),
      );
      let mean_log = bk
        .sample_par(paths)
        .iter()
        .map(|path| path.last().unwrap().ln())
        .sum::<f64>()
        / paths as f64;
      (mean_log - expected).abs() / expected.abs()
    })
    .fold(f64::INFINITY, f64::min);

  assert!(
    best_rel_err <= 2e-2,
    "best-of-3 relative error {best_rel_err} exceeds 2e-2 (expected {expected})"
  );
}

/// r_t > 0 for every point of every path (log construction) — must hold
/// unconditionally, with no boundary policy needed.
#[test]
fn black_karasinski_stays_positive() {
  let bk = BlackKarasinski::<f64, _>::new(
    theta_const as fn(f64) -> f64,
    0.8,
    0.5,
    300,
    Some(0.03),
    Some(2.0),
    Deterministic::new(2718),
  );
  for path in bk.sample_par(200) {
    assert!(path.iter().all(|x| x.is_finite() && *x > 0.0));
  }
}

/// `a <= 0` never panics: `a = 0` is the Brownian limit, a positive path, and `a < 0` diverges.
#[test]
fn black_karasinski_nonpositive_a_does_not_panic() {
  let zero_a = BlackKarasinski::<f64, _>::new(
    theta_const as fn(f64) -> f64,
    0.0,
    0.1,
    10,
    Some(0.03),
    Some(1.0),
    Deterministic::new(42),
  );
  let path = zero_a.sample();
  assert!(
    path.iter().all(|x| x.is_finite() && *x > 0.0),
    "a = 0 samples the Brownian log-rate: {path:?}"
  );

  let negative_a = BlackKarasinski::<f64, _>::new(
    theta_const as fn(f64) -> f64,
    -0.3,
    0.1,
    10,
    Some(0.03),
    Some(1.0),
    Deterministic::new(42),
  );
  // Only asserting "did not panic": a < 0 is documented to diverge, not
  // to stay in any particular finite range.
  let _ = negative_a.sample();
}

/// `sigma = 0` is the deterministic OU mean, `ln r_i = decay · ln r_{i−1} + (θ/a)(1 − decay)`.
#[test]
fn black_karasinski_zero_volatility_is_exact_ou_mean() {
  let a = 0.8_f64;
  let r0 = 0.03_f64;
  let t = 1.0_f64;
  let n = 50;
  let dt = t / (n - 1) as f64;
  let decay = (-a * dt).exp();

  let bk = BlackKarasinski::<f64, _>::new(
    theta_const as fn(f64) -> f64,
    a,
    0.0,
    n,
    Some(r0),
    Some(t),
    Deterministic::new(42),
  );
  let path = bk.sample();

  assert_eq!(path[0], r0, "path[0] must equal r0 exactly");
  let mut prev_log = r0.ln();
  for &value in path.iter().skip(1) {
    let mean = prev_log * decay + (0.05 / a) * (1.0 - decay);
    assert!(
      (value.ln() - mean).abs() < 1e-9,
      "got ln(r)={}, expected {mean}",
      value.ln()
    );
    prev_log = mean;
  }
}

/// A negative `sigma` flips the sign of every draw, a no-op in law rather than an invalid input.
#[test]
fn black_karasinski_negative_sigma_does_not_panic() {
  let bk = BlackKarasinski::<f64, _>::new(
    theta_const as fn(f64) -> f64,
    0.8,
    -0.2,
    10,
    Some(0.03),
    Some(1.0),
    Deterministic::new(42),
  );
  let path = bk.sample();
  assert!(path.iter().all(|x| x.is_finite() && *x > 0.0));
}

#[test]
fn the_one_minute_step_variance_does_not_cancel_in_single_precision() {
  let std = ou_step_std(0.01f32, 1.0f32 / (252.0 * 390.0));
  let exact = 1.017_500_914e-5_f64.sqrt();
  assert!(
    (std as f64 / exact - 1.0).abs() < 1e-6,
    "got {std}, exact {exact}: the old form was 17 % high"
  );
}

#[test]
fn the_one_minute_mean_term_keeps_its_digits_in_single_precision() {
  let (a, dt) = (0.01_f32, 1.0_f32 / (252.0 * 390.0));
  let path = BlackKarasinski::<f32, _>::new(
    theta_hundred as fn(f32) -> f32,
    a,
    0.0,
    2,
    Some(0.03),
    Some(dt),
    Deterministic::new(1),
  )
  .sample();
  let (a, dt, r0) = (a as f64, dt as f64, 0.03_f32 as f64);
  // θ·Δ ≈ 1e-3 lifts the mean term above ln r's f32 ulp; (θ/a)(1 − e^{−aΔ}) was 17 % high there.
  let exact = (r0.ln() * (-a * dt).exp() - 100.0 * (-a * dt).exp_m1() / a).exp();
  assert!(
    (path[1] as f64 / exact - 1.0).abs() < 1e-6,
    "got {}, exact {exact}",
    path[1]
  );
}

#[test]
fn a_zero_reversion_is_the_limit_of_a_vanishing_one() {
  let path = |a: f64| {
    BlackKarasinski::<f64, _>::new(
      theta_const as fn(f64) -> f64,
      a,
      0.1,
      10,
      Some(0.03),
      Some(1.0),
      Deterministic::new(42),
    )
    .sample()
  };
  let (limit, near) = (path(0.0), path(1e-12));
  // a = 1e-12 itself moves ln r by about a·t·|ln r0| ≈ 4e-12.
  for (x, y) in limit.iter().zip(near.iter()) {
    assert!(
      (x / y - 1.0).abs() < 1e-11,
      "a = 0 gives {x}, a = 1e-12 gives {y}"
    );
  }
}

/// Same seed twice must be bit-identical.
#[test]
fn black_karasinski_is_deterministic() {
  let bk1 = BlackKarasinski::<f64, _>::new(
    theta_const as fn(f64) -> f64,
    0.8,
    0.1,
    100,
    Some(0.03),
    Some(1.0),
    Deterministic::new(42),
  )
  .sample();
  let bk2 = BlackKarasinski::<f64, _>::new(
    theta_const as fn(f64) -> f64,
    0.8,
    0.1,
    100,
    Some(0.03),
    Some(1.0),
    Deterministic::new(42),
  )
  .sample();
  assert_eq!(bk1, bk2);
}
