use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

/// E[X_t] = x0 + δ·t for BESQ(δ) (Revuz & Yor, *Continuous Martingales and
/// Brownian Motion*, Ch. XI §1) — Monte Carlo check against the closed form.
#[test]
fn besq_mean_matches_closed_form() {
  let delta = 3.0;
  let x0 = 1.0;
  let t = 1.0;
  let n = 200;
  let paths = 20_000;
  let expected = x0 + delta * t;

  let best_rel_err = [2718u64, 999, 42]
    .into_iter()
    .map(|seed| {
      let besq =
        SquaredBessel::<f64, _>::new(delta, n, Some(x0), Some(t), None, Deterministic::new(seed));
      let mean = besq
        .sample_par(paths)
        .iter()
        .map(|path| *path.last().unwrap())
        .sum::<f64>()
        / paths as f64;
      (mean - expected).abs() / expected
    })
    .fold(f64::INFINITY, f64::min);

  assert!(
    best_rel_err <= 2e-2,
    "best-of-3 relative error {best_rel_err} exceeds 2e-2 (expected {expected})"
  );
}

/// δ=1 is sub-boundary (δ < 2): the discretized path must stay
/// non-negative and finite under both `use_sym` branches.
#[test]
fn besq_stays_nonnegative() {
  for use_sym in [None, Some(true)] {
    let besq = SquaredBessel::<f64, _>::new(
      1.0,
      500,
      Some(0.5),
      Some(1.0),
      use_sym,
      Deterministic::new(2718),
    );
    let path = besq.sample();
    assert!(
      path.iter().all(|x| x.is_finite() && *x >= 0.0),
      "use_sym = {use_sym:?}"
    );
  }
}

/// BES(δ) squared has the same law as BESQ(δ): compare terminal means.
#[test]
fn bessel_squared_matches_besq_mean() {
  let delta = 3.0;
  let x0 = 1.0;
  let t = 1.0;
  let n = 200;
  let paths = 20_000;

  let best_rel_err = [2718u64, 999, 42]
    .into_iter()
    .map(|seed| {
      let besq =
        SquaredBessel::<f64, _>::new(delta, n, Some(x0), Some(t), None, Deterministic::new(seed));
      let besq_mean = besq
        .sample_par(paths)
        .iter()
        .map(|path| *path.last().unwrap())
        .sum::<f64>()
        / paths as f64;

      let bes = Bessel::<f64, _>::new(
        delta,
        n,
        Some(x0.sqrt()),
        Some(t),
        None,
        Deterministic::new(seed),
      );
      let bes_squared_mean = bes
        .sample_par(paths)
        .iter()
        .map(|path| path.last().unwrap().powi(2))
        .sum::<f64>()
        / paths as f64;

      (bes_squared_mean - besq_mean).abs() / besq_mean
    })
    .fold(f64::INFINITY, f64::min);

  assert!(
    best_rel_err <= 5e-2,
    "best-of-3 relative error {best_rel_err} exceeds 5e-2"
  );
}

/// Independent golden, not derived from the BESQ recursion `Bessel` is
/// built on: for BES(δ) started at 0, `X_t = sqrt(t) * chi_delta`
/// (Revuz & Yor, *Continuous Martingales and Brownian Motion*, Ch. XI
/// §1), so `E[X_t] = sqrt(2t) * Gamma((delta+1)/2) / Gamma(delta/2)`; at
/// `delta = 3`, `t = 1` this simplifies to `2 * sqrt(2/pi) ≈ 1.595769`.
#[test]
fn bessel_mean_matches_chi_distribution_identity() {
  let delta = 3.0;
  let t = 1.0;
  let n = 200;
  let paths = 400_000;
  let expected = 2.0 * (2.0 / std::f64::consts::PI).sqrt();

  let best_rel_err = [2718u64, 999, 42]
    .into_iter()
    .map(|seed| {
      let bes = Bessel::<f64, _>::new(delta, n, Some(0.0), Some(t), None, Deterministic::new(seed));
      let mean = bes
        .sample_par(paths)
        .iter()
        .map(|path| *path.last().unwrap())
        .sum::<f64>()
        / paths as f64;
      (mean - expected).abs() / expected
    })
    .fold(f64::INFINITY, f64::min);

  assert!(
    best_rel_err <= 5e-3,
    "best-of-3 relative error {best_rel_err} exceeds 5e-3 (expected {expected})"
  );
}

/// Same seed twice must be bit-identical, for both process types.
#[test]
fn besq_is_deterministic() {
  let besq1 =
    SquaredBessel::<f64, _>::new(3.0, 50, Some(1.0), Some(1.0), None, Deterministic::new(42))
      .sample();
  let besq2 =
    SquaredBessel::<f64, _>::new(3.0, 50, Some(1.0), Some(1.0), None, Deterministic::new(42))
      .sample();
  assert_eq!(besq1, besq2);
}

/// Same seed twice must be bit-identical, for both process types.
#[test]
fn bessel_is_deterministic() {
  let bes1 =
    Bessel::<f64, _>::new(3.0, 50, Some(1.0), Some(1.0), None, Deterministic::new(42)).sample();
  let bes2 =
    Bessel::<f64, _>::new(3.0, 50, Some(1.0), Some(1.0), None, Deterministic::new(42)).sample();
  assert_eq!(bes1, bes2);
}
