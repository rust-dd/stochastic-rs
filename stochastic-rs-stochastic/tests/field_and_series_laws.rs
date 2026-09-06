//! Closed-form laws two samplers are held to, from the papers they implement.
//!
//! The fractional Brownian field of Stein (2002) has, for any two points of
//! the unit disk, `E[(X(s) − X(t))²] = 2‖s − t‖^α` with `α = 2H` — Kroese &
//! Botev (2015, §4.4, eq. 16) — and Gaussian increments. The standard
//! classical tempered stable process of Kim, Rachev, Bianchi & Fabozzi has
//! variance `T` at horizon `T` by construction of its scale, and Rosiński's
//! (2007) series reproduces it once the arrival bound carries the total mass
//! of the Lévy measure over the horizon, `2 C T`.

use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::jump::cgmy::Cgmy;
use stochastic_rs_stochastic::jump::cts::Cts;
use stochastic_rs_stochastic::sheet::fbs::Fbs;
use stochastic_rs_stochastic::traits::ProcessExt;

/// The sample mean of `f` over the sheets.
fn mean_over(sheets: &[Array2<f64>], f: impl Fn(&Array2<f64>) -> f64) -> f64 {
  sheets.iter().map(f).sum::<f64>() / sheets.len() as f64
}

/// `E[(X(s) − X(t))²] = 2‖s − t‖^α` on the grid `(0, 1]²`, and the increments
/// are Gaussian: their fourth moment is three times the squared second.
#[test]
fn the_fractional_brownian_field_has_the_variogram_of_its_paper() {
  const SHEETS: usize = 20_000;
  let (h, m, n, r) = (0.7_f64, 9usize, 9usize, 1.0_f64);
  let alpha = 2.0 * h;
  let sheets = Fbs::<f64, _>::new(h, m, n, r, Deterministic::new(311)).sample_par(SHEETS);
  let coord = |i: usize, j: usize| (r * (i + 1) as f64 / m as f64, r * (j + 1) as f64 / n as f64);
  // Pairs whose distance stays inside the unit disk, where the embedding is
  // the field's own covariance.
  for (p, q) in [((0, 0), (6, 6)), ((2, 3), (6, 1)), ((0, 8), (4, 4)), ((1, 1), (1, 7))] {
    let (ps, qs) = (coord(p.0, p.1), coord(q.0, q.1));
    let dist = ((ps.0 - qs.0).powi(2) + (ps.1 - qs.1).powi(2)).sqrt();
    assert!(dist <= 1.0);
    let second = mean_over(&sheets, |s| (s[p] - s[q]).powi(2));
    let fourth = mean_over(&sheets, |s| (s[p] - s[q]).powi(4));
    let expected = 2.0 * dist.powf(alpha);
    assert!(
      (second / expected - 1.0).abs() < 0.05,
      "variogram at distance {dist}: sample {second}, paper {expected}"
    );
    assert!(
      (fourth / (3.0 * second * second) - 1.0).abs() < 0.15,
      "increment kurtosis at distance {dist}: {} (Gaussian is 3)",
      fourth / (second * second)
    );
    let mean = mean_over(&sheets, |s| s[p] - s[q]);
    assert!(mean.abs() < 5.0 * (second / SHEETS as f64).sqrt(), "increment mean {mean}");
  }
}

/// The standard CTS has variance one per unit of time by the construction of
/// its scale `C`; the truncated Rosiński series reproduces it at `T = 1` and
/// scales it with the horizon at `T = 2`.
#[test]
fn the_standard_tempered_stable_series_has_the_variance_of_its_scale() {
  const PATHS: usize = 20_000;
  let terminal_variance = |t: f64, seed: u64| {
    let paths =
      Cts::<f64, _>::new(2.0, 6.0, 0.5, 2, 4_000, Some(0.0), Some(t), Deterministic::new(seed))
        .sample_par(PATHS);
    let last: Vec<f64> = paths.iter().map(|p| p[p.len() - 1]).collect();
    let mean = last.iter().sum::<f64>() / PATHS as f64;
    last.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / PATHS as f64
  };
  let at_one = terminal_variance(1.0, 313);
  assert!(
    (at_one - 1.0).abs() < 0.06,
    "standard CTS variance over T = 1: {at_one}, expected 1"
  );
  let at_two = terminal_variance(2.0, 317);
  assert!(
    (at_two - 2.0).abs() < 0.12,
    "standard CTS variance over T = 2: {at_two}, expected 2"
  );
}

/// The CGMY variance `C Γ(2 − Y)(G^{Y−2} + M^{Y−2}) T` (Carr, Geman, Madan &
/// Yor 2002), which the sibling series with the same `2CT` arrival bound
/// reproduces.
#[test]
fn the_cgmy_series_has_the_variance_of_its_levy_measure() {
  const PATHS: usize = 20_000;
  let (c, g, m, y, t) = (0.5_f64, 4.0_f64, 7.0_f64, 0.6_f64, 1.5_f64);
  let paths = Cgmy::<f64, _>::new(c, g, m, y, 2, 4_000, Some(0.0), Some(t), Deterministic::new(331))
    .sample_par(PATHS);
  let last: Vec<f64> = paths.iter().map(|p| p[p.len() - 1]).collect();
  let mean = last.iter().sum::<f64>() / PATHS as f64;
  let variance = last.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / PATHS as f64;
  let expected = c * scilib::math::basic::gamma(2.0 - y) * (g.powf(y - 2.0) + m.powf(y - 2.0)) * t;
  assert!(
    (variance / expected - 1.0).abs() < 0.06,
    "CGMY variance over T = {t}: {variance}, closed form {expected}"
  );
}
