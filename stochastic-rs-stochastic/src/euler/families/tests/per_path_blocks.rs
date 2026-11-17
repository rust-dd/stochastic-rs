use super::*;

/// No family declares two of the history, series and table clauses: the
/// frame keeps one per-path array for the three.
#[test]
fn a_family_declares_at_most_one_per_path_block() {
  for family in Family::ALL {
    let blocks = usize::from(family.history_slot().is_some())
      + usize::from(family.has_series())
      + usize::from(family.has_table());
    assert!(blocks <= 1, "{family:?} declares {blocks} per-path blocks");
  }
}

/// Rosiński's term size: the arrival bound `(Γ rate)^{-1/α}` against the
/// tempered draw `e_scale E^{e_pow} U^{1/α} / λ_side`, on the side the
/// uniform picks.
#[test]
fn series_size_is_the_smaller_of_bound_and_draw_on_the_drawn_side() {
  let (rate, alpha, e_scale, e_pow, w_plus, lp, lm) = (0.25, 0.5, 0.5, 0.5, 0.4, 2.0, 6.0);
  let params = [0.01, rate, 1.0 / alpha, e_scale, e_pow, w_plus, lp, lm];
  let (gj, ej, uj): (f64, f64, f64) = (2.0, 1.5, 0.3);
  let cap = (gj * rate).powf(-1.0 / alpha);
  for (uv, side, sign) in [(0.2, lp, 1.0), (0.9, lm, -1.0)] {
    let draw = e_scale * ej.powf(e_pow) * uj.powf(1.0 / alpha) / side;
    let size = host_series(
      Family::TemperedStableSeries,
      &[0.0; 4],
      &params,
      0.01,
      gj,
      ej,
      uj,
      uv,
    )
    .expect("a series family");
    assert!(
      (size - sign * cap.min(draw)).abs() < 1e-12,
      "uv = {uv}: {size}"
    );
  }
  assert!(
    host_series(
      Family::GeometricBrownian,
      &[0.0; 4],
      &[0.05, 0.2],
      0.01,
      gj,
      ej,
      uj,
      0.2
    )
    .is_none()
  );
}

/// The live series of the stochastic-volatility CGMY reads the variance: the
/// arrival bound's rate is `rate0 / v`, so a larger variance admits larger
/// terms, and the family says so of itself.
#[test]
fn live_series_size_reads_the_variance_slot() {
  assert!(Family::StochasticVolatilityCgmy.series_live());
  assert!(!Family::TemperedStableSeries.series_live());
  let (rate0, alpha, lp, lm) = (0.25_f64, 0.5_f64, 2.0_f64, 6.0_f64);
  let params = [rate0, 1.0 / alpha, lp, lm, -0.1, 400.0, 0.98, 0.3];
  let (gj, ej, uj, uv): (f64, f64, f64, f64) = (2.0, 1.5, 0.3, 0.2);
  for v in [0.04_f64, 0.16_f64] {
    let cap = (gj * rate0 / v).powf(-1.0 / alpha);
    let draw = ej * uj.powf(1.0 / alpha) / lp;
    let size = host_series(
      Family::StochasticVolatilityCgmy,
      &[0.0, v, 0.0, 0.0],
      &params,
      0.01,
      gj,
      ej,
      uj,
      uv,
    )
    .expect("a series family");
    assert!((size - cap.min(draw)).abs() < 1e-12, "v = {v}: {size}");
  }
}

/// Kanter's positive-stable draw at the table's own scale.
#[test]
fn table_increment_is_the_positive_stable_draw_at_the_spacing() {
  let (alpha, c, tv) = (0.7_f64, 1.5_f64, 0.05_f64);
  let params = [
    alpha,
    c,
    1.0 / alpha,
    1.0 - alpha,
    (1.0 - alpha) / alpha,
    std::f64::consts::PI,
  ];
  let (uj, uv) = (0.37_f64, 0.61_f64);
  let u = uj * std::f64::consts::PI;
  let w = -uv.ln();
  let s1 = (alpha * u).sin() / u.sin().powf(1.0 / alpha);
  let s2 = (((1.0 - alpha) * u).sin() / w).powf((1.0 - alpha) / alpha);
  let inc = host_table(Family::InverseStableSubordinator, &params, 0.01, uj, uv, tv)
    .expect("a table family");
  assert!(
    (inc - (c * tv).powf(1.0 / alpha) * s1 * s2).abs() < 1e-12,
    "increment {inc}"
  );
  assert!(host_table(Family::GeometricBrownian, &[0.05, 0.2], 0.01, uj, uv, tv).is_none());
}
