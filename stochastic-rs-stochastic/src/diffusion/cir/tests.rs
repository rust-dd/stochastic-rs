use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

fn law(theta: f64, mu: f64, sigma: f64) -> Cir<f64> {
  Cir::new(
    theta,
    mu,
    sigma,
    2,
    None,
    None,
    VarianceBoundary::Absorption,
    Unseeded,
  )
}

fn exact_moments(theta: f64, mu: f64, sigma: f64, t: f64, r_t: f64) -> (f64, f64) {
  let decay = (-theta * t).exp();
  let mean = mu + (r_t - mu) * decay;
  let variance = r_t * sigma.powi(2) * decay * (1.0 - decay) / theta
    + mu * sigma.powi(2) * (1.0 - decay).powi(2) / (2.0 * theta);
  (mean, variance)
}

/// `2·theta·mu = 0.1 < sigma² = 1.0` violates Feller; the reflected path must still build and
/// stay finite.
#[test]
fn cir_accepts_sub_feller_under_reflection() {
  let cir = Cir::<f64, _>::new(
    0.5,
    0.1,
    1.0,
    256,
    Some(0.1),
    Some(1.0),
    VarianceBoundary::Reflect,
    Deterministic::new(7),
  );
  let path = cir.sample();
  assert_eq!(path.len(), 256);
  assert!(
    path.iter().all(|x| x.is_finite()),
    "sub-Feller Cir path must stay finite under VarianceBoundary::Reflect"
  );
}

/// The floored step accepts sub-Feller parameters too; only the diagnostic warning differs.
#[test]
fn cir_accepts_sub_feller_under_the_floor() {
  let cir = Cir::<f64, _>::new(
    0.5,
    0.1,
    1.0,
    256,
    Some(0.1),
    Some(1.0),
    VarianceBoundary::Absorption,
    Deterministic::new(7),
  );
  let path = cir.sample();
  assert!(path.iter().all(|x| x.is_finite() && *x >= 0.0));
}

/// A vanishing `σ²` sends `df + ncp` past the double range, by a term or (last case) only the sum:
/// the draw is the conditional mean, not `NaN` or `∞`.
#[test]
fn vanishing_variance_draws_the_conditional_mean() {
  let (theta, mu, t) = (0.5_f64, 0.04, 1.0);
  for (sigma, r_t) in [
    (1e-200, 0.0),
    (1e-200, 0.03),
    (1e-155, 0.0),
    (1e-155, 0.03),
    (2.83e-155, 0.026),
  ] {
    let mean = r_t * (-theta * t).exp() + mu * -(-theta * t).exp_m1();
    let draw = law(theta, mu, sigma).sample_transition(t, r_t, &Deterministic::new(7));
    assert_eq!(draw, mean, "sigma = {sigma:e}, r_t = {r_t}");
  }
}

#[test]
fn zero_horizon_and_invalid_inputs_are_explicit() {
  let (cir, seed) = (law(2.0, 0.04, 0.4), Deterministic::new(7));
  assert_eq!(cir.sample_transition(0.0, 0.05, &seed), 0.05);
  assert!(cir.sample_transition(1.0, -0.05, &seed).is_nan());
  assert!(
    law(0.0, 0.04, 0.4)
      .sample_transition(1.0, 0.05, &seed)
      .is_nan()
  );
  assert!(
    law(2.0, -0.04, 0.4)
      .sample_transition(1.0, 0.05, &seed)
      .is_nan()
  );
  assert!(cir.transition_pdf(0.0, 0.05, 0.05).is_nan());
  assert!(cir.transition_pdf(1.0, 0.05, -0.01).is_nan());
  assert!(cir.transition_pdf(1.0, -0.05, 0.05).is_nan());
  assert!(cir.stationary_pdf(0.05).is_finite());
  assert!(law(2.0, 0.0625, 0.5).stationary_pdf(-0.1).is_nan());
  assert!(law(2.0, 0.04, -0.4).stationary_pdf(0.05).is_nan());
  assert!(law(0.0, 0.04, 0.4).stationary_pdf(0.05).is_nan());
  assert!(cir.stationary_pdf(f64::NAN).is_nan());
  let mut written = law(2.0, 0.04, 0.4);
  written.sigma = f64::NAN;
  assert!(written.stationary_pdf(0.05).is_nan());
}

#[test]
fn a_seed_reproduces_its_draw() {
  let cir = law(2.0, 0.04, 0.4);
  let draw = |seed| cir.sample_transition(1.0, 0.05, &Deterministic::new(seed));
  assert_eq!(draw(11).to_bits(), draw(11).to_bits());
  assert_ne!(draw(11), draw(12));
}

#[test]
fn seeded_draws_match_the_conditional_moments() {
  let (theta, mu, sigma, t, r_t) = (2.0, 0.04, 0.4, 1.0, 0.05);
  let cir = law(theta, mu, sigma);
  let samples = (0..20_000)
    .map(|index| cir.sample_transition(t, r_t, &Deterministic::new(index + 11)))
    .collect::<Vec<_>>();
  let mean = samples.iter().sum::<f64>() / samples.len() as f64;
  let variance = samples
    .iter()
    .map(|value| (value - mean).powi(2))
    .sum::<f64>()
    / samples.len() as f64;
  let (expected_mean, expected_variance) = exact_moments(theta, mu, sigma, t, r_t);
  assert!(
    (mean - expected_mean).abs() < 7.5e-4,
    "{mean} vs {expected_mean}"
  );
  assert!(
    (variance - expected_variance).abs() / expected_variance < 0.04,
    "{variance} vs {expected_variance}"
  );
}

#[test]
fn transition_density_has_unit_mass_and_correct_first_moment() {
  let (theta, mu, sigma, t, r_t) = (2.0, 0.04, 0.3, 0.5, 0.05);
  let cir = law(theta, mu, sigma);
  let (upper, intervals) = (0.8, 1_000);
  let step = upper / intervals as f64;
  let (mut mass, mut first_moment) = (0.0, 0.0);
  for index in 0..intervals {
    let value = (index as f64 + 0.5) * step;
    let density = cir.transition_pdf(t, r_t, value);
    mass += density * step;
    first_moment += value * density * step;
  }
  let (expected_mean, _) = exact_moments(theta, mu, sigma, t, r_t);
  assert!((mass - 1.0).abs() < 3e-3, "integrated mass {mass}");
  assert!(
    (first_moment - expected_mean).abs() < 2e-4,
    "{first_moment} vs {expected_mean}"
  );
}

/// From `r_t = 5e-320` the density is its central limit, `8.79526042415e-93` by mpmath.
#[test]
fn a_subnormal_start_has_the_central_limit() {
  let (cir, t, x) = (law(2.0, 0.04, 0.3), 1.0 / 252.0, 0.04);
  let (subnormal, central) = (
    cir.transition_pdf(t, 5e-320, x),
    cir.transition_pdf(t, 0.0, x),
  );
  assert!(
    ((subnormal - central) / central).abs() < 1e-12,
    "{subnormal} vs {central}"
  );
  assert!(((central - 8.79526042415e-93) / 8.79526042415e-93).abs() < 1e-10);
}

/// The central limit either side of the range switch: `1e-300` keeps `v / u` and `u · v` normal,
/// `1e-311` overflows the ratio alone, and `1e-200` with a value as small underflows the product.
#[test]
fn the_density_is_continuous_where_its_noncentral_terms_leave_the_normal_range() {
  let (cir, t) = (law(2.0, 0.04, 0.3), 1.0 / 252.0);
  for (start, x) in [(1e-300, 0.04), (1e-311, 0.04), (1e-200, 1e-200)] {
    let (near, central) = (
      cir.transition_pdf(t, start, x),
      cir.transition_pdf(t, 0.0, x),
    );
    assert!(
      ((near - central) / central).abs() < 1e-12,
      "from {start:e} at {x:e}: {near:e} vs {central:e}"
    );
  }
}

#[test]
fn zero_initial_state_reduces_to_the_central_chi_square_density() {
  let density = law(2.0, 0.04, 0.4).transition_pdf(1.0, 0.0, 0.03);
  assert!(density.is_finite() && density > 0.0);
}

/// `mpmath.besseli` at 80 digits, including `q < 0`, `z` in the thousands and `q` near 2000.
#[test]
fn transition_density_matches_mpmath() {
  for ((theta, mu, sigma, t, r_t, x), want) in [
    ((2.0, 0.04, 0.3, 0.5, 0.05, 0.05), 11.078_458_423_438_674),
    ((0.5, 0.02, 0.3, 1.0, 0.03, 0.01), 14.035_771_834_181_954),
    (
      (0.5, 0.05, 0.1, 1.0 / 252.0, 0.05, 0.0505),
      264.242_211_771_611_84,
    ),
    ((1.0, 0.1, 0.01, 1.0, 8.6e-5, 0.0632), 282.151_624_977_580_8),
    ((1.0, 0.1, 0.01, 1.0, 0.05, 0.0816), 224.405_805_350_199_4),
  ] {
    let got = law(theta, mu, sigma).transition_pdf(t, r_t, x);
    assert!(
      ((got - want) / want).abs() < 1e-11,
      "transition_pdf({theta}, {mu}, {sigma}; {t}, {r_t}, {x}) = {got}, want {want}"
    );
    let log = law(theta, mu, sigma).transition_ln_pdf(t, r_t, x);
    assert!((log - want.ln()).abs() < 1e-11, "{log} vs {}", want.ln());
  }
}

/// mpmath at 50 digits. The first is at the speed a far search ends at, with a `σ` small enough
/// to underflow, the second a central tail: both densities are below the double range.
#[test]
fn the_log_density_is_finite_where_the_density_underflows() {
  for ((theta, mu, sigma, t, r_t, x), want) in [
    (
      (2.393_253_854_325_386e-4, 0.5, 1e-3, 1.0 / 12.0, 0.03, 0.033),
      -1_695.674_964_605_331_8,
    ),
    (
      (5.0, 1.0, 0.2, 1.0 / 252.0, 0.0, 1.0),
      -11_491.098_046_762_26,
    ),
  ] {
    let cir = law(theta, mu, sigma);
    assert_eq!(cir.transition_pdf(t, r_t, x), 0.0);
    let got = cir.transition_ln_pdf(t, r_t, x);
    assert!(((got - want) / want).abs() < 1e-13, "{got} vs {want}");
  }
}

/// mpmath at 50 digits, at shapes `2θμ/σ²` of 80, 250, 250 and 100: `c^shape` and `Γ(shape)`
/// leave the double range there while the density does not.
#[test]
fn a_zero_start_with_a_large_shape_matches_mpmath() {
  for ((theta, mu, sigma, t, x), density, log) in [
    (
      (2.0, 0.2, 0.1, 1.0 / 252.0, 0.0015),
      2_131.430_664_606_624_5,
      7.664_548_706_756_003,
    ),
    (
      (5.0, 1.0, 0.2, 1.0 / 252.0, 0.02),
      302.873_377_107_051_6,
      5.713_314_820_829_479,
    ),
    (
      (5.0, 1.0, 0.2, 1.0 / 12.0, 0.3),
      3.004_904_063_367_840_6,
      1.100_245_641_809_596,
    ),
    (
      (0.5, 1.0, 0.1, 1.0 / 12.0, 0.041),
      97.117_558_793_046_94,
      4.575_922_191_009_349,
    ),
  ] {
    let cir = law(theta, mu, sigma);
    let (got, got_log) = (
      cir.transition_pdf(t, 0.0, x),
      cir.transition_ln_pdf(t, 0.0, x),
    );
    assert!(
      ((got - density) / density).abs() < 1e-11,
      "{got} vs {density}"
    );
    assert!((got_log - log).abs() < 1e-11, "{got_log} vs {log}");
  }
}

#[test]
fn the_log_density_is_nan_and_infinite_where_the_density_is_nan_zero_and_infinite() {
  let cir = law(2.0, 0.04, 0.4);
  for (t, r_t, x) in [(0.0, 0.05, 0.05), (1.0, 0.05, -0.01), (1.0, -0.05, 0.05)] {
    assert!(cir.transition_pdf(t, r_t, x).is_nan() && cir.transition_ln_pdf(t, r_t, x).is_nan());
  }
  for (sigma, r_t, density) in [
    (0.3, 0.05, 0.0),
    (0.3, 0.0, 0.0),
    (0.5, 0.05, f64::INFINITY),
    (0.5, 0.0, f64::INFINITY),
  ] {
    let cir = law(2.0, 0.04, sigma);
    assert_eq!(cir.transition_pdf(1.0, r_t, 0.0), density);
    assert_eq!(cir.transition_ln_pdf(1.0, r_t, 0.0), density.ln());
  }
  let boundary = law(2.0, 0.0625, 0.5);
  for r_t in [0.05, 0.0] {
    let (density, log) = (
      boundary.transition_pdf(1.0, r_t, 0.0),
      boundary.transition_ln_pdf(1.0, r_t, 0.0),
    );
    assert!(density.is_finite() && density > 0.0 && log == density.ln());
  }
}

/// Twenty mean-reversion times out, the transition density has forgotten `r_t`: it is the
/// stationary Gamma.
#[test]
fn the_long_run_transition_is_the_stationary_law() {
  let cir = law(2.0, 0.04, 0.3);
  for x in [0.005, 0.02, 0.04, 0.08, 0.2] {
    let (long_run, stationary) = (cir.transition_pdf(20.0, 0.3, x), cir.stationary_pdf(x));
    assert!(
      ((long_run - stationary) / stationary).abs() < 1e-12,
      "x = {x}: {long_run} vs {stationary}"
    );
  }
}

/// mpmath at 50 digits, at a shape `2θμ/σ²` of 250: `β^shape` and `Γ(shape)` leave the double
/// range there while the density does not.
#[test]
fn the_stationary_density_at_a_large_shape_matches_mpmath() {
  let cir = law(5.0, 1.0, 0.2);
  for (r, want) in [
    (1.0, 6.305_729_046_132_535),
    (0.9, 1.834_349_997_158_214_5),
    (1.15, 0.424_086_533_548_943_5),
  ] {
    let got = cir.stationary_pdf(r);
    assert!(
      ((got - want) / want).abs() < 1e-11,
      "stationary_pdf({r}) = {got}, want {want}"
    );
  }
}

/// An `f32` process evaluates the `f64` law at its own values and rounds the result once.
#[test]
fn the_f32_law_is_the_f64_law_rounded_once() {
  let (mu, sigma, r, x) = (0.04_f32, 0.3_f32, 0.05_f32, 0.06_f32);
  let narrow = Cir::<f32>::new(
    2.0,
    mu,
    sigma,
    2,
    None,
    None,
    VarianceBoundary::Absorption,
    Unseeded,
  );
  let wide = law(2.0, mu as f64, sigma as f64);
  assert_eq!(
    narrow.transition_pdf(0.5, r, x),
    wide.transition_pdf(0.5, r as f64, x as f64) as f32
  );
  assert_eq!(
    narrow.transition_ln_pdf(0.5, r, x),
    wide.transition_ln_pdf(0.5, r as f64, x as f64) as f32
  );
  assert_eq!(
    narrow.stationary_pdf(x),
    wide.stationary_pdf(x as f64) as f32
  );
}
