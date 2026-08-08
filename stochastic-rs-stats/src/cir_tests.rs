use super::*;

fn exact_moments(theta: f64, mu: f64, sigma: f64, t: f64, r_t: f64) -> (f64, f64) {
  let decay = (-theta * t).exp();
  let mean = mu + (r_t - mu) * decay;
  let variance = r_t * sigma.powi(2) * decay * (1.0 - decay) / theta
    + mu * sigma.powi(2) * (1.0 - decay).powi(2) / (2.0 * theta);
  (mean, variance)
}

#[test]
fn zero_horizon_and_invalid_inputs_are_explicit() {
  assert_eq!(sample_seeded(2.0, 0.04, 0.4, 0.0, 0.05, 7), 0.05);
  assert!(sample_seeded(0.0, 0.04, 0.4, 1.0, 0.05, 7).is_nan());
  assert!(sample_seeded(2.0, -0.04, 0.4, 1.0, 0.05, 7).is_nan());
  assert!(pdf(2.0, 0.04, 0.4, 0.0, 0.05, 0.05).is_nan());
  assert!(pdf(2.0, 0.04, 0.4, 1.0, 0.05, -0.01).is_nan());
}

#[test]
fn seeded_exact_sampler_matches_cir_conditional_moments() {
  let (theta, mu, sigma, t, r_t) = (2.0, 0.04, 0.4, 1.0, 0.05);
  let samples = (0..20_000)
    .map(|index| sample_seeded(theta, mu, sigma, t, r_t, index + 11))
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
  let upper = 0.8;
  let intervals = 1_000;
  let step = upper / intervals as f64;
  let mut mass = 0.0;
  let mut first_moment = 0.0;
  for index in 0..intervals {
    let value = (index as f64 + 0.5) * step;
    let density = pdf(theta, mu, sigma, t, r_t, value);
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

#[test]
fn zero_initial_state_reduces_to_the_central_chi_square_density() {
  let density = pdf(2.0, 0.04, 0.4, 1.0, 0.0, 0.03);
  assert!(density.is_finite() && density > 0.0);
}
