//! What a fit says where its estimate is a limit of its family, on normal samples.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::skew_t::SimdSkewT;
use stochastic_rs_stats::distfit::fit_skew_t;
use stochastic_rs_stats::garch::GarchSpec;
use stochastic_rs_stats::garch::fit_garch;

fn normals(seed: u64) -> Array1<f64> {
  let mut z = vec![0.0; 500];
  SimdNormal::<f64>::new(0.0, 1.0)
    .seeded(&Deterministic::new(seed))
    .fill_slice(&mut z);
  Array1::from(z)
}

fn garch(seed: u64) -> (bool, String) {
  let fit = fit_garch(normals(seed).view(), GarchSpec::garch(1, 1)).unwrap();
  (fit.converged, fit.message)
}

/// White noise whose `α̂` is `4.9e-15`: the constant-variance law fits as well.
#[test]
fn a_garch_fit_with_no_arch_effect_stops_at_the_bound_of_alpha() {
  assert_eq!(garch(2), (false, "stopped at the bound of alpha[1]".into()));
}

/// White noise whose `β̂` is `2.5e-13`: the ARCH(1) law fits as well.
#[test]
fn a_garch_fit_with_no_memory_stops_at_the_bound_of_beta() {
  assert_eq!(garch(3), (false, "stopped at the bound of beta[1]".into()));
}

#[test]
fn a_garch_fit_inside_its_region_converges() {
  assert_eq!(garch(1), (true, "converged".into()));
}

#[test]
fn a_skew_t_fit_of_a_normal_sample_does_not_identify_eta() {
  let fit = fit_skew_t(normals(1).view()).unwrap();
  assert_eq!(
    (fit.converged, fit.message.as_str()),
    (false, "eta is not identified")
  );
}

/// Sixty values near `1.5e308`, whose sum overflows, carry no tail to tell `η` by.
#[test]
fn a_skew_t_fit_of_sixty_values_near_the_largest_double_does_not_identify_eta() {
  let x = Array1::from_iter((0..60).map(|j| 1.5e308 * (0.9 + 0.1 * ((j * 37) % 61) as f64 / 61.0)));
  let fit = fit_skew_t(x.view()).unwrap();
  assert_eq!(
    (fit.converged, fit.message.as_str()),
    (false, "eta is not identified")
  );
}

/// Hansen's law at `η = 6`, `λ = −0.3`, shifted and scaled: an interior fit, four errors wide.
#[test]
fn a_skew_t_fit_of_its_own_law_converges_within_four_standard_errors() {
  let mut z = vec![0.0; 2_000];
  SimdSkewT::<f64>::new(6.0, -0.3)
    .seeded(&Deterministic::new(31))
    .fill_slice(&mut z);
  let fit = fit_skew_t(Array1::from_iter(z.iter().map(|z| 0.2 + 0.5 * z)).view()).unwrap();
  assert_eq!((fit.converged, fit.message.as_str()), (true, "converged"));
  let pairs = [
    (fit.mu, 0.2),
    (fit.sigma, 0.5),
    (fit.eta, 6.0),
    (fit.lambda, -0.3),
  ];
  for (i, (got, truth)) in pairs.into_iter().enumerate() {
    let off = (got - truth).abs() / fit.std_errors[i];
    assert!(off < 4.0, "parameter {i} is {off} errors off");
  }
}
