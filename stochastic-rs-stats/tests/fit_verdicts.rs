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
use stochastic_rs_stats::gmm_cir::fit_cir_gmm;
use stochastic_rs_stats::qmle::DiffusionKind;
use stochastic_rs_stats::qmle::fit_diffusion_qmle;

fn draws(seed: u64, n: usize) -> Vec<f64> {
  let mut z = vec![0.0; n];
  SimdNormal::<f64>::new(0.0, 1.0)
    .seeded(&Deterministic::new(seed))
    .fill_slice(&mut z);
  z
}

fn normals(seed: u64) -> Array1<f64> {
  Array1::from(draws(seed, 500))
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

/// An integrated path (`α + β = 1`): the unit-persistence law fits as well as the estimate.
#[test]
fn a_garch_fit_of_an_integrated_path_stops_at_the_bound_of_persistence() {
  let mut variance = 1.0_f64;
  let path = draws(6, 500).into_iter().map(|z| {
    let r = variance.sqrt() * z;
    variance = 0.02 + 0.1 * r * r + 0.9 * variance;
    r
  });
  let fit = fit_garch(path.collect::<Array1<f64>>().view(), GarchSpec::garch(1, 1)).unwrap();
  let persistence = "stopped at the bound of persistence";
  assert_eq!((fit.converged, fit.message.as_str()), (false, persistence));
}

/// A Cauchy sample has no variance: `η̂` runs to 2 with `σ̂` growing, a ridge no one axis shows.
#[test]
fn a_skew_t_fit_of_a_cauchy_sample_stops_at_the_bound_of_eta() {
  let cauchy = draws(1, 500)
    .chunks(2)
    .map(|z| z[0] / z[1])
    .collect::<Array1<f64>>();
  let fit = fit_skew_t(cauchy.view()).unwrap();
  assert_eq!(
    (fit.converged, fit.message.as_str()),
    (false, "stopped at the bound of eta")
  );
}

/// The census's slow CIR path (`θ = 0.02`, 1,000 points, seed 24, from splitmix64): four years
/// that show no reversion, where `θ̂` runs to zero and the data tell `θμ` alone.
fn slow_cir() -> Array1<f64> {
  let mut state = 9_024_u64;
  let mut uniform = || {
    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
  };
  let mut normal =
    || (-2.0 * uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * uniform()).cos();
  let (mut x, h) = (0.05_f64, 1.0 / 2520.0);
  let step = |_| {
    for _ in 0..10 {
      x = (x + 0.02 * (0.05 - x) * h + 0.1 * (x.max(0.0) * h).sqrt() * normal()).max(1e-8);
    }
    x
  };
  (0..1_000).map(step).collect()
}

#[test]
fn a_cir_fit_of_a_path_without_reversion_does_not_identify_mu() {
  let path = slow_cir();
  let qmle = fit_diffusion_qmle(path.view(), 1.0 / 252.0, DiffusionKind::Cir).unwrap();
  let gmm = fit_cir_gmm(path.view(), 1.0 / 252.0).unwrap();
  assert_eq!(
    (qmle.converged, qmle.message.as_str()),
    (false, "mu is not identified")
  );
  assert_eq!(
    (gmm.converged, gmm.message.as_str()),
    (false, "mu is not identified")
  );
}
