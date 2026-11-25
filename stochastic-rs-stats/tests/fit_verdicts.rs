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

/// EGARCH(1,q) at `α = 0.15`, `γ = −0.08`, the log-variance about zero.
fn egarch_path(seed: u64, n: usize, beta: [f64; 2]) -> Array1<f64> {
  let (mut last, mut before) = (0.0_f64, 0.0_f64);
  let step = |z: f64| {
    let r = (0.5 * last).exp() * z;
    let news = 0.15 * (z.abs() - (2.0 / std::f64::consts::PI).sqrt()) - 0.08 * z;
    (before, last) = (last, news + beta[0] * last + beta[1] * before);
    r
  };
  draws(seed, n).into_iter().map(step).collect()
}

fn egarch(sample: Array1<f64>, q: usize) -> stochastic_rs_stats::garch::GarchFit {
  fit_garch(sample.view(), GarchSpec::egarch(1, q)).unwrap()
}

/// White noise has no news curve: one of its slopes runs to zero.
#[test]
fn an_egarch_fit_of_white_noise_stops_at_the_bound_of_a_slope() {
  let fit = egarch(Array1::from(draws(3, 250)), 1);
  let slope = "stopped at the bound of alpha[1] + gamma[1]";
  assert_eq!((fit.converged, fit.message.as_str()), (false, slope));
}

/// White noise whose likelihood is highest at `α < |γ|`: the fit ends on that slope's bound.
#[test]
fn an_egarch_fit_once_outside_the_invertible_region_stops_at_the_bound_of_a_slope() {
  let fit = egarch(Array1::from(draws(4, 250)), 1);
  assert_eq!(fit.message, "stopped at the bound of alpha[1] - gamma[1]");
  assert!(fit.params[2] >= fit.params[3].abs() && (0.0..1.0).contains(&fit.params[4]));
}

#[test]
fn an_egarch_fit_of_its_own_law_converges_inside_the_region() {
  let fit = egarch(egarch_path(2, 2_000, [0.95, 0.0]), 1);
  assert_eq!((fit.converged, fit.message.as_str()), (true, "converged"));
  let truth = [0.15, -0.08, 0.95];
  for (i, truth) in truth.into_iter().enumerate() {
    let off = (fit.params[2 + i] - truth).abs() / fit.std_errors[2 + i];
    assert!(off < 4.0, "parameter {i} is {off} errors off");
  }
}

/// Two lags of the log-variance, `β = (0.6, 0.3)`: each `β̂ⱼ ≥ 0`, their sum under one.
#[test]
fn an_egarch_fit_with_two_variance_lags_converges_inside_the_region() {
  let fit = egarch(egarch_path(1, 2_000, [0.6, 0.3]), 2);
  assert_eq!((fit.converged, fit.message.as_str()), (true, "converged"));
  let (alpha, gamma, betas) = (fit.params[2], fit.params[3], [fit.params[4], fit.params[5]]);
  assert!(alpha >= gamma.abs() && betas.iter().all(|b| *b >= 0.0) && betas[0] + betas[1] < 1.0);
}

/// The census's GJR path (250 points, seed 31, from splitmix64): the pin toward `β = 0` passes
/// an interior maximum at `β ≈ 0.08`, `0.05` above the fit at the bound.
#[test]
fn a_search_restarts_from_a_better_fit_on_a_limits_path() {
  let mut state = 531_u64;
  let mut uniform = || {
    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
  };
  let mut normal =
    || (-2.0 * uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * uniform()).cos();
  let mut h = 1e-4_f64;
  let step = |_| {
    let r = h.sqrt() * normal();
    h = 1e-4 * 0.04 + (0.03 + if r < 0.0 { 0.10 } else { 0.0 }) * r * r + 0.88 * h;
    r
  };
  let path = (0..250).map(step).collect::<Array1<f64>>();
  let fit = fit_garch(path.view(), GarchSpec::gjr(1, 1)).unwrap();
  assert!(
    fit.log_likelihood > 797.69,
    "{} [{}]",
    fit.log_likelihood,
    fit.message
  );
}

/// The census's normals, from splitmix64.
fn mixed(seed: u64, n: usize) -> Vec<f64> {
  let mut state = seed;
  let mut uniform = || {
    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
  };
  let normal = |_| (-2.0 * uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * uniform()).cos();
  (0..n).map(normal).collect()
}

/// The top Lyapunov exponent of EGARCH(1,2)'s inverted recursion along the fitted path: the
/// log-variance's Jacobian has the first row `(β₁ − ½(γz + α|z|), β₂)` over `(1, 0)`.
fn exponent(fit: &stochastic_rs_stats::garch::GarchFit) -> f64 {
  let (alpha, gamma, beta) = (fit.params[2], fit.params[3], [fit.params[4], fit.params[5]]);
  let (mut v, mut sum) = ([1.0_f64, 0.0], 0.0);
  for z in &fit.standardized_residuals {
    let slope = beta[0] - 0.5 * (gamma * z + alpha * z.abs());
    let next = [slope * v[0] + beta[1] * v[1], v[0]];
    let norm = next[0].hypot(next[1]);
    sum += norm.ln();
    v = [next[0] / norm, next[1] / norm];
  }
  sum / fit.standardized_residuals.len() as f64
}

/// Two census samples of EGARCH(1,2) at `β = (0.6, 0.3)` whose likelihood is higher at `β₂` near
/// one over `β₁` near zero, where the inverted recursion expands.
#[test]
fn an_egarch_fit_with_two_variance_lags_keeps_them_ordered_and_contracting() {
  for seed in [16_u64, 27] {
    let level = 1e-4_f64.ln();
    let (mut last, mut before) = (level, level);
    let step = |z: f64| {
      let r = (0.5 * last).exp() * z;
      let news = 0.15 * (z.abs() - (2.0 / std::f64::consts::PI).sqrt()) - 0.08 * z;
      (before, last) = (last, level * 0.1 + news + 0.6 * last + 0.3 * before);
      r
    };
    let path = mixed(seed + 700, 250)
      .into_iter()
      .map(step)
      .collect::<Array1<f64>>();
    let fit = egarch(path, 2);
    let (first, second) = (fit.params[4], fit.params[5]);
    assert!(
      first >= second && second >= 0.0,
      "seed {seed}: {first}, {second}"
    );
    assert!(exponent(&fit) <= 0.0, "seed {seed}: {}", exponent(&fit));
  }
}

/// Two white-noise census samples with a second maximum `0.301` and `0.130` above the one a
/// single descent from the best start reaches.
#[test]
fn a_garch_fit_descends_from_every_persistence_level() {
  for (seed, single) in [(18_u64, 6_399.359_270_958), (50, 6_394.212_803_119)] {
    let noise = mixed(seed, 2_000)
      .into_iter()
      .map(|z| 1e-2 * z)
      .collect::<Array1<f64>>();
    let fit = fit_garch(noise.view(), GarchSpec::garch(1, 1)).unwrap();
    assert!(
      fit.log_likelihood > single + 0.12,
      "seed {seed}: {}",
      fit.log_likelihood
    );
  }
}
