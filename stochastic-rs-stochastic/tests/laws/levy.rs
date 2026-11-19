//! The pure-jump processes against their sources' characteristic and Laplace transforms, which
//! are bounded and state the whole law. `t = n − 1` makes a grid step one unit of time.

use ndarray::Array1;
use num_complex::Complex64;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::jump::bilateral_gamma::BilateralGamma;
use stochastic_rs_stochastic::jump::nig::Nig;
use stochastic_rs_stochastic::jump::rdts::Rdts;
use stochastic_rs_stochastic::jump::vg::Vg;
use stochastic_rs_stochastic::process::subordinator::alpha_stable::AlphaStableSubordinator;
use stochastic_rs_stochastic::process::subordinator::gamma_subordinator::GammaSubordinator;
use stochastic_rs_stochastic::process::subordinator::ig_subordinator::IgSubordinator;
use stochastic_rs_stochastic::process::subordinator::poisson_subordinator::PoissonSubordinator;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::characteristic_function_holds;
use super::common::holds_on_one_of;
use super::common::increments;
use super::common::over_paths;
use super::common::spread_over_paths;
use super::common::terminals;
use super::common::transform_holds;

/// Grid points per path. One more than the unit steps each path contributes.
const STEPS: usize = 4_097;

/// Paths per case: `PATHS · (STEPS − 1)` ≈ 260 000 draws of the one-step
/// law, which puts the standard error of a transform near 0.002.
const PATHS: usize = 64;

/// The horizon that makes one grid step one unit of time.
fn unit_horizon() -> f64 {
  (STEPS - 1) as f64
}

/// The frequencies every characteristic function is checked at: small enough that `φ` has not
/// decayed into its standard error, spread enough that a wrong exponent cannot agree at all.
const FREQUENCIES: [f64; 4] = [0.3, 0.7, 1.5, 2.5];

/// Variance gamma: `φ(u) = (1 − iuθν + σ²νu²/2)^{−t/ν}`, Brownian motion
/// with drift `θ` under a gamma clock (Madan, Carr & Chang 1998, eq. 7).
#[test]
fn variance_gamma_matches_the_madan_carr_chang_transform() {
  let (theta, sigma, nu) = (-0.2, 0.4, 0.35);
  let paths = Vg::<f64, _>::new(
    theta,
    sigma,
    nu,
    STEPS,
    None,
    Some(unit_horizon()),
    Deterministic::new(211),
  )
  .sample_par(PATHS);
  let sample = increments(&paths);
  for u in FREQUENCIES {
    let base = Complex64::new(1.0 + 0.5 * sigma * sigma * nu * u * u, -u * theta * nu);
    characteristic_function_holds(&sample, u, base.powf(-1.0 / nu), "variance gamma");
  }
}

/// Normal inverse Gaussian, `φ(u) = exp((t/κ)(1 − √(1 − 2iuθκ + σ²u²κ)))`: Brownian motion with
/// drift `θ` under an inverse-Gaussian clock, in the constructor's `(θ, σ, κ)` parameters.
/// Reference: Barndorff-Nielsen (1997), §3
#[test]
fn normal_inverse_gaussian_matches_the_barndorff_nielsen_transform() {
  let (theta, sigma, kappa) = (0.15, 0.5, 0.4);
  let paths = Nig::<f64, _>::new(
    theta,
    sigma,
    kappa,
    STEPS,
    None,
    Some(unit_horizon()),
    Deterministic::new(223),
  )
  .sample_par(PATHS);
  let sample = increments(&paths);
  for u in FREQUENCIES {
    let inner = Complex64::new(
      1.0 + sigma * sigma * u * u * kappa,
      -2.0 * u * theta * kappa,
    );
    let phi = ((Complex64::new(1.0, 0.0) - inner.sqrt()) / kappa).exp();
    characteristic_function_holds(&sample, u, phi, "normal inverse Gaussian");
  }
}

/// Bilateral gamma, the difference of two gamma subordinators:
/// `φ(u) = (1 − iu/λ₊)^{−α₊t}(1 + iu/λ₋)^{−α₋t}`. Unequal sides, so a sign error cannot pass.
/// Reference: Küchler & Tappe (2008), §2
#[test]
fn bilateral_gamma_matches_the_kuchler_tappe_transform() {
  let (alpha_p, lambda_p, alpha_m, lambda_m) = (1.2, 3.0, 0.8, 5.0);
  let paths = BilateralGamma::<f64, _>::new(
    alpha_p,
    lambda_p,
    alpha_m,
    lambda_m,
    STEPS,
    None,
    Some(unit_horizon()),
    Deterministic::new(227),
  )
  .sample_par(PATHS);
  let sample = increments(&paths);
  for u in FREQUENCIES {
    let plus = Complex64::new(1.0, -u / lambda_p).powf(-alpha_p);
    let minus = Complex64::new(1.0, u / lambda_m).powf(-alpha_m);
    characteristic_function_holds(&sample, u, plus * minus, "bilateral gamma");
  }
}

/// The standard RDTS of Kim, Rachev, Bianchi & Fabozzi (2010): mean 0 by the drift, variance 1
/// by the scale `C`. At `α < 1` the terms past 256 shift both by under a tenth of a standard error.
#[test]
fn rapidly_decreasing_tempered_stable_has_the_standard_mean_and_variance() {
  for (alpha, lambda_plus, lambda_minus) in [(0.5, 2.0, 3.0), (0.3, 1.5, 4.0)] {
    let x1 = [2718, 999, 42].map(|seed| {
      let rdts = Rdts::<f64, _>::new(
        lambda_plus,
        lambda_minus,
        alpha,
        2,
        256,
        None,
        None,
        Deterministic::new(seed),
      );
      terminals(&rdts.sample_par(4_096))
    });
    let at = format!("RDTS at α = {alpha}, λ = ({lambda_plus}, {lambda_minus})");
    let means = x1.each_ref().map(|x| over_paths(x));
    holds_on_one_of(&means, 0.0, &format!("{at}: mean"));
    let variances = x1.each_ref().map(|x| spread_over_paths(x));
    holds_on_one_of(&variances, 1.0, &format!("{at}: variance"));
  }
}

/// `E[e^{−uX_t}] = exp(δt(1 − √(1 + 2u)))`, the Laplace transform of `IG(δt, (δt)²)`, the law at
/// `gamma = 1` (Chhikara & Folks 1989, §2.2).
#[test]
fn inverse_gaussian_subordinator_matches_its_laplace_transform() {
  let delta = 1.4;
  let paths = IgSubordinator::<f64, _>::new(
    delta,
    1.0,
    STEPS,
    None,
    Some(unit_horizon()),
    Deterministic::new(229),
  )
  .sample_par(PATHS);
  let sample = increments(&paths);
  assert!(
    sample.iter().all(|&x| x > 0.0),
    "a subordinator's increment was not positive"
  );
  for u in [0.25_f64, 0.75, 2.0, 5.0] {
    transform_holds(
      &sample,
      |x| (-u * x).exp(),
      (delta * (1.0 - (1.0 + 2.0 * u).sqrt())).exp(),
      &format!("inverse Gaussian subordinator at u = {u}"),
    );
  }
}

/// The gamma subordinator: `E[e^{−uX_t}] = (1 + u/β)^{−νt}` for
/// `X_t ∼ Γ(νt, 1/β)` (Cont & Tankov 2004, §4.4.1).
#[test]
fn gamma_subordinator_matches_its_laplace_transform() {
  let (nu, rate) = (1.6, 2.5);
  let paths = GammaSubordinator::<f64, _>::new(
    nu,
    rate,
    STEPS,
    None,
    Some(unit_horizon()),
    Deterministic::new(233),
  )
  .sample_par(PATHS);
  let sample = increments(&paths);
  for u in [0.25, 0.75, 2.0, 5.0] {
    transform_holds(
      &sample,
      |x| (-u * x).exp(),
      (1.0 + u / rate).powf(-nu),
      &format!("gamma subordinator at u = {u}"),
    );
  }
}

/// The α-stable subordinator, `E[e^{−uX_t}] = exp(−ctu^α)`. At `α = 0.7` it has no finite mean,
/// so only a transform has a standard error; this checks the sampler's positive-stable draw.
/// Reference: Bertoin (1996), §III.1
/// Reference: Chambers, Mallows & Stuck (1976)
#[test]
fn alpha_stable_subordinator_matches_its_laplace_exponent() {
  let (alpha, c) = (0.7, 0.8);
  let paths = AlphaStableSubordinator::<f64, _>::new(
    alpha,
    c,
    STEPS,
    None,
    Some(unit_horizon()),
    Deterministic::new(239),
  )
  .sample_par(PATHS);
  let sample = increments(&paths);
  assert!(
    sample.iter().all(|&x| x > 0.0),
    "a subordinator's increment was not positive"
  );
  for u in [0.2_f64, 0.6, 1.5, 4.0] {
    transform_holds(
      &sample,
      |x| (-u * x).exp(),
      (-c * u.powf(alpha)).exp(),
      &format!("α-stable subordinator at u = {u}"),
    );
  }
}

/// The Poisson subordinator: a unit step counts `Poisson(λ)` events, so mean and variance are
/// `λ`, an empty step has probability `e^{−λ}`, and `φ(u) = exp(λ(e^{iu} − 1))`.
/// Reference: Cont & Tankov (2004), §3.1
#[test]
fn poisson_subordinator_counts_match_the_poisson_law() {
  let lambda = 2.3;
  let paths = PoissonSubordinator::<f64, _>::new(
    lambda,
    STEPS,
    None,
    Some(unit_horizon()),
    Deterministic::new(241),
  )
  .sample_par(PATHS);
  let sample = increments(&paths);
  assert!(
    sample.iter().all(|&x| x >= 0.0 && x.fract() == 0.0),
    "a count was not a non-negative whole number"
  );
  transform_holds(&sample, |x| x, lambda, "Poisson subordinator mean");
  transform_holds(
    &sample,
    |x| (x - lambda).powi(2),
    lambda,
    "Poisson subordinator variance",
  );
  transform_holds(
    &sample,
    |x| f64::from(x == 0.0),
    (-lambda).exp(),
    "Poisson subordinator empty-step probability",
  );
  for u in FREQUENCIES {
    let phi = (Complex64::new(0.0, u).exp() - 1.0) * lambda;
    characteristic_function_holds(&sample, u, phi.exp(), "Poisson subordinator");
  }
}

/// Every path of every process here starts where it was told to and stays in
/// the reals — the structural check the transforms above assume.
#[test]
fn every_levy_path_stays_in_the_reals() {
  let checks: Vec<(&str, Array1<f64>)> = vec![
    (
      "Vg",
      Vg::<f64, _>::new(0.1, 0.3, 0.4, 1_024, Some(2.0), None, Deterministic::new(2)).sample(),
    ),
    (
      "Nig",
      Nig::<f64, _>::new(0.1, 0.3, 0.4, 1_024, Some(2.0), None, Deterministic::new(3)).sample(),
    ),
    (
      "BilateralGamma",
      BilateralGamma::<f64, _>::new(
        1.0,
        2.0,
        1.0,
        2.0,
        1_024,
        Some(2.0),
        None,
        Deterministic::new(5),
      )
      .sample(),
    ),
    (
      "IgSubordinator",
      IgSubordinator::<f64, _>::new(1.0, 1.0, 1_024, Some(2.0), None, Deterministic::new(7))
        .sample(),
    ),
    (
      "GammaSubordinator",
      GammaSubordinator::<f64, _>::new(1.0, 2.0, 1_024, Some(2.0), None, Deterministic::new(11))
        .sample(),
    ),
    (
      "AlphaStableSubordinator",
      AlphaStableSubordinator::<f64, _>::new(
        0.6,
        1.0,
        1_024,
        Some(2.0),
        None,
        Deterministic::new(13),
      )
      .sample(),
    ),
    (
      "PoissonSubordinator",
      PoissonSubordinator::<f64, _>::new(1.5, 1_024, Some(2.0), None, Deterministic::new(17))
        .sample(),
    ),
  ];
  for (name, path) in checks {
    assert_eq!(path[0], 2.0, "{name} did not start at x0");
    assert!(path.iter().all(|v| v.is_finite()), "{name} left the reals");
    assert!(
      path.windows(2).into_iter().all(|w| w[1] >= w[0])
        || name.starts_with("Vg")
        || name.starts_with("Nig")
        || name.starts_with("Bilateral"),
      "{name} is a subordinator and must not decrease"
    );
  }
}
