//! The diffusions whose state lives on an interval or a half-line, against
//! the stationary laws their coefficients imply.
//!
//! A one-dimensional diffusion `dX = b(x)dt + s(x)dW` that has a stationary
//! law at all has this one:
//!
//! ```text
//! p(x) ∝ s(x)^{-2} exp ∫ 2b(u)/s(u)² du
//! ```
//!
//! which is where every closed form below comes from — a Beta for the
//! Jacobi, a gamma for the logistic, an inverse gamma for the 3/2 model, a
//! lognormal for the Gompertz. Each case names the family, then holds the
//! sampler to its first two moments and to the shape statistic that
//! distinguishes it from a normal law with the same ones.
//!
//! As in [`super::short_rate`], the samplers step Euler-Maruyama, so the band
//! carries the scheme's own bias where it is not negligible, and every case
//! keeps its coefficients well inside the region where the boundaries are
//! unattainable — a clamped path is the scheme's law, not the model's.

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::gompertz::Gompertz;
use stochastic_rs_stochastic::diffusion::jacobi::Jacobi;
use stochastic_rs_stochastic::diffusion::logistic::Logistic;
use stochastic_rs_stochastic::diffusion::pearson::Pearson;
use stochastic_rs_stochastic::diffusion::three_half::ThreeHalf;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::N;
use super::common::PATHS;
use super::common::across_paths;
use super::common::holds;
use super::common::mean;
use super::common::variance;

/// The horizon every case runs to, as in the short-rate wave: long enough for
/// the burn-in prefix to be several mean-reversion times, short enough that
/// the step stays small against every rate below.
const HORIZON: f64 = 64.0;

/// The time step the horizon and the grid imply.
fn step() -> f64 {
  HORIZON / (N - 1) as f64
}

/// The sample skewness, the shape statistic that separates a Beta or a gamma
/// from the normal law with the same mean and variance.
fn skewness(x: &[f64]) -> f64 {
  let (m, v) = (mean(x), variance(x));
  x.iter().map(|s| (s - m).powi(3)).sum::<f64>() / x.len() as f64 / v.powf(1.5)
}

/// Jacobi: `dX = (α − βX)dt + σ√(X(1−X))dW` is the Wright-Fisher diffusion,
/// whose stationary law is `Beta(2α/σ², 2(β−α)/σ²)` (Karlin & Taylor 1981,
/// §15.2). Its mean is `α/β` and its variance `θ(1−θ)/(1 + 2β/σ²)`, and both
/// shape parameters here exceed one, so neither boundary is attainable and
/// the sampler's clamp never fires.
#[test]
fn jacobi_stationary_law_is_the_beta_its_coefficients_name() {
  let (alpha, beta, sigma) = (0.9, 3.0, 0.5);
  let (a, b) = (
    2.0 * alpha / (sigma * sigma),
    2.0 * (beta - alpha) / (sigma * sigma),
  );
  assert!(a > 1.0 && b > 1.0, "the boundaries must be unattainable");
  let paths = Jacobi::<f64, _>::new(
    alpha,
    beta,
    sigma,
    N,
    Some(alpha / beta),
    Some(HORIZON),
    Deterministic::new(401),
  )
  .sample_par(PATHS);
  assert!(
    paths
      .iter()
      .all(|p| p.iter().all(|&x| (0.0..=1.0).contains(&x))),
    "a Jacobi path left the unit interval"
  );
  holds(
    across_paths(&paths, mean),
    a / (a + b),
    "Jacobi stationary mean",
  );
  holds(
    across_paths(&paths, variance),
    a * b / ((a + b).powi(2) * (a + b + 1.0)),
    "Jacobi stationary variance",
  );
  holds(
    across_paths(&paths, skewness),
    2.0 * (b - a) * (a + b + 1.0).sqrt() / ((a + b + 2.0) * (a * b).sqrt()),
    "Jacobi stationary skewness",
  );
}

/// The stochastic logistic `dX = X(1 − aX)dt + bX dW`: the density above
/// gives `Gamma(2/b² − 1, b²/(2a))` (Goel & Richter-Dyn 1974, §3.3 — the
/// deterministic carrying capacity `1/a` is *not* the stochastic mean, which
/// sits `b²/2` of it lower).
#[test]
fn logistic_stationary_law_is_the_gamma_its_noise_shifts() {
  let (a, b) = (2.0, 0.4);
  let shape = 2.0 / (b * b) - 1.0;
  let scale = b * b / (2.0 * a);
  let paths = Logistic::<f64, _>::new(
    a,
    b,
    N,
    Some(shape * scale),
    Some(HORIZON),
    Deterministic::new(409),
  )
  .sample_par(PATHS);
  assert!(
    paths.iter().all(|p| p.iter().all(|&x| x > 0.0)),
    "a logistic path was not positive"
  );
  holds(
    across_paths(&paths, mean),
    shape * scale,
    "logistic stationary mean",
  );
  holds(
    across_paths(&paths, variance),
    shape * scale * scale,
    "logistic stationary variance",
  );
  holds(
    across_paths(&paths, skewness),
    2.0 / shape.sqrt(),
    "logistic stationary skewness",
  );
  // The carrying capacity the drift alone would settle at, against the mean
  // the noise actually leaves: the whole content of the closed form.
  assert!(
    shape * scale < 1.0 / a,
    "the noise must pull the mean below the deterministic capacity"
  );
}

/// Pearson: `dX = κ(μ − X)dt + √(2κ(aX² + bX + c))dW`. Whatever family the
/// quadratic puts it in, stationarity forces `Var = (aμ² + bμ + c)/(1 − a)` —
/// multiply the generator's action on `(x − μ)²` out and set it to zero
/// (Forman & Sørensen 2008, §2). The case below takes `a > 0`, which is the
/// heavy-tailed (Student) branch, so the identity is doing real work.
#[test]
fn pearson_stationary_variance_follows_from_its_quadratic() {
  let (kappa, mu, a, b, c) = (2.0, 0.5, 0.05, 0.0, 0.02);
  let law = (a * mu * mu + b * mu + c) / (1.0 - a);
  let paths = Pearson::<f64, _>::new(
    kappa,
    mu,
    a,
    b,
    c,
    N,
    Some(mu),
    Some(HORIZON),
    Deterministic::new(419),
  )
  .sample_par(PATHS);
  holds(across_paths(&paths, mean), mu, "Pearson stationary mean");
  holds(
    across_paths(&paths, variance),
    law,
    "Pearson stationary variance",
  );
}

/// Gompertz: `dX = (a − b ln X)X dt + σX dW`, so `ln X` is an
/// Ornstein-Uhlenbeck with rate `b` and long-run mean `(a − σ²/2)/b` — the
/// Itô correction is the whole point, and a sampler that dropped it would sit
/// `σ²/(2b)` too high (Gompertz 1825; the stochastic form, Lo 2007, §2).
#[test]
fn gompertz_log_state_is_an_ornstein_uhlenbeck_with_the_ito_shift() {
  let (a, b, sigma) = (0.6, 1.5, 0.35);
  let log_mean = (a - 0.5 * sigma * sigma) / b;
  let paths = Gompertz::<f64, _>::new(
    a,
    b,
    sigma,
    N,
    Some(log_mean.exp()),
    Some(HORIZON),
    Deterministic::new(421),
  )
  .sample_par(PATHS);
  assert!(
    paths.iter().all(|p| p.iter().all(|&x| x > 0.0)),
    "a Gompertz path was not positive"
  );
  let logs: Vec<_> = paths.iter().map(|p| p.mapv(f64::ln)).collect();
  let dt = step();
  let phi = 1.0 - b * dt;
  let chain = sigma * sigma * dt / (1.0 - phi * phi);
  holds(
    across_paths(&logs, mean),
    log_mean,
    "Gompertz mean log-state",
  );
  super::common::holds_within(
    across_paths(&logs, variance),
    sigma * sigma / (2.0 * b),
    chain - sigma * sigma / (2.0 * b),
    "Gompertz log-state variance",
  );
}

/// The 3/2 model `dV = κV(μ − V)dt + σV^{3/2}dW`: the density above gives an
/// inverse gamma with shape `2 + 2κ/σ²` and scale `2κμ/σ²` (Lewis 2000,
/// §2.2), whose mean is `2κμ/(2κ + σ²)` — below `μ`, where a square-root
/// model would sit at `μ` exactly. That gap is what the case pins.
#[test]
fn three_half_stationary_law_is_the_inverse_gamma() {
  let (kappa, mu, sigma) = (4.0, 0.09, 0.8);
  let shape = 2.0 + 2.0 * kappa / (sigma * sigma);
  let scale = 2.0 * kappa * mu / (sigma * sigma);
  let law_mean = scale / (shape - 1.0);
  let paths = ThreeHalf::<f64, _>::new(
    kappa,
    mu,
    sigma,
    N,
    Some(law_mean),
    Some(HORIZON),
    Deterministic::new(431),
  )
  .sample_par(PATHS);
  assert!(
    paths.iter().all(|p| p.iter().all(|&v| v > 0.0)),
    "a 3/2 path was not positive"
  );
  holds(across_paths(&paths, mean), law_mean, "3/2 stationary mean");
  holds(
    across_paths(&paths, variance),
    law_mean * law_mean / (shape - 2.0),
    "3/2 stationary variance",
  );
  assert!(
    law_mean < mu,
    "the 3/2 model's stationary mean must sit below its drift target"
  );
}
