//! The short-rate models, against the stationary laws their source papers
//! derive.
//!
//! Every sampler here steps Euler-Maruyama, so what it produces is not the
//! diffusion's law but the law of the chain that approximates it: an
//! Ornstein-Uhlenbeck step is the AR(1) `r + θdt(μ − r) + σ√dt z`, whose own
//! stationary variance is `σ²/(2θ − θ²dt)` rather than `σ²/(2θ)`. Each case
//! states the *model's* closed form and widens the band by exactly that
//! difference, computed from the two forms rather than chosen — which is why
//! the gap has to be small: at the `θdt` used here it is under a tenth of a
//! percent, so nothing a wrong drift could hide behind.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::traits::Fn1D;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::interest::black_karasinski::BlackKarasinski;
use stochastic_rs_stochastic::interest::ho_lee::HoLee;
use stochastic_rs_stochastic::interest::hull_white::HullWhite;
use stochastic_rs_stochastic::interest::vasicek::Vasicek;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::N;
use super::common::PATHS;
use super::common::across_paths;
use super::common::over_paths;
use super::common::spread_over_paths;
use super::common::terminals;
use super::common::autocorr;
use super::common::autocovariance;
use super::common::holds;
use super::common::holds_within;
use super::common::mean;
use super::common::variance;

/// The horizon every stationary case runs to. Long enough that the burn-in
/// prefix is several mean-reversion times at the rates used below, short
/// enough that `θdt` stays near a thousandth.
const HORIZON: f64 = 64.0;

/// The time step the horizon and the grid imply.
fn step() -> f64 {
  HORIZON / (N - 1) as f64
}

/// An Ornstein-Uhlenbeck chain's stationary variance under Euler-Maruyama:
/// the AR(1) with `φ = 1 − θdt` and innovation variance `σ²dt` settles at
/// `σ²dt/(1 − φ²)`, against the diffusion's own `σ²/(2θ)`.
fn euler_stationary_variance(theta: f64, sigma: f64, dt: f64) -> f64 {
  let phi = 1.0 - theta * dt;
  sigma * sigma * dt / (1.0 - phi * phi)
}

/// The mean-reversion rate a path implies: least squares of the increment on
/// the gap to the known long-run mean, which is the maximum-likelihood
/// estimator of `θ` for the chain the sampler steps.
///
/// An autoregressive slope is biased low by about `2φ/n` (Marriott & Pope
/// 1954), which at these lengths is a full two standard errors of `θ̂` —
/// large enough to fail a correct sampler and small enough to look like a
/// wrong one. The correction is applied here rather than widened into the
/// band, so the band stays five standard errors of the estimator itself.
fn reversion_rate(x: &[f64], mu: f64, dt: f64) -> f64 {
  let (mut num, mut den) = (0.0, 0.0);
  for t in 1..x.len() {
    num += (x[t - 1] - mu) * (x[t] - x[t - 1]);
    den += (x[t - 1] - mu).powi(2);
  }
  let slope = 1.0 + num / den;
  let corrected = slope + 2.0 * slope / (x.len() - 1) as f64;
  (1.0 - corrected) / dt
}

/// The diffusion coefficient a path implies: the root mean square of the
/// increments left over once the drift the model names is removed, over `√dt`.
fn diffusion_scale(x: &[f64], mu: f64, dt: f64, theta: f64) -> f64 {
  let residual: f64 = (1..x.len())
    .map(|t| ((x[t] - x[t - 1]) - theta * dt * (mu - x[t - 1])).powi(2))
    .sum();
  (residual / ((x.len() - 1) as f64 * dt)).sqrt()
}

/// Vasicek: the Ornstein-Uhlenbeck stationary law — mean `μ`, variance
/// `σ²/(2θ)`, autocorrelation `e^{−θτ}` (Vasicek 1977, §2; Uhlenbeck &
/// Ornstein 1930). The autocorrelation is what pins the reversion rate:
/// a mean and a variance together leave `θ` and `σ` free along one curve,
/// and the decay picks the point on it.
#[test]
fn vasicek_stationary_moments_match_the_ornstein_uhlenbeck_law() {
  let (theta, mu, sigma) = (1.5, 0.04, 0.3);
  let dt = step();
  let paths = Vasicek::<f64, _>::new(
    theta,
    mu,
    sigma,
    N,
    Some(mu),
    Some(HORIZON),
    Deterministic::new(311),
  )
  .sample_par(PATHS);
  holds(across_paths(&paths, mean), mu, "Vasicek stationary mean");
  let stationary = sigma * sigma / (2.0 * theta);
  let chain = euler_stationary_variance(theta, sigma, dt);
  // The coefficients themselves, which the stationary moments alone cannot
  // separate: `σ²/(2θ)` is one number for a curve of `(θ, σ)` pairs.
  holds(
    across_paths(&paths, |x| reversion_rate(x, mu, dt)),
    theta,
    "Vasicek reversion rate",
  );
  holds(
    across_paths(&paths, |x| diffusion_scale(x, mu, dt, theta)),
    sigma,
    "Vasicek diffusion scale",
  );
  for lag in [0usize, 8, 64, 512] {
    let tau = lag as f64 * dt;
    let law = stationary * (-theta * tau).exp();
    let exact = chain * (1.0 - theta * dt).powi(lag as i32);
    holds_within(
      across_paths(&paths, |x| autocovariance(x, lag, mu)),
      law,
      exact - law,
      &format!("Vasicek autocovariance over {tau}"),
    );
  }
}

/// The same process seen from away from its mean: started at `x0`, the mean
/// returns to `μ` as `μ + (x₀ − μ)e^{−θt}`, which is the transient half of
/// Vasicek's (1977) solution and the half a stationary case cannot see.
#[test]
fn vasicek_reverts_at_the_rate_it_was_given() {
  let (theta, mu, sigma, x0) = (2.0, 0.04, 0.25, 0.14);
  let horizon = 0.5;
  let grid = 2_049;
  let dt = horizon / (grid - 1) as f64;
  let paths = Vasicek::<f64, _>::new(
    theta,
    mu,
    sigma,
    grid,
    Some(x0),
    Some(horizon),
    Deterministic::new(313),
  )
  .sample_par(8_192);
  let terminal = terminals(&paths);
  let law = mu + (x0 - mu) * (-theta * horizon).exp();
  let exact = mu + (x0 - mu) * (1.0 - theta * dt).powi(grid as i32 - 1);
  holds_within(
    over_paths(&terminal),
    law,
    exact - law,
    "Vasicek mean after half a unit of time",
  );
}

/// Ho-Lee with a flat `θ`: `dr = θdt + σdW` integrates to a Brownian motion
/// with drift, so `r_t − r_0` has mean `θt` and variance `σ²t` and its
/// increments are uncorrelated (Ho & Lee 1986). Euler is exact for this one —
/// the sum of the steps *is* the solution — so no scheme bias enters.
#[test]
fn ho_lee_is_a_brownian_motion_with_drift() {
  let (drift, sigma) = (0.02, 0.15);
  let grid = 4_097;
  let horizon = 4.0;
  let paths = HoLee::<f64, _>::new(
    None::<Fn1D<f64>>,
    Some(drift),
    sigma,
    grid,
    Some(horizon),
    Deterministic::new(317),
  )
  .sample_par(4_096);
  let terminal = terminals(&paths);
  holds(over_paths(&terminal), drift * horizon, "Ho-Lee terminal mean");
  holds(
    spread_over_paths(&terminal),
    sigma * sigma * horizon,
    "Ho-Lee terminal variance",
  );
  let differences = |x: &[f64]| {
    let d: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
    autocorr(&d, 1)
  };
  holds(
    across_paths(&paths, differences),
    0.0,
    "Ho-Lee increment autocorrelation",
  );
}

/// Hull-White with a constant `θ` is a Vasicek reverting to `θ/α`
/// (Hull & White 1990, §2 — the extension is that `θ` may move with time,
/// and a flat one is the case whose stationary law is written down).
#[test]
fn hull_white_with_a_flat_theta_reverts_to_theta_over_alpha() {
  const THETA: f64 = 0.09;
  let (alpha, sigma) = (1.5, 0.25);
  let dt = step();
  let flat = (|_t: f64| THETA) as fn(f64) -> f64;
  let paths = HullWhite::<f64, _>::new(
    flat,
    alpha,
    sigma,
    N,
    Some(THETA / alpha),
    Some(HORIZON),
    Deterministic::new(331),
  )
  .sample_par(PATHS);
  holds(
    across_paths(&paths, mean),
    THETA / alpha,
    "Hull-White stationary mean",
  );
  let law = sigma * sigma / (2.0 * alpha);
  holds_within(
    across_paths(&paths, variance),
    law,
    euler_stationary_variance(alpha, sigma, dt) - law,
    "Hull-White stationary variance",
  );
  holds(
    across_paths(&paths, |x| reversion_rate(x, THETA / alpha, dt)),
    alpha,
    "Hull-White reversion rate",
  );
  holds(
    across_paths(&paths, |x| diffusion_scale(x, THETA / alpha, dt, alpha)),
    sigma,
    "Hull-White diffusion scale",
  );
}

/// Black-Karasinski: the *logarithm* of the rate is the Ornstein-Uhlenbeck
/// process, so `ln r` has stationary mean `θ/a` and variance `σ²/(2a)`, and
/// the rate itself is lognormal and positive (Black & Karasinski 1991, §1).
#[test]
fn black_karasinski_log_rate_is_an_ornstein_uhlenbeck() {
  const THETA: f64 = -4.5;
  let (a, sigma) = (1.5, 0.4);
  let dt = step();
  let flat = (|_t: f64| THETA) as fn(f64) -> f64;
  let paths = BlackKarasinski::<f64, _>::new(
    flat,
    a,
    sigma,
    N,
    Some((THETA / a).exp()),
    Some(HORIZON),
    Deterministic::new(337),
  )
  .sample_par(PATHS);
  assert!(
    paths.iter().all(|p| p.iter().all(|&r| r > 0.0)),
    "a Black-Karasinski rate was not positive"
  );
  let logs: Vec<Array1<f64>> = paths.iter().map(|p| p.mapv(f64::ln)).collect();
  holds(
    across_paths(&logs, mean),
    THETA / a,
    "Black-Karasinski mean log-rate",
  );
  let law = sigma * sigma / (2.0 * a);
  holds_within(
    across_paths(&logs, variance),
    law,
    euler_stationary_variance(a, sigma, dt) - law,
    "Black-Karasinski log-rate variance",
  );
}

/// Cox-Ingersoll-Ross: the stationary law is the gamma with shape
/// `2θμ/σ²` and scale `σ²/(2θ)` (Cox, Ingersoll & Ross 1985, eq. 20), whose
/// mean is `μ` and whose variance is `μσ²/(2θ)`. The parameters keep the
/// Feller condition `2θμ > σ²` comfortably — at `2θμ/σ² = 6` the floor at
/// zero the Euler step applies is never reached, so the sampled law is the
/// model's rather than the scheme's boundary treatment.
#[test]
fn cir_stationary_moments_match_the_gamma_law() {
  let (theta, mu, sigma) = (1.5, 0.05, 0.15);
  assert!(
    2.0 * theta * mu / (sigma * sigma) > 5.0,
    "the case must sit well inside the Feller condition"
  );
  let paths = Cir::<f64, _>::new(
    theta,
    mu,
    sigma,
    N,
    Some(mu),
    Some(HORIZON),
    None,
    Deterministic::new(347),
  )
  .sample_par(PATHS);
  assert!(
    paths.iter().all(|p| p.iter().all(|&r| r >= 0.0)),
    "a CIR rate went negative"
  );
  holds(across_paths(&paths, mean), mu, "CIR stationary mean");
  let law = mu * sigma * sigma / (2.0 * theta);
  holds_within(
    across_paths(&paths, variance),
    law,
    euler_stationary_variance(theta, sigma * mu.sqrt(), step()) - law,
    "CIR stationary variance",
  );
  // The gamma's shape shows in its skewness, `2/√shape`, which is what
  // separates the CIR from a Vasicek with the same first two moments.
  let skewness = |x: &[f64]| {
    let (m, v) = (mean(x), variance(x));
    x.iter().map(|s| (s - m).powi(3)).sum::<f64>() / x.len() as f64 / v.powf(1.5)
  };
  holds(
    across_paths(&paths, skewness),
    2.0 / (2.0 * theta * mu / (sigma * sigma)).sqrt(),
    "CIR stationary skewness",
  );
}
