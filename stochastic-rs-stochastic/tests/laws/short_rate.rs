//! The short-rate models against their sources' stationary laws. The samplers step
//! Euler-Maruyama, so each band is widened by the computed gap from the chain's law to the model's.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::interest::black_karasinski::BlackKarasinski;
use stochastic_rs_stochastic::interest::ho_lee::HoLee;
use stochastic_rs_stochastic::interest::hull_white::HullWhite;
use stochastic_rs_stochastic::interest::vasicek::Vasicek;
use stochastic_rs_stochastic::params::HoLeeDrift;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::N;
use super::common::PATHS;
use super::common::across_paths;
use super::common::autocorr;
use super::common::autocovariance;
use super::common::holds;
use super::common::holds_within;
use super::common::mean;
use super::common::over_paths;
use super::common::spread_over_paths;
use super::common::terminals;
use super::common::variance;

/// The horizon every stationary case runs to: the burn-in prefix is several mean-reversion
/// times at the rates below, and `θdt` stays near a thousandth.
const HORIZON: f64 = 64.0;

/// The time step the horizon and the grid imply.
fn step() -> f64 {
  HORIZON / (N - 1) as f64
}

/// An Ornstein-Uhlenbeck chain's stationary variance under Euler-Maruyama: the AR(1) with
/// `φ = 1 − θdt` and innovation variance `σ²dt` settles at `σ²dt/(1 − φ²)`, not `σ²/(2θ)`.
fn euler_stationary_variance(theta: f64, sigma: f64, dt: f64) -> f64 {
  let phi = 1.0 - theta * dt;
  sigma * sigma * dt / (1.0 - phi * phi)
}

/// The mean-reversion rate a path implies, by least squares of the increment on the gap to the
/// known mean, with the slope's `2φ/n` downward bias (two standard errors here) corrected.
/// Reference: Marriott & Pope (1954), DOI 10.1093/biomet/41.3-4.390
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

/// Vasicek: stationary mean `μ`, variance `σ²/(2θ)` and autocorrelation `e^{−θτ}`. The decay
/// pins the reversion rate, which a mean and a variance leave free along one curve.
/// Reference: Vasicek (1977), §2, DOI 10.1016/0304-405X(77)90016-2
/// Reference: Uhlenbeck & Ornstein (1930), DOI 10.1103/PhysRev.36.823
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

/// Started at `x0`, the mean returns to `μ` as `μ + (x₀ − μ)e^{−θt}`: the transient half of the
/// solution, which a stationary case cannot see.
/// Reference: Vasicek (1977), DOI 10.1016/0304-405X(77)90016-2
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

/// Ho-Lee with a flat `θ`, `dr = θdt + σdW`: `r_t − r_0` has mean `θt` and variance `σ²t` with
/// uncorrelated increments. Euler is exact here, so no scheme bias enters.
/// Reference: Ho & Lee (1986), DOI 10.1111/j.1540-6261.1986.tb02528.x
#[test]
fn ho_lee_is_a_brownian_motion_with_drift() {
  let (drift, sigma) = (0.02, 0.15);
  let grid = 4_097;
  let horizon = 4.0;
  let paths = HoLee::<f64, _>::new(
    HoLeeDrift::Constant(drift),
    sigma,
    grid,
    Some(horizon),
    Deterministic::new(317),
  )
  .sample_par(4_096);
  let terminal = terminals(&paths);
  holds(
    over_paths(&terminal),
    drift * horizon,
    "Ho-Lee terminal mean",
  );
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

/// Hull-White with a constant `θ` is a Vasicek reverting to `θ/α`, the case of the extension
/// whose stationary law is written down.
/// Reference: Hull & White (1990), §2, DOI 10.1093/rfs/3.4.573
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

/// Black-Karasinski: `ln r` is the Ornstein-Uhlenbeck process, with stationary mean `θ/a` and
/// variance `σ²/(2a)`, so the rate is lognormal and positive.
/// Reference: Black & Karasinski (1991), §1, DOI 10.2469/faj.v47.n4.52
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

/// Cox-Ingersoll-Ross: a stationary gamma with shape `2θμ/σ²` and scale `σ²/(2θ)`, so mean `μ`
/// and variance `μσ²/(2θ)`. At `2θμ/σ² = 6` the Euler step's floor at zero is never reached.
/// Reference: Cox, Ingersoll & Ross (1985), eq. 20, DOI 10.2307/1911242
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
    VarianceBoundary::Floor,
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
