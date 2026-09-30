//! Geometric Brownian motion, against the moments of the log-normal law.
//!
//! `Gbm` steps Euler-Maruyama on the price, `S ← S(1 + μΔt + σΔW)`, so the
//! chain's own moments are exact products over its `m` steps,
//! `E[S_T] = S₀(1 + μΔt)ᵐ` and `E[S_T²] = S₀²((1 + μΔt)² + σ²Δt)ᵐ`, and
//! each case holds the diffusion's closed form widened by exactly the gap to
//! them. `GbmLog` steps the logarithm exactly, so its law is the diffusion's
//! own and needs no widening.

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::diffusion::gbm_log::GbmLog;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::holds;
use super::common::holds_within;
use super::common::over_paths;
use super::common::spread_over_paths;
use super::common::terminals;

/// Paths per case: each contributes one terminal value.
const PATHS: usize = 4_096;

/// Steps per path.
const STEPS: usize = 256;

#[test]
fn gbm_terminal_moments_match_the_lognormal_law() {
  let (mu, sigma, s0, t) = (0.05, 0.2, 100.0, 1.0);
  let paths = Gbm::<f64, _>::new(
    mu,
    sigma,
    STEPS + 1,
    Some(s0),
    Some(t),
    Deterministic::new(21),
  )
  .sample_par(PATHS);
  let ends = terminals(&paths);
  let dt = t / STEPS as f64;
  let m = STEPS as i32;

  let mean = s0 * (mu * t).exp();
  let chain_mean = s0 * (1.0 + mu * dt).powi(m);
  holds_within(
    over_paths(&ends),
    mean,
    chain_mean - mean,
    "Gbm terminal mean",
  );

  let variance = s0 * s0 * (2.0 * mu * t).exp() * ((sigma * sigma * t).exp() - 1.0);
  let chain_second = s0 * s0 * ((1.0 + mu * dt).powi(2) + sigma * sigma * dt).powi(m);
  let chain_variance = chain_second - chain_mean * chain_mean;
  holds_within(
    spread_over_paths(&ends),
    variance,
    chain_variance - variance,
    "Gbm terminal variance",
  );
}

/// `ln(S_T/S₀) ~ N((μ − σ²/2)T, σ²T)` exactly: the module doc's log-increment
/// scheme.
#[test]
fn gbm_log_is_exactly_lognormal() {
  let (mu, sigma, s0, t) = (0.03, 0.35, 50.0, 2.0);
  let paths = GbmLog::<f64, _>::new(
    Some(mu),
    None,
    None,
    None,
    sigma,
    STEPS + 1,
    Some(s0),
    Some(t),
    Deterministic::new(22),
  )
  .sample_par(PATHS);
  let log_returns = terminals(&paths)
    .iter()
    .map(|s| (s / s0).ln())
    .collect::<Vec<_>>();

  holds(
    over_paths(&log_returns),
    (mu - 0.5 * sigma * sigma) * t,
    "GbmLog log-return mean",
  );
  holds(
    spread_over_paths(&log_returns),
    sigma * sigma * t,
    "GbmLog log-return variance",
  );
}
