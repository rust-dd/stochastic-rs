//! Gaussian noise, fractional Gaussian noise, Brownian motion and the Brownian bridge against
//! their covariances: each law is exact on any grid, so no scheme bias widens a band.

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::noise::cgns::Cgns;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::noise::gn::Gn;
use stochastic_rs_stochastic::process::bm::Bm;
use stochastic_rs_stochastic::process::brownian_bridge::BrownianBridge;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::N;
use super::common::PATHS;
use super::common::across_paths;
use super::common::autocovariance;
use super::common::holds;
use super::common::over_paths;
use super::common::spread_over_paths;
use super::common::terminals;

/// Paths for the cases that read one point per path, which carries less information than a
/// path's covariance: more paths for the same band, on a short grid to keep them cheap.
const POINT_PATHS: usize = 4_096;

/// Grid for those cases: 257 points put an index exactly at the midpoint.
const POINT_GRID: usize = 257;

/// Gaussian noise: independent `N(0, t/n)` steps (the module doc's
/// `ΔW_i ~ N(0, Δt)` with `Δt = t/n`).
#[test]
fn gaussian_noise_has_the_step_variance_and_no_memory() {
  let t = 4.0;
  let paths = Gn::<f64, _>::new(N, Some(t), Deterministic::new(11)).sample_par(PATHS);
  let dt = t / N as f64;

  holds(
    across_paths(&paths, |x| autocovariance(x, 0, 0.0)),
    dt,
    "Gn variance",
  );
  holds(
    across_paths(&paths, |x| autocovariance(x, 1, 0.0)),
    0.0,
    "Gn lag-1 autocovariance",
  );
}

/// Correlated Gaussian noise: two `N(0, t/n)` streams whose steps have
/// correlation `ρ`, `Z₂ = ρZ₁ + √(1 − ρ²)ε`.
#[test]
fn correlated_noise_has_the_requested_correlation() {
  let (rho, t) = (-0.6, 1.0);
  let pairs = Cgns::<f64, _>::new(rho, N, Some(t), Deterministic::new(12)).sample_par(PATHS);

  let correlations = pairs
    .iter()
    .map(|[a, b]| {
      let cross = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>();
      let norms = a.iter().map(|x| x * x).sum::<f64>() * b.iter().map(|y| y * y).sum::<f64>();
      cross / norms.sqrt()
    })
    .collect::<Vec<_>>();
  holds(over_paths(&correlations), rho, "Cgns step correlation");

  let first = pairs.iter().map(|[a, _]| a.clone()).collect::<Vec<_>>();
  holds(
    across_paths(&first, |x| autocovariance(x, 0, 0.0)),
    t / N as f64,
    "Cgns step variance",
  );
}

/// Fractional Gaussian noise at `H` on `len` steps over `[0, t]`: variance `(t/len)^{2H}` and
/// lag-`k` correlation `½(|k+1|^{2H} − 2|k|^{2H} + |k−1|^{2H})`.
#[test]
fn fractional_noise_has_the_fractional_covariance() {
  let (hurst, t) = (0.3, 1.0);
  let paths = Fgn::<f64, _>::new(hurst, N, Some(t), Deterministic::new(13)).sample_par(PATHS);
  let len = paths[0].len() as f64;
  let two_h = 2.0 * hurst;
  let correlation =
    |k: f64| 0.5 * ((k + 1.0).powf(two_h) - 2.0 * k.powf(two_h) + (k - 1.0).abs().powf(two_h));
  let variance = (t / len).powf(two_h);

  holds(
    across_paths(&paths, |x| autocovariance(x, 0, 0.0)),
    variance,
    "fGn variance",
  );
  holds(
    across_paths(&paths, |x| autocovariance(x, 1, 0.0)),
    variance * correlation(1.0),
    "fGn lag-1 autocovariance",
  );
  holds(
    across_paths(&paths, |x| autocovariance(x, 2, 0.0)),
    variance * correlation(2.0),
    "fGn lag-2 autocovariance",
  );
}

/// Brownian motion from zero: `B_t ~ N(0, t)` at the horizon.
#[test]
fn brownian_motion_reaches_the_horizon_variance() {
  let t = 2.0;
  let paths =
    Bm::<f64, _>::new(POINT_GRID, Some(t), Deterministic::new(14)).sample_par(POINT_PATHS);
  let ends = terminals(&paths);

  holds(over_paths(&ends), 0.0, "Bm terminal mean");
  holds(spread_over_paths(&ends), t, "Bm terminal variance");
}

/// The Brownian bridge pinned at `x₀` and `x_T`: `X_{T/2} ~ N((x₀ + x_T)/2, σ²T/4)`. Each point
/// is drawn from its exact conditional law, so this holds on any grid and the last point is `x_T`.
#[test]
fn brownian_bridge_has_the_pinned_midpoint_law() {
  let (sigma, x0, xt, t) = (0.8, 1.0, -0.5, 3.0);
  let paths = BrownianBridge::<f64, _>::new(
    sigma,
    POINT_GRID,
    Some(x0),
    Some(xt),
    Some(t),
    Deterministic::new(15),
  )
  .sample_par(POINT_PATHS);
  let midpoints = paths
    .iter()
    .map(|p| p[(POINT_GRID - 1) / 2])
    .collect::<Vec<_>>();

  holds(
    over_paths(&midpoints),
    0.5 * (x0 + xt),
    "bridge midpoint mean",
  );
  holds(
    spread_over_paths(&midpoints),
    sigma * sigma * t / 4.0,
    "bridge midpoint variance",
  );
  assert!(paths.iter().all(|p| p[POINT_GRID - 1] == xt));
}
