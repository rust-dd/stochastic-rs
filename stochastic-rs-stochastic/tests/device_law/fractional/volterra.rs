use ndarray::Array1;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::rough::kernel::RlKernel;
use stochastic_rs_stochastic::rough::rl_bs::RlBlackScholes;
use stochastic_rs_stochastic::rough::rl_fbm::RlFbm;
use stochastic_rs_stochastic::rough::rl_fou::RlFou;
use stochastic_rs_stochastic::rough::rl_heston::RlHeston;
use stochastic_rs_stochastic::volterra::Volterra;
use stochastic_rs_stochastic::volterra::VolterraKernelSpec;
use stochastic_rs_stochastic::volterra::gaussian_polynomial::GaussianPolynomialVolatility;
use stochastic_rs_stochastic::volterra::square_root::VolterraSquareRoot;

use super::*;

/// Riemann-Liouville fBm through the Markov lift, whose node states the frame carries per path.
/// Horizon 4 tells the spread `t^H` (1.52 at `H = 0.3`) from Brownian motion's `sqrt(t)` (2.00).
#[test]
fn rl_fbm_agrees_with_the_cpu_law() {
  let build = || RlFbm::<f32, _>::new(0.3, N, Some(4.0), None, Deterministic::new(29));
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  assert_eq!(device[0].len(), N);
  assert_eq!(device[0][0], 0.0, "every path starts at the origin");
  all_finite(&device, "RL-fBm");
  agrees(
    terminal_std(&host),
    terminal_std(&device),
    0.06,
    "RL-fBm terminal spread",
  );
  // The spread a quarter of the way in pins the roughness, not just the scale:
  // `(t/4)^H / t^H = 4^-H` is 0.66 at `H = 0.3` and 0.5 for Brownian motion.
  let quarter = |paths: &[Array1<f32>]| {
    let k = paths[0].len() / 4;
    let n = paths.len() as f64;
    let mean = paths.iter().map(|p| p[k] as f64).sum::<f64>() / n;
    (paths
      .iter()
      .map(|p| (p[k] as f64 - mean).powi(2))
      .sum::<f64>()
      / n)
      .sqrt()
  };
  agrees(
    quarter(&host),
    quarter(&device),
    0.06,
    "RL-fBm quarter-horizon spread",
  );
}

/// The rough OU takes the lifted fBm's increments in its own Euler loop; the
/// terminal mean pins `kappa` and `mu`, the spread `nu` and the roughness.
#[test]
fn rl_fou_agrees_with_the_cpu_law() {
  let build = || {
    RlFou::<f32, _>::new(
      0.3,
      2.0,
      0.05,
      0.3,
      N,
      Some(0.02),
      Some(1.0),
      None,
      Deterministic::new(37),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  assert_eq!(device[0].len(), N);
  assert!(
    (device[0][0] - 0.02).abs() < 1e-6,
    "every path starts at x0"
  );
  all_finite(&device, "RL-fOU");
  let (hm, dm) = (terminal_mean(&host), terminal_mean(&device));
  assert!(
    (hm - dm).abs() < 0.01,
    "RL-fOU terminal mean: host {hm}, device {dm}"
  );
  agrees(
    terminal_std(&host),
    terminal_std(&device),
    0.06,
    "RL-fOU terminal spread",
  );
}

/// The rough Black-Scholes is a closed form of the lifted fBm plus a curve;
/// the terminal log-return's mean pins the curve and its spread the lift.
#[test]
fn rl_black_scholes_agrees_with_the_cpu_law() {
  let build = || {
    RlBlackScholes::<f32, _>::new(
      0.3,
      100.0,
      0.03,
      0.2,
      N,
      Some(1.0),
      None,
      Deterministic::new(41),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  assert_eq!(device[0][0], 100.0, "every path starts at s0");
  all_finite(&device, "RL-Black-Scholes");
  let log_ret = |paths: &[Array1<f32>]| -> Vec<f64> {
    let last = paths[0].len() - 1;
    paths
      .iter()
      .map(|p| (p[last] as f64 / 100.0).ln())
      .collect()
  };
  let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
  let sd = |v: &[f64]| {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
  };
  let (h, d) = (log_ret(&host), log_ret(&device));
  assert!(
    (mean(&h) - mean(&d)).abs() < 0.01,
    "RL-Black-Scholes log-return mean: host {}, device {}",
    mean(&h),
    mean(&d)
  );
  agrees(sd(&h), sd(&d), 0.06, "RL-Black-Scholes log-return spread");
}

/// The rough Heston puts the lift on the variance. Its terminal mean pins `kappa` and `theta`,
/// its spread `nu` and the roughness, and the spot/variance correlation pins `rho`.
#[test]
fn rl_heston_agrees_with_the_cpu_law() {
  let build = || {
    RlHeston::<f32, _>::new(
      0.3,
      100.0,
      0.04,
      2.0,
      0.04,
      0.3,
      -0.6,
      0.03,
      N,
      Some(1.0),
      None,
      Deterministic::new(43),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  assert_eq!(device[0][0][0], 100.0, "the spot starts at s0");
  assert_eq!(device[0][1][0], 0.04, "the variance starts at v0");
  assert!(
    device
      .iter()
      .all(|[s, v]| s.iter().all(|x| x.is_finite()) && v.iter().all(|x| x.is_finite() && *x >= 0.0)),
    "rough Heston: a device path left its domain"
  );
  let last = N - 1;
  let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
  let sd = |v: &[f64]| {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
  };
  let corr = |a: &[f64], b: &[f64]| {
    let (ma, mb) = (mean(a), mean(b));
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
      sab += (x - ma) * (y - mb);
      saa += (x - ma).powi(2);
      sbb += (y - mb).powi(2);
    }
    sab / (saa * sbb).sqrt()
  };
  let log_spot = |paths: &[[Array1<f32>; 2]]| -> Vec<f64> {
    paths
      .iter()
      .map(|[s, _]| (s[last] as f64 / 100.0).ln())
      .collect()
  };
  let variance = |paths: &[[Array1<f32>; 2]]| -> Vec<f64> {
    paths.iter().map(|[_, v]| v[last] as f64).collect()
  };
  let (hs, ds, hv, dv) = (
    log_spot(&host),
    log_spot(&device),
    variance(&host),
    variance(&device),
  );
  assert!(
    (mean(&hs) - mean(&ds)).abs() < 0.01,
    "rough Heston log-spot mean: host {}, device {}",
    mean(&hs),
    mean(&ds)
  );
  agrees(sd(&hs), sd(&ds), 0.06, "rough Heston log-spot spread");
  assert!(
    (mean(&hv) - mean(&dv)).abs() < 0.002,
    "rough Heston variance mean: host {}, device {}",
    mean(&hv),
    mean(&dv)
  );
  agrees(sd(&hv), sd(&dv), 0.08, "rough Heston variance spread");
  let (hc, dc) = (corr(&hs, &hv), corr(&ds, &dv));
  assert!(
    (hc - dc).abs() < 0.05,
    "rough Heston spot/variance correlation: host {hc}, device {dc}"
  );
}

/// The general Volterra process: the lift branch is fBm under the Markov lift, and a kernel
/// outside the rough range takes the reference convolution, run by the history block.
#[test]
fn volterra_lift_and_reference_agree_with_the_cpu_law() {
  let build = || {
    Volterra::<f32, _>::new(
      VolterraKernelSpec::FractionalBM { h: 0.3 },
      N,
      Some(4.0),
      Deterministic::new(47),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  assert_eq!(device[0][0], 0.0, "every path starts at the origin");
  all_finite(&device, "Volterra lift");
  agrees(
    terminal_std(&host),
    terminal_std(&device),
    0.06,
    "Volterra lift terminal spread",
  );
  let reference = || {
    Volterra::<f32, _>::new(
      VolterraKernelSpec::FractionalBM { h: 0.7 },
      N,
      Some(4.0),
      Deterministic::new(53),
    )
  };
  let device = reference().on::<Device>().sample_par(PATHS);
  let host = reference().sample_par(PATHS);
  assert_eq!(
    device[0][0], 0.0,
    "every reference path starts at the origin"
  );
  all_finite(&device, "Volterra reference");
  agrees(
    terminal_std(&host),
    terminal_std(&device),
    0.06,
    "Volterra reference terminal spread",
  );
  // A second point on the path's spread: a launch that convolved with the wrong lags would move
  // the quarter-way spread by a different amount than the terminal one.
  let quarter = |paths: &[Array1<f32>]| {
    let k = paths[0].len() / 4;
    let n = paths.len() as f64;
    let mean = paths.iter().map(|p| p[k] as f64).sum::<f64>() / n;
    (paths
      .iter()
      .map(|p| (p[k] as f64 - mean).powi(2))
      .sum::<f64>()
      / n)
      .sqrt()
  };
  agrees(
    quarter(&host),
    quarter(&device),
    0.06,
    "Volterra reference quarter-horizon spread",
  );
}

/// A reference grid longer than the kernels' history samples on the host,
/// and is then the host build to the bit.
#[test]
fn a_longer_reference_grid_keeps_volterra_on_the_host() {
  let build = || {
    Volterra::<f32, _>::new(
      VolterraKernelSpec::Exponential { beta: 2.0 },
      600,
      Some(1.0),
      Deterministic::new(59),
    )
  };
  assert_eq!(build().on::<Device>().sample_par(8), build().sample_par(8));
}

#[test]
fn volterra_square_root_agrees_with_the_cpu_law() {
  let build = || {
    VolterraSquareRoot::<f32, _, _>::new(
      RlKernel::new(0.3, 12),
      2.0,
      0.04,
      0.3,
      N,
      0.04,
      Some(1.0),
      Deterministic::new(43),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device[0][0], 0.04, "every path starts at v0");
  within(&device, 0.0, f32::INFINITY, "Volterra square root");
  means_agree(&host, &device, "Volterra square root terminal mean");
  spreads_agree(&host, &device, "Volterra square root terminal spread");
}

#[test]
fn gaussian_polynomial_volatility_agrees_with_the_cpu_law() {
  let build = || {
    GaussianPolynomialVolatility::<f32, _, _>::new(
      RlKernel::new(0.3, 12),
      array![0.2_f32, 0.5, 0.1],
      N,
      Some(1.0),
      Deterministic::new(47),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert!(
    device.iter().all(|p| p[0] == 0.2),
    "every path starts at the constant term"
  );
  all_finite(&device, "Gaussian polynomial volatility");
  agrees(
    terminal_mean(&host),
    terminal_mean(&device),
    0.08,
    "Gaussian polynomial volatility terminal mean",
  );
  agrees(
    terminal_std(&host),
    terminal_std(&device),
    0.06,
    "Gaussian polynomial volatility terminal spread",
  );
}

/// Nine coefficients exceed the kernels' slots, so the device build samples
/// on the host and is the host build to the bit.
#[test]
fn a_longer_polynomial_keeps_the_process_on_the_host() {
  let build = || {
    GaussianPolynomialVolatility::<f32, _, _>::new(
      RlKernel::new(0.3, 12),
      array![0.2_f32, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01],
      N,
      Some(1.0),
      Deterministic::new(53),
    )
  };
  assert_eq!(build().on::<Device>().sample_par(8), build().sample_par(8));
}
