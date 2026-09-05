//! The fGN-driven processes reach a device through the Euler engine: the
//! device runs the fractional-noise pipeline itself, keeps the increments in
//! its own memory and hands them to the same families that serve the Gaussian
//! processes. The two sides draw different streams, so what is pinned here is
//! the law and the boundary, not the path.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::cfou::Cfou;
use stochastic_rs_stochastic::diffusion::fcir::Fcir;
use stochastic_rs_stochastic::diffusion::fgbm::Fgbm;
use stochastic_rs_stochastic::diffusion::fjacobi::FJacobi;
use stochastic_rs_stochastic::diffusion::fou::Fou;
use stochastic_rs_stochastic::interest::fractional_vasicek::FVasicek;
use stochastic_rs_stochastic::noise::cfgns::Cfgns;
use stochastic_rs_stochastic::process::cfbms::Cfbms;
use stochastic_rs_stochastic::process::fbm::Fbm;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::Device;
use super::common::agrees;
use super::common::all_finite;
use super::common::terminal_mean;
use super::common::terminal_std;
use super::common::within;

/// Fewer paths than the Gaussian cases use: each one carries a 512-point fGN
/// draw, and the comparisons below are on statistics that settle early.
const M: usize = 2_000;
const N: usize = 512;

#[test]
fn fou_agrees_with_the_cpu_law() {
  let build = || {
    Fou::<f32, _>::new(
      0.7,
      2.0,
      1.0,
      0.3,
      N,
      Some(0.0),
      Some(1.0),
      Deterministic::new(9),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  assert_eq!(device.len(), M);
  assert_eq!(device[0].len(), N);
  assert_eq!(device[0][0], 0.0, "every path starts at x0");
  all_finite(&device, "fOU");
  let (host, dev) = (
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
  );
  assert!(
    (host - dev).abs() < 0.05,
    "fOU terminal mean: host {host}, device {dev}"
  );
}

#[test]
fn fgbm_agrees_with_the_cpu_law() {
  let build = || {
    Fgbm::<f32, _>::new(
      0.7,
      0.05,
      0.2,
      N,
      Some(100.0),
      Some(1.0),
      Deterministic::new(9),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "fGBM");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.02,
    "fGBM terminal mean",
  );
}

#[test]
fn fcir_stays_nonnegative_and_agrees() {
  let build = || {
    Fcir::<f32, _>::new(
      0.7,
      2.0,
      0.04,
      0.1,
      N,
      Some(0.04),
      Some(1.0),
      None,
      Deterministic::new(9),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, f32::INFINITY, "fCIR");
  let (host, dev) = (
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
  );
  assert!(
    (host - dev).abs() < 0.01,
    "fCIR terminal mean: host {host}, device {dev}"
  );
}

#[test]
fn fjacobi_stays_in_the_unit_interval() {
  let build = || {
    FJacobi::<f32, _>::new(
      0.7,
      0.3,
      0.6,
      0.2,
      N,
      Some(0.5),
      Some(1.0),
      Deterministic::new(9),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, 1.0, "fJacobi");
  let (host, dev) = (
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
  );
  assert!(
    (host - dev).abs() < 0.02,
    "fJacobi terminal mean: host {host}, device {dev}"
  );
}

/// Fractional Brownian motion is the additive family fed fractional
/// increments, and its mean is zero by construction, so the spread carries
/// the law.
#[test]
fn fbm_agrees_with_the_cpu_law() {
  let build = || Fbm::<f32, _>::new(0.7, N, Some(1.0), Deterministic::new(9));
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "fBM");
  agrees(
    terminal_std(&build().sample_par(M)),
    terminal_std(&device),
    0.06,
    "fBM terminal spread",
  );
}

/// The fractional Vasicek is the fOU under short-rate names, and it reaches
/// the device through the wrapped process's own increment pipeline rather
/// than one of its own.
#[test]
fn fractional_vasicek_agrees_with_the_cpu_law() {
  let build = || {
    FVasicek::<f32, _>::new(
      0.7,
      2.0,
      0.04,
      0.02,
      N,
      Some(0.03),
      Some(1.0),
      Deterministic::new(9),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "fractional Vasicek");
  let (host, dev) = (
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
  );
  assert!(
    (host - dev).abs() < 0.01,
    "fractional Vasicek terminal mean: host {host}, device {dev}"
  );
}

/// The correlated fractional pair is the only family that reads two streams
/// out of one embedding: the device draws `2 · m` paths in a single batched
/// call and the step takes its second stream from the buffer's next `paths`
/// rows. Both the marginal spread and the pair's own correlation are checked,
/// since a mis-indexed second block would keep the marginals right and lose
/// exactly the correlation.
#[test]
fn correlated_fbm_agrees_with_the_cpu_law() {
  // Horizon 4 rather than 1: at `t = 1` the terminal spread of fBm is
  // `t^H = 1` and of Brownian motion `sqrt(t) = 1`, so a launch that read a
  // hashed Gaussian instead of the fractional stream would pass. At 4 they
  // are 2.64 and 2.00. The path count is raised with it because the
  // correlation estimator's standard error at `M` paths is a third of the
  // tolerance, which is too thin to call an agreement.
  const PATHS: usize = 6 * M;
  let build = || Cfbms::<f32, _>::new(0.7, 0.4, N, Some(4.0), Deterministic::new(23));
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  assert_eq!(device[0][0].len(), N);
  assert_eq!(device[0][0][0], 0.0, "both rows start at zero");
  assert_eq!(device[0][1][0], 0.0, "both rows start at zero");
  assert!(
    device
      .iter()
      .all(|p| p.iter().all(|row| row.iter().all(|v| v.is_finite()))),
    "correlated fBM: a device path left the reals"
  );
  assert!(
    host
      .iter()
      .zip(device.iter())
      .any(|(h, d)| h[0].iter().zip(d[0].iter()).any(|(a, b)| a != b)),
    "{}: the device drew the host's own stream, so `on::<Device>()` did nothing",
    "correlated fBM"
  );
  let spread = |paths: &[[Array1<f32>; 2]], c: usize| {
    let last = paths[0][c].len() - 1;
    let n = paths.len() as f64;
    let mean = paths.iter().map(|p| p[c][last] as f64).sum::<f64>() / n;
    (paths
      .iter()
      .map(|p| (p[c][last] as f64 - mean).powi(2))
      .sum::<f64>()
      / n)
      .sqrt()
  };
  let corr = |paths: &[[Array1<f32>; 2]]| {
    let last = paths[0][0].len() - 1;
    let (mut sxy, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64);
    for p in paths {
      let (x, y) = (p[0][last] as f64, p[1][last] as f64);
      sxy += x * y;
      sxx += x * x;
      syy += y * y;
    }
    sxy / (sxx * syy).sqrt()
  };
  agrees(
    spread(&host, 0),
    spread(&device, 0),
    0.08,
    "correlated fBM first-row spread",
  );
  agrees(
    spread(&host, 1),
    spread(&device, 1),
    0.08,
    "correlated fBM second-row spread",
  );
  agrees(
    corr(&host),
    corr(&device),
    0.12,
    "correlated fBM terminal correlation",
  );
}

/// Correlated fGn is the one process whose output *is* the noise: every grid
/// point is a draw, so the frame steps before it writes the first point and
/// each point consumes one increment rather than `steps - 1` of them. The
/// statistics are taken over all points of all paths, since no single index
/// is more meaningful than another for a noise process.
#[test]
fn correlated_fgn_agrees_with_the_cpu_law() {
  let build = || Cfgns::<f32, _>::new(0.7, -0.4, N, Some(1.0), Deterministic::new(31));
  let device = build().on::<Device>().sample_par(M);
  let host = build().sample_par(M);
  assert_eq!(device.len(), M);
  assert_eq!(device[0][0].len(), N);
  assert!(
    device
      .iter()
      .all(|p| p.iter().all(|row| row.iter().all(|v| v.is_finite()))),
    "correlated fGn: a device path left the reals"
  );
  assert!(
    host
      .iter()
      .zip(device.iter())
      .any(|(h, d)| h[0].iter().zip(d[0].iter()).any(|(a, b)| a != b)),
    "{}: the device drew the host's own stream, so `on::<Device>()` did nothing",
    "correlated fGn"
  );
  let std = |paths: &[[Array1<f32>; 2]], c: usize| {
    let n = (paths.len() * paths[0][c].len()) as f64;
    let mean = paths
      .iter()
      .map(|p| p[c].iter().map(|v| *v as f64).sum::<f64>())
      .sum::<f64>()
      / n;
    (paths
      .iter()
      .map(|p| p[c].iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>())
      .sum::<f64>()
      / n)
      .sqrt()
  };
  let corr = |paths: &[[Array1<f32>; 2]]| {
    let (mut sxy, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64);
    for p in paths {
      for (a, b) in p[0].iter().zip(p[1].iter()) {
        let (x, y) = (*a as f64, *b as f64);
        sxy += x * y;
        sxx += x * x;
        syy += y * y;
      }
    }
    sxy / (sxx * syy).sqrt()
  };
  agrees(
    std(&host, 0),
    std(&device, 0),
    0.05,
    "correlated fGn first row",
  );
  agrees(
    std(&host, 1),
    std(&device, 1),
    0.05,
    "correlated fGn second row",
  );
  agrees(
    corr(&host),
    corr(&device),
    0.08,
    "correlated fGn correlation",
  );
}

/// The complex fOU reaches the engine through a two-component view of its own
/// real and imaginary rows, since the process itself reports one complex path.
/// Both parts mean-revert to zero, so the terminal spread of each is what
/// carries the law.
#[test]
fn complex_fou_agrees_with_the_cpu_law() {
  let build = || {
    Cfou::<f32, _>::new(
      0.7,
      2.0,
      1.5,
      0.6,
      N,
      Some(0.1),
      Some(-0.1),
      Some(1.0),
      Deterministic::new(17),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  let host = build().sample_par(M);
  assert_eq!(device.len(), M);
  assert_eq!(device[0].len(), N);
  assert_eq!(device[0][0].re, 0.1, "the real part starts at x1_0");
  assert_eq!(device[0][0].im, -0.1, "the imaginary part starts at x2_0");
  assert!(
    device
      .iter()
      .all(|p| p.iter().all(|z| z.re.is_finite() && z.im.is_finite())),
    "complex fOU: a device path left the reals"
  );
  let spread = |paths: &[Array1<num_complex::Complex<f32>>], im: bool| {
    let last = paths[0].len() - 1;
    let part = |z: &num_complex::Complex<f32>| if im { z.im as f64 } else { z.re as f64 };
    let n = paths.len() as f64;
    let mean = paths.iter().map(|p| part(&p[last])).sum::<f64>() / n;
    (paths
      .iter()
      .map(|p| (part(&p[last]) - mean).powi(2))
      .sum::<f64>()
      / n)
      .sqrt()
  };
  // The terminal mean is the only statistic that sees `omega`: the path
  // decays as `Z0 exp(-(lambda - i omega) T)`, so the rotation shows up as
  // the balance between the parts and nowhere in either spread. Comparing
  // the spreads alone passes with `omega` negated, or dropped entirely.
  let part_mean = |paths: &[Array1<num_complex::Complex<f32>>], im: bool| {
    let last = paths[0].len() - 1;
    let part = |z: &num_complex::Complex<f32>| if im { z.im as f64 } else { z.re as f64 };
    paths.iter().map(|p| part(&p[last])).sum::<f64>() / paths.len() as f64
  };
  for im in [false, true] {
    let (h, d) = (part_mean(&host, im), part_mean(&device, im));
    assert!(
      (h - d).abs() < 0.02,
      "complex fOU terminal mean ({}): host {h}, device {d}",
      if im { "imaginary" } else { "real" }
    );
  }
  agrees(
    spread(&host, false),
    spread(&device, false),
    0.08,
    "complex fOU real-part spread",
  );
  agrees(
    spread(&host, true),
    spread(&device, true),
    0.08,
    "complex fOU imaginary-part spread",
  );
}
