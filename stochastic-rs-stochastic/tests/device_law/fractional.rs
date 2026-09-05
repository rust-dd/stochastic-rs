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
use stochastic_rs_stochastic::process::volterra::Volterra;
use stochastic_rs_stochastic::process::volterra::VolterraKernelSpec;
use stochastic_rs_stochastic::rough::rl_bs::RlBlackScholes;
use stochastic_rs_stochastic::rough::rl_fbm::RlFBm;
use stochastic_rs_stochastic::rough::rl_fou::RlFOU;
use stochastic_rs_stochastic::rough::rl_heston::RlHeston;
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

/// Riemann-Liouville fBm through the Markov lift, the first family whose
/// state lives outside the four slots: the frame carries the lift's node
/// states per path. Horizon 4 rather than 1 so the terminal spread `t^H`
/// (1.52 at `H = 0.3`) is told apart from Brownian motion's `sqrt(t)` (2.00) —
/// a launch that dropped the history sum and kept only the boundary term
/// would look Brownian.
#[test]
fn rl_fbm_agrees_with_the_cpu_law() {
  let build = || RlFBm::<f32, _>::new(0.3, N, Some(4.0), None, Deterministic::new(29));
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
  // The spread a quarter of the way in pins the roughness, not just the
  // scale: `(t/4)^H / t^H = 4^-H` is 0.66 at `H = 0.3` and 0.5 for Brownian
  // motion.
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
    RlFOU::<f32, _>::new(
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

/// The rough Heston puts the lift on the variance and correlates its shock
/// with the spot's. The variance's terminal mean pins `kappa` and `theta`, its
/// spread `nu` and the roughness, and the spot/variance correlation `rho` —
/// the one statistic a launch that dropped the correlation keeps every
/// marginal of.
#[test]
fn rl_heston_agrees_with_the_cpu_law() {
  let build = || {
    RlHeston::<f32, _>::new(
      0.3,
      Some(100.0),
      Some(0.04),
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

/// The general Volterra process's lift branch is fBm under the Markov lift —
/// the family `RlFBm` rides — while a kernel outside the rough range takes the
/// reference convolution on the host whatever the backend.
#[test]
fn volterra_lift_agrees_with_the_cpu_law_and_the_reference_still_samples() {
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
  let reference = Volterra::<f32, _>::new(
    VolterraKernelSpec::FractionalBM { h: 0.7 },
    64,
    Some(1.0),
    Deterministic::new(53),
  )
  .on::<Device>()
  .sample_par(8);
  assert_eq!(reference.len(), 8);
  assert!(
    reference
      .iter()
      .all(|p| p.len() == 64 && p.iter().all(|v| v.is_finite()))
  );
}
