//! Device laws of the processes whose jump sizes come from a user-supplied
//! distribution: the engine recognises the normal and exponential laws at
//! runtime and draws them in the kernel, and keeps anything else on the host.

use ndarray::Array1;
use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::scalar::ScalarExp;
use stochastic_rs_distributions::scalar::ScalarNormal;
use stochastic_rs_stochastic::jump::kou::Kou;
use stochastic_rs_stochastic::jump::levy_diffusion::LevyDiffusion;
use stochastic_rs_stochastic::jump::merton::Merton;
use stochastic_rs_stochastic::process::ccustom::CompoundCustom;
use stochastic_rs_stochastic::process::cpoisson::CompoundPoisson;
use stochastic_rs_stochastic::process::customjt::CustomJt;
use stochastic_rs_stochastic::process::poisson::Poisson;
use stochastic_rs_stochastic::process::subordinator::ctrw::Ctrw;
use stochastic_rs_stochastic::process::subordinator::ctrw::CtrwJumpLaw;
use stochastic_rs_stochastic::process::subordinator::ctrw::CtrwWaitingLaw;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::Device;
use super::common::M;
use super::common::agrees;
use super::common::all_finite;
use super::common::terminal_mean;
use super::common::terminal_std;

const N: usize = 253;

fn nondecreasing(paths: &[[Array1<f32>; 3]], what: &str) {
  assert!(
    paths
      .iter()
      .all(|p| p[0].windows(2).into_iter().all(|w| w[1] >= w[0])),
    "{what}: an arrival time went backwards"
  );
}

/// The second row is the running sum of the third, to within single-precision
/// rounding of the accumulation — what the kernel's two slots promise.
fn accumulates(paths: &[[Array1<f32>; 3]], what: &str) {
  for p in paths {
    assert_eq!(p[1][0], 0.0, "{what}: the cumulative row starts at zero");
    assert_eq!(p[2][0], 0.0, "{what}: the jump row starts at zero");
    for i in 1..p[1].len() {
      let residual = (p[1][i] - p[1][i - 1] - p[2][i]).abs();
      assert!(
        residual <= 1e-5 * p[1][i].abs().max(1.0),
        "{what}: cumulative row is not the running sum of the jumps at {i}"
      );
    }
  }
}

fn mean_of(paths: &[[Array1<f32>; 3]], row: usize) -> f64 {
  let last = paths[0][row].len() - 1;
  paths.iter().map(|p| p[row][last] as f64).sum::<f64>() / paths.len() as f64
}

fn std_of(paths: &[[Array1<f32>; 3]], row: usize, at: usize) -> f64 {
  let n = paths.len() as f64;
  let mean = paths.iter().map(|p| p[row][at] as f64).sum::<f64>() / n;
  (paths
    .iter()
    .map(|p| (p[row][at] as f64 - mean).powi(2))
    .sum::<f64>()
    / n)
    .sqrt()
}

#[test]
fn merton_agrees_with_the_cpu_law() {
  let build = || {
    Merton::<f32, _, _>::new(
      0.05,
      0.2,
      3.0,
      -0.02,
      ScalarNormal::<f32>::new(-0.02, 0.1),
      N,
      Some(1.0),
      Some(1.0),
      Deterministic::new(11),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device.len(), PATHS);
  assert_eq!(device[0].len(), N);
  assert_eq!(device[0][0], 1.0, "every path starts at x0");
  all_finite(&device, "Merton");
  agrees(
    terminal_mean(&host),
    terminal_mean(&device),
    0.02,
    "Merton terminal mean",
  );
  // The spread is where the jumps show: without them it would be the
  // diffusion's 0.2, with them near 0.27.
  agrees(
    terminal_std(&host),
    terminal_std(&device),
    0.05,
    "Merton terminal spread",
  );
}

/// Kou steps the same scheme under its own drift; the check is that its own
/// wiring reaches the kernel.
#[test]
fn kou_agrees_with_the_cpu_law() {
  let build = || {
    Kou::<f32, _, _>::new(
      0.05,
      0.2,
      3.0,
      0.01,
      ScalarNormal::<f32>::new(0.01, 0.12),
      N,
      Some(1.0),
      Some(1.0),
      Deterministic::new(13),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  all_finite(&device, "Kou");
  agrees(
    terminal_mean(&host),
    terminal_mean(&device),
    0.02,
    "Kou terminal mean",
  );
  agrees(
    terminal_std(&host),
    terminal_std(&device),
    0.05,
    "Kou terminal spread",
  );
}

/// Exponential sizes travel as the double-exponential law that only jumps
/// up; the drift and the one-sided jumps both push the terminal mean.
#[test]
fn levy_diffusion_agrees_with_the_cpu_law() {
  let build = || {
    LevyDiffusion::<f32, _, _>::new(
      0.1,
      0.2,
      2.0,
      ScalarExp::<f32>::new(8.0),
      N,
      Some(1.0),
      Some(1.0),
      Deterministic::new(19),
    )
  };
  const PATHS: usize = 4 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  all_finite(&device, "Lévy diffusion");
  assert!(
    terminal_mean(&device) > 1.25,
    "the exponential jumps did not lift the terminal mean: {}",
    terminal_mean(&device)
  );
  agrees(
    terminal_mean(&host),
    terminal_mean(&device),
    0.02,
    "Lévy diffusion terminal mean",
  );
  agrees(
    terminal_std(&host),
    terminal_std(&device),
    0.06,
    "Lévy diffusion terminal spread",
  );
}

#[test]
fn compound_poisson_events_agree_with_the_cpu_law() {
  const EVENTS: usize = 64;
  let build = || {
    CompoundPoisson::<f32, _, _>::new(
      ScalarNormal::<f32>::new(0.5, 0.2),
      Poisson::new(4.0, Some(EVENTS), None, Unseeded),
      Deterministic::new(17),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  let host = build().sample_par(M);
  assert_eq!(device.len(), M);
  assert!(
    device
      .iter()
      .all(|p| p.iter().all(|row| row.len() == EVENTS))
  );
  nondecreasing(&device, "compound Poisson");
  accumulates(&device, "compound Poisson");
  agrees(
    mean_of(&host, 0),
    mean_of(&device, 0),
    0.03,
    "compound Poisson last arrival time",
  );
  agrees(
    mean_of(&host, 1),
    mean_of(&device, 1),
    0.02,
    "compound Poisson cumulative jumps",
  );
  agrees(
    std_of(&host, 2, EVENTS / 2),
    std_of(&device, 2, EVENTS / 2),
    0.06,
    "compound Poisson jump size spread",
  );
}

/// Exponential inter-arrivals in count mode are Poisson arrivals, so the
/// device draws them as such.
#[test]
fn custom_jump_times_agree_with_the_cpu_law() {
  const EVENTS: usize = 50;
  let build = || {
    CustomJt::<f32, _, _>::new(
      Some(EVENTS),
      None,
      ScalarExp::<f32>::new(2.0),
      Deterministic::new(23),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  let host = build().sample_par(M);
  assert!(device.iter().all(|p| p.len() == EVENTS));
  assert!(
    device
      .iter()
      .all(|p| p[0] == 0.0 && p.windows(2).into_iter().all(|w| w[1] >= w[0])),
    "an arrival time went backwards"
  );
  agrees(
    terminal_mean(&host),
    terminal_mean(&device),
    0.03,
    "custom jump times last arrival",
  );
  agrees(
    terminal_std(&host),
    terminal_std(&device),
    0.06,
    "custom jump times last arrival spread",
  );
}

#[test]
fn compound_custom_agrees_with_the_cpu_law() {
  const EVENTS: usize = 50;
  let build = || {
    CompoundCustom::<f32, _, _, _>::new(
      Some(EVENTS),
      None,
      ScalarNormal::<f32>::new(1.0, 0.3),
      ScalarExp::<f32>::new(2.0),
      CustomJt::new(Some(EVENTS), None, ScalarExp::<f32>::new(2.0), Unseeded),
      Deterministic::new(29),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  let host = build().sample_par(M);
  nondecreasing(&device, "compound custom");
  accumulates(&device, "compound custom");
  agrees(
    mean_of(&host, 0),
    mean_of(&device, 0),
    0.03,
    "compound custom last arrival time",
  );
  agrees(
    mean_of(&host, 1),
    mean_of(&device, 1),
    0.02,
    "compound custom cumulative jumps",
  );
  agrees(
    std_of(&host, 2, EVENTS / 2),
    std_of(&device, 2, EVENTS / 2),
    0.06,
    "compound custom jump size spread",
  );
}

/// A jump law the kernels do not carry: every draw is one half.
struct Halves;

impl Distribution<f32> for Halves {
  fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> f32 {
    0.5
  }
}

/// A device build with a size law the kernels cannot draw samples on the
/// host, and is then the host build to the bit: the same seed, the same
/// sampler, the same chunking.
#[test]
fn an_unrecognised_jump_law_keeps_the_process_on_the_host() {
  let build = || {
    Merton::<f32, _, _>::new(
      0.05,
      0.2,
      3.0,
      0.5,
      Halves,
      N,
      Some(1.0),
      Some(1.0),
      Deterministic::new(31),
    )
  };
  let device = build().on::<Device>().sample_par(8);
  let host = build().sample_par(8);
  assert_eq!(
    device, host,
    "the host fallback diverged from the host build"
  );
  assert_eq!(build().on::<Device>().sample(), build().sample());
  let device = build()
    .on::<Device>()
    .try_sample_par(8)
    .expect("the host fallback cannot fail on the device's account");
  assert_eq!(device, host);
}

fn std_at(paths: &[Array1<f32>], at: usize) -> f64 {
  let n = paths.len() as f64;
  let mean = paths.iter().map(|p| p[at] as f64).sum::<f64>() / n;
  (paths
    .iter()
    .map(|p| (p[at] as f64 - mean).powi(2))
    .sum::<f64>()
    / n)
    .sqrt()
}

fn iqr_at(paths: &[Array1<f32>], at: usize) -> f64 {
  let mut values: Vec<f64> = paths.iter().map(|p| p[at] as f64).collect();
  values.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
  values[values.len() * 3 / 4] - values[values.len() / 4]
}

/// Exponential waits make the arrivals in a cell the Poisson count the
/// kernels draw, and normal sizes the law they aggregate; the mean grows as
/// `rate · mean · t` and the spread as `√(rate t (mean² + std²))`.
#[test]
fn ctrw_with_normal_jumps_agrees_with_the_cpu_law() {
  let build = || {
    Ctrw::<f32, _>::new(
      CtrwWaitingLaw::Exponential { rate: 20.0 },
      CtrwJumpLaw::Normal {
        mean: 0.1,
        std: 0.3,
      },
      N,
      Some(0.0),
      Some(1.0),
      Deterministic::new(173),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  assert_eq!(device[0][0], 0.0, "every path starts at x0");
  all_finite(&device, "CTRW normal");
  agrees(
    terminal_mean(&host),
    terminal_mean(&device),
    0.03,
    "CTRW normal terminal mean",
  );
  agrees(
    terminal_std(&host),
    terminal_std(&device),
    0.05,
    "CTRW normal terminal spread",
  );
}

/// Rademacher sizes are the one law the kernels sum by coin flips; a wrong
/// sign convention or a dropped flip would move the spread, `scale √(rate t)`,
/// while the mean stays at zero either way.
#[test]
fn ctrw_with_rademacher_jumps_agrees_with_the_cpu_law() {
  let build = || {
    Ctrw::<f32, _>::new(
      CtrwWaitingLaw::Exponential { rate: 20.0 },
      CtrwJumpLaw::Rademacher { scale: 0.5 },
      N,
      Some(0.0),
      Some(1.0),
      Deterministic::new(179),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  all_finite(&device, "CTRW Rademacher");
  let spread = terminal_std(&host);
  assert!(
    terminal_mean(&device).abs() < 0.05 * spread,
    "CTRW Rademacher terminal mean {} against a spread of {spread}",
    terminal_mean(&device)
  );
  agrees(
    spread,
    terminal_std(&device),
    0.05,
    "CTRW Rademacher terminal spread",
  );
  agrees(
    std_at(&host, N / 4),
    std_at(&device, N / 4),
    0.05,
    "CTRW Rademacher quarter-horizon spread",
  );
}

/// Symmetric stable sizes have no variance, so the spread is read off the
/// quartiles; the kernel draws a cell's sum in one Chambers–Mallows–Stuck
/// step at scale `scale · n^{1/α}`, the host draws each jump.
#[test]
fn ctrw_with_stable_jumps_agrees_with_the_cpu_law() {
  let build = || {
    Ctrw::<f32, _>::new(
      CtrwWaitingLaw::Exponential { rate: 20.0 },
      CtrwJumpLaw::SymmetricStable {
        alpha: 1.7,
        scale: 0.2,
      },
      N,
      Some(0.0),
      Some(1.0),
      Deterministic::new(181),
    )
  };
  const PATHS: usize = 3 * M;
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  all_finite(&device, "CTRW stable");
  agrees(
    iqr_at(&host, N - 1),
    iqr_at(&device, N - 1),
    0.06,
    "CTRW stable terminal interquartile range",
  );
  agrees(
    iqr_at(&host, N / 4),
    iqr_at(&device, N / 4),
    0.06,
    "CTRW stable quarter-horizon interquartile range",
  );
}

/// A waiting law without the memoryless property has no per-cell Poisson
/// count, so the device build samples on the host and is the host build to
/// the bit.
#[test]
fn ctrw_with_gamma_waits_keeps_the_process_on_the_host() {
  let build = || {
    Ctrw::<f32, _>::new(
      CtrwWaitingLaw::Gamma {
        shape: 2.0,
        rate: 10.0,
      },
      CtrwJumpLaw::Normal {
        mean: 0.1,
        std: 0.3,
      },
      N,
      Some(0.0),
      Some(1.0),
      Deterministic::new(191),
    )
  };
  assert_eq!(build().on::<Device>().sample_par(8), build().sample_par(8));
}
