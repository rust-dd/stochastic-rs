//! The kernels draw a step's Poisson count up to 64 events, exact to `f32` up to 25 expected: a
//! denser count samples on the host, the Cpu build to the bit, and a sparser one keeps the law.

use ndarray::Array1;
use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::jump::merton::Merton;
use stochastic_rs_stochastic::process::subordinator::ctrw::Ctrw;
use stochastic_rs_stochastic::process::subordinator::ctrw::CtrwJumpLaw;
use stochastic_rs_stochastic::process::subordinator::ctrw::CtrwWaitingLaw;
use stochastic_rs_stochastic::process::subordinator::tempered_stable::TemperedStableSubordinator;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::bns::Bns;

use super::common::Device;
use super::common::M;
use super::common::means_agree;
use super::common::spreads_agree;

/// 64 steps of `dt = 1/64`, so a rate of `64·k` expects `k` events a step.
const N: usize = 65;

fn merton(rate: f32, seed: u64) -> Merton<f32, SimdNormal<f32>, Deterministic> {
  Merton::new(
    0.05,
    0.2,
    rate,
    0.0,
    SimdNormal::<f32>::new(0.001, 0.01),
    N,
    Some(1.0),
    Some(1.0),
    Deterministic::new(seed),
  )
}

/// Every sampling form of a device-backed build at a hundred expected events a step is the Cpu
/// build's, bit for bit.
fn host_to_the_bit<P, C>(device: impl Fn() -> P, cpu: impl Fn() -> C, what: &str)
where
  P: ProcessExt<f32>,
  C: ProcessExt<f32, Output = P::Output>,
  P::Output: Clone + PartialEq + Send + std::fmt::Debug,
{
  assert!(!device().device_ready(), "{what}: a dense count is ready");
  assert_eq!(device().sample(), cpu().sample(), "{what}: sample");
  assert_eq!(
    device().try_sample().unwrap(),
    cpu().sample(),
    "{what}: try_sample"
  );
  assert_eq!(
    device().sample_par(8),
    cpu().sample_par(8),
    "{what}: sample_par"
  );
  assert_eq!(
    device().try_sample_par(8).unwrap(),
    cpu().sample_par(8),
    "{what}: try_sample_par"
  );
  assert_eq!(
    device().sample_map(8, |path| path.clone()),
    cpu().sample_map(8, |path| path.clone()),
    "{what}: sample_map"
  );
}

/// [`host_to_the_bit`]'s borrowing form, for a process whose output is one path.
fn view_to_the_bit<P, C>(device: impl Fn() -> P, cpu: impl Fn() -> C, what: &str)
where
  P: ProcessExt<f32, Output = Array1<f32>>,
  C: ProcessExt<f32, Output = Array1<f32>>,
{
  let path = |path: ArrayView1<f32>| path.to_owned();
  assert_eq!(
    device().sample_map_view(8, path),
    cpu().sample_map_view(8, path),
    "{what}: sample_map_view"
  );
}

#[test]
fn a_hundred_expected_jumps_a_step_sample_on_the_host() {
  host_to_the_bit(
    || merton(6400.0, 29).on::<Device>(),
    || merton(6400.0, 29),
    "Merton",
  );
  view_to_the_bit(
    || merton(6400.0, 29).on::<Device>(),
    || merton(6400.0, 29),
    "Merton",
  );
  let stable = || {
    TemperedStableSubordinator::<f32, _>::new(
      0.5,
      32.0,
      1.0,
      1e-4,
      N,
      None,
      Some(1.0),
      Deterministic::new(31),
    )
  };
  host_to_the_bit(
    || stable().on::<Device>(),
    stable,
    "TemperedStableSubordinator",
  );
  view_to_the_bit(
    || stable().on::<Device>(),
    stable,
    "TemperedStableSubordinator",
  );
  let bns = || {
    Bns::<f32, _>::new(
      100.0,
      0.04,
      6400.0,
      0.05,
      1.0,
      0.0001,
      N,
      Some(1.0),
      Deterministic::new(37),
    )
  };
  host_to_the_bit(|| bns().on::<Device>(), bns, "Bns");
}

/// Twenty expected jumps a step stay on the device and keep the host's terminal law.
#[test]
fn twenty_expected_jumps_a_step_keep_the_law_on_the_device() {
  assert!(merton(1280.0, 41).on::<Device>().device_ready());
  let device = merton(1280.0, 41).on::<Device>().sample_par(M);
  let host = merton(1280.0, 41).sample_par(M);
  means_agree(
    &host,
    &device,
    "Merton at twenty jumps a step, terminal mean",
  );
  spreads_agree(
    &host,
    &device,
    "Merton at twenty jumps a step, terminal spread",
  );
}

/// A hundred expected events per step is past the kernels' 64-event count, so the device build
/// samples on the host and is the Cpu build to the bit.
#[test]
fn a_dense_ctrw_keeps_to_the_host_on_any_backend() {
  let build = || {
    Ctrw::<f32, _>::new(
      CtrwWaitingLaw::Exponential { rate: 6400.0 },
      CtrwJumpLaw::Normal {
        mean: 1.0,
        std: 0.01,
      },
      65,
      Some(0.0),
      Some(1.0),
      Deterministic::new(179),
    )
  };
  assert!(!build().on::<Device>().device_ready());
  assert_eq!(build().on::<Device>().sample_par(4), build().sample_par(4));
  assert_eq!(build().on::<Device>().sample(), build().sample());
}
