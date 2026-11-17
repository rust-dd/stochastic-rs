//! What a process will say about where it runs, before it runs there.
//!
//! Two questions a caller has no other way to answer. Which handle is this
//! process holding — the type parameter records the *marker*, not the
//! ordinal or the batch budget the handle carries — and will this
//! configuration reach a kernel at all, or fall back to the host sampler at
//! an order of magnitude's cost. The second one used to be a bare `bool`;
//! the reason is what makes it actionable.

use ndarray::Array2;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::autoregressive::ar::Ar;
use stochastic_rs_stochastic::device::Cpu;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::diffusion::wishart::Wishart;
use stochastic_rs_stochastic::interest::black_karasinski::BlackKarasinski;
use stochastic_rs_stochastic::interest::vasicek::Vasicek;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::process::subordinator::ctrw::Ctrw;
use stochastic_rs_stochastic::process::subordinator::ctrw::CtrwJumpLaw;
use stochastic_rs_stochastic::process::subordinator::ctrw::CtrwWaitingLaw;
use stochastic_rs_stochastic::traits::ProcessExt;

/// A process hands back the handle it holds, and the handle opens.
#[test]
fn a_process_reports_the_backend_it_holds() {
  let gbm = Gbm::<f64, _>::new(0.05, 0.2, 256, 100.0, None, Deterministic::new(1));
  assert_eq!(*gbm.backend(), Cpu);
  let info = gbm.probe().expect("the CPU device always opens");
  assert_eq!(info.backend, "Cpu");
  assert!(info.precisions.contains(&"f64"));
}

/// The fractional processes carry their backend inside the noise pipeline
/// rather than in a field of their own; the accessor is the same either way,
/// which is the point of having one.
#[test]
fn a_fractional_process_reports_its_backend_too() {
  let fgn = Fgn::<f64, _>::new(0.7, 128, Some(1.0), Deterministic::new(2));
  assert_eq!(*fgn.backend(), Cpu);
  assert_eq!(fgn.probe().expect("cpu").backend, "Cpu");
}

/// A configuration the kernels carry whole says nothing; one they cannot
/// says why. The reason travels with the process, so a caller can print it
/// rather than discovering the fallback in a profile.
#[test]
fn a_configuration_past_the_kernels_names_its_reason() {
  let first = Ar::<f64, _>::new(array![0.5], 1.0, 128, None, Deterministic::new(3));
  assert!(first.device_ready());
  assert_eq!(first.device_fallback(), None);

  let second = Ar::<f64, _>::new(array![0.5, 0.2], 1.0, 128, None, Deterministic::new(3));
  assert!(!second.device_ready());
  let why = second
    .device_fallback()
    .expect("an order above one has a reason");
  assert!(
    why.contains("order above one"),
    "the reason should name the order, got {why:?}"
  );
}

/// The two answers cannot disagree: `device_ready` is defined as the absence
/// of a reason, and every override states the reason rather than the bool.
#[test]
fn readiness_is_exactly_the_absence_of_a_reason() {
  let identity = Array2::<f64>::eye(3);
  let wide = Wishart::<f64, _>::new(
    4.0,
    identity.clone(),
    identity.clone(),
    identity,
    64,
    Some(1.0),
    Deterministic::new(5),
  );
  assert_eq!(wide.device_ready(), wide.device_fallback().is_none());
  assert!(
    wide.device_fallback().is_some(),
    "a three-dimensional Wishart is past the kernel's two"
  );

  let plain = Vasicek::<f64, _>::new(1.5, 0.04, 0.3, 128, None, None, Deterministic::new(7));
  assert_eq!(plain.device_ready(), plain.device_fallback().is_none());
  assert_eq!(plain.device_fallback(), None);
}

fn flat(_t: f64) -> f64 {
  0.05
}

/// A Black–Karasinski reversion of exactly zero reaches the kernels, which step its limit.
#[test]
fn a_zero_reversion_black_karasinski_reaches_the_device() {
  let build = |a: f64| {
    BlackKarasinski::<f64, _>::new(
      flat as fn(f64) -> f64,
      a,
      0.1,
      64,
      Some(0.03),
      Some(1.0),
      Deterministic::new(11),
    )
  };
  assert!(build(0.0).device_ready() && build(0.0).device_fallback().is_none());
  assert!(build(0.1).device_ready());
}

/// The kernels' per-step count stops at 64 events, so a Ctrw expecting over 25 per step samples
/// on the host, where every event is drawn.
#[test]
fn a_dense_ctrw_keeps_to_the_host() {
  let build = |rate: f64, n: usize| {
    Ctrw::<f64, _>::new(
      CtrwWaitingLaw::Exponential { rate },
      CtrwJumpLaw::Normal {
        mean: 1.0,
        std: 1e-6,
      },
      n,
      None,
      Some(1.0),
      Deterministic::new(2),
    )
  };
  assert!(!build(4.0e6, 2).device_ready());
  assert!(
    build(4.0e6, 2)
      .device_fallback()
      .is_some_and(|why| why.contains("64")),
    "the reason should name the kernels' count limit"
  );
  assert!(build(2.0, 512).device_ready());
}

/// A setter that rebuilds no cache is defined for every backend, so a process can be adjusted after
/// it has moved to a device.
#[test]
fn a_setter_is_defined_for_every_backend() {
  fn retune<B>(c: Cir<f64, Unseeded, B>) -> Cir<f64, Unseeded, B> {
    c.with_sigma(0.3).with_points(128)
  }
  let c = retune(Cir::<f64>::default());
  assert_eq!((c.sigma, c.n), (0.3, 128));
}

#[cfg(metal_backend)]
#[test]
fn a_device_backed_process_takes_its_setters() {
  use stochastic_rs_stochastic::device::Metal;
  let c = Cir::<f32>::default()
    .on::<Metal>()
    .with_sigma(0.3)
    .with_points(128);
  assert_eq!((c.sigma, c.n), (0.3, 128));
}

/// One macro writes both device handles: the same constructors, the same one-byte budget floor.
#[cfg(device_backend)]
macro_rules! handle_surface {
  ($test:ident, $handle:ident) => {
    #[test]
    fn $test() {
      use stochastic_rs_stochastic::device::{DEFAULT_BATCH_BUDGET_BYTES, DeviceError, $handle};
      let h = $handle::new(0).with_batch_budget(1 << 20);
      assert_eq!((h.ordinal, h.batch_budget), (0, 1 << 20));
      assert_eq!($handle::default().batch_budget, DEFAULT_BATCH_BUDGET_BYTES);
      assert_eq!($handle::new(1).with_batch_budget(0).batch_budget, 1);
      let _: &dyn Fn() -> Result<$handle, DeviceError> = &$handle::from_env;
    }
  };
}

#[cfg(metal_backend)]
handle_surface!(
  the_metal_handle_has_the_same_surface_as_the_cuda_handle,
  Metal
);

#[cfg(feature = "cuda")]
handle_surface!(
  the_cuda_handle_has_the_same_surface_as_the_metal_handle,
  Cuda
);

/// A release frees caches, not a seed's stream: two identically built processes, one sampled before
/// the release and one after, draw the same paths.
#[cfg(metal_backend)]
#[test]
fn a_released_metal_cache_rebuilds_on_the_next_launch() {
  use stochastic_rs_stochastic::device::{Metal, release_host_scratch};

  let gbm = || {
    Gbm::<f32, Deterministic>::new(0.05, 0.2, 64, 100.0, Some(1.0), Deterministic::new(1))
      .on::<Metal>()
  };
  let before = gbm().sample_par(8);
  Metal::default().release();
  release_host_scratch();
  assert_eq!(before, gbm().sample_par(8));
}
