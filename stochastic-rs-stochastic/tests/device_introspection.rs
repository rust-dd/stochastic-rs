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
use stochastic_rs_stochastic::autoregressive::ar::ARp;
use stochastic_rs_stochastic::device::Cpu;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::diffusion::wishart::Wishart;
use stochastic_rs_stochastic::interest::vasicek::Vasicek;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::traits::ProcessExt;

/// A process hands back the handle it holds, and the handle opens.
#[test]
fn a_process_reports_the_backend_it_holds() {
  let gbm = Gbm::<f64, _>::new(0.05, 0.2, 256, Some(100.0), None, Deterministic::new(1));
  assert_eq!(gbm.backend(), Cpu);
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
  assert_eq!(fgn.backend(), Cpu);
  assert_eq!(fgn.probe().expect("cpu").backend, "Cpu");
}

/// A configuration the kernels carry whole says nothing; one they cannot
/// says why. The reason travels with the process, so a caller can print it
/// rather than discovering the fallback in a profile.
#[test]
fn a_configuration_past_the_kernels_names_its_reason() {
  let first = ARp::<f64, _>::new(array![0.5], 1.0, 128, None, Deterministic::new(3));
  assert!(first.device_ready());
  assert_eq!(first.device_fallback(), None);

  let second = ARp::<f64, _>::new(array![0.5, 0.2], 1.0, 128, None, Deterministic::new(3));
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
