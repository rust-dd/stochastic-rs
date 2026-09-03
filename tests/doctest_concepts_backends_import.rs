// docs: concepts/backends#the-backend-trait
//! Backs the always-available-marker import example on the backends
//! concept page: every marker is a [`Backend`] (the bare device marker),
//! and every marker shipped today also carries the [`FgnBackend`] fGN
//! sampling capability.

#[cfg(feature = "metal")]
use stochastic_rs::stochastic::device::MetalNative;
use stochastic_rs::traits::Backend;
use stochastic_rs::traits::Cpu;
use stochastic_rs::traits::FgnBackend;

#[test]
fn cpu_marker_is_a_backend_with_the_fgn_capability() {
  fn assert_marker<B: Backend>() {}
  fn assert_fgn<B: FgnBackend<f64>>() {}
  assert_marker::<Cpu>();
  assert_fgn::<Cpu>();
  #[cfg(feature = "metal")]
  assert_marker::<MetalNative>();
  #[cfg(feature = "metal")]
  assert_fgn::<MetalNative>();
}
