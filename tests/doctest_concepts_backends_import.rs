// docs: concepts/backends#the-backend-trait
//! Backs the always-available-marker import example on the backends
//! concept page.

#[cfg(feature = "metal")]
use stochastic_rs::stochastic::device::MetalNative;
use stochastic_rs::traits::Backend;
use stochastic_rs::traits::Cpu;

#[test]
fn cpu_marker_is_a_backend() {
  fn assert_backend<B: Backend>() {}
  assert_backend::<Cpu>();
  #[cfg(feature = "metal")]
  assert_backend::<MetalNative>();
}
