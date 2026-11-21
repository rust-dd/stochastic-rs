// docs: concepts/backends#the-backend-trait
//! Backs the always-available-marker import example on the backends concept page: every marker is a
//! [`Backend`] and carries [`FgnBackend`] for its scalar (`f64`/`f32` host, `f32` Apple GPU).

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
use stochastic_rs::stochastic::device::Metal;
use stochastic_rs::traits::Backend;
use stochastic_rs::traits::Cpu;
use stochastic_rs::traits::FgnBackend;

#[test]
fn cpu_marker_is_a_backend_with_the_fgn_capability() {
  fn assert_marker<B: Backend>() {}
  fn assert_fgn_f64<B: FgnBackend<f64>>() {}
  fn assert_fgn_f32<B: FgnBackend<f32>>() {}
  assert_marker::<Cpu>();
  assert_fgn_f64::<Cpu>();
  assert_fgn_f32::<Cpu>();
  #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
  assert_marker::<Metal>();
  #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
  assert_fgn_f32::<Metal>();
}
