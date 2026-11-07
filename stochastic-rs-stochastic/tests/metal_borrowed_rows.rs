//! Nested launches preserve rows borrowed from the shared Metal context.

#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use ndarray::{Array1, ArrayView1};
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::{
  device::Metal, noise::fgn::Fgn, process::fbm::Fbm, traits::ProcessExt,
};

fn bits(rows: Vec<Array1<f32>>) -> Vec<Array1<u32>> {
  rows.into_iter().map(|row| row.mapv(f32::to_bits)).collect()
}

#[test]
fn nested_sampling_preserves_borrowed_noise_and_euler_rows() {
  rayon::ThreadPoolBuilder::new()
    .num_threads(1)
    .build()
    .unwrap()
    .install(|| {
      let owner = std::thread::current().id();
      let noise = || Fgn::new(0.7_f32, 512, Some(1.0), Deterministic::new(37)).on::<Metal>();
      let euler = || Fbm::new(0.7_f32, 512, Some(1.0), Deterministic::new(41)).on::<Metal>();
      let nested = || Fbm::new(0.7_f32, 512, Some(1.0), Deterministic::new(91)).on::<Metal>();
      let expected_nested = bits(nested().sample_par(3));
      let consume = |row: ArrayView1<'_, f32>| {
        assert_eq!(std::thread::current().id(), owner);
        let before = row.mapv(f32::to_bits);
        assert_eq!(bits(nested().sample_par(3)), expected_nested);
        assert_eq!(row.mapv(f32::to_bits), before);
        before
      };
      let expected_noise = bits(noise().sample_par(5));
      assert_eq!(noise().sample_map_view(5, consume), expected_noise);
      let expected_euler = bits(euler().sample_par(5));
      assert_eq!(euler().sample_map_view(5, consume), expected_euler);
    });
}
