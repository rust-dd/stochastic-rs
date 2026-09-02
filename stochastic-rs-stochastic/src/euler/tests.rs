use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;

use super::*;
use crate::diffusion::cir::Cir;
use crate::diffusion::gbm::Gbm;
use crate::diffusion::ou::Ou;

/// The CPU back-end is the process's own sampler: bit for bit what
/// `sample_par` returns with the seed pinned through `Deterministic`.
#[test]
fn cpu_backend_is_the_process_sampler() {
  let gbm = Gbm::new(0.05, 0.2, 64, Some(100.0), Some(1.0), Unseeded);
  let engine = sample_paths::<f64, Cpu, _>(&gbm, 16, 42);
  let own = gbm.seeded(42).sample_par(16);
  assert_eq!(engine.dim(), (16, 64));
  for (i, path) in own.iter().enumerate() {
    assert_eq!(engine.row(i).to_vec(), path.to_vec());
  }
  let direct = Gbm::new(
    0.05,
    0.2,
    64,
    Some(100.0),
    Some(1.0),
    Deterministic::new(42),
  )
  .sample_par(16);
  assert_eq!(engine.row(3).to_vec(), direct[3].to_vec());
  let ou = Ou::new(2.0, 1.0, 0.5, 33, Some(1.0), Some(2.0), Unseeded);
  assert_eq!(
    sample_paths::<f64, Cpu, _>(&ou, 4, 7).row(2).to_vec(),
    ou.seeded(7).sample_par(4)[2].to_vec()
  );
  let cir = Cir::new(1.5, 0.04, 0.3, 33, Some(0.09), Some(1.0), None, Unseeded);
  assert_eq!(
    sample_paths::<f64, Cpu, _>(&cir, 4, 7).row(1).to_vec(),
    cir.seeded(7).sample_par(4)[1].to_vec()
  );
  assert_eq!(sample_paths::<f64, Cpu, _>(&gbm, 0, 1).dim(), (0, 64));
}

#[test]
fn cpu_backend_seeds_are_reproducible_and_discriminating() {
  let gbm = Gbm::new(0.05, 0.2, 64, Some(100.0), Some(1.0), Unseeded);
  let a = sample_paths::<f64, Cpu, _>(&gbm, 16, 42);
  let b = sample_paths::<f64, Cpu, _>(&gbm, 16, 42);
  let c = sample_paths::<f64, Cpu, _>(&gbm, 16, 43);
  assert_eq!(a, b);
  assert_ne!(a, c);
  assert_ne!(a.row(0), a.row(1));
}

#[test]
fn spec_encoding_carries_the_family_and_its_parameters() {
  assert_eq!(
    EulerSpec::GeometricBrownian {
      mu: 0.1,
      sigma: 0.3
    }
    .encode(),
    (0, [0.1, 0.3, 0.0, 0.0])
  );
  assert_eq!(
    EulerSpec::OrnsteinUhlenbeck {
      theta: 2.0,
      mu: 1.0,
      sigma: 0.5
    }
    .encode(),
    (1, [2.0, 1.0, 0.5, 0.0])
  );
  assert_eq!(
    EulerSpec::SquareRoot {
      kappa: 1.0,
      theta: 2.0,
      sigma: 3.0
    }
    .encode(),
    (2, [1.0, 2.0, 3.0, 0.0])
  );
  let cir = Cir::new(1.5, 0.04, 0.3, 10, Some(0.09), Some(2.0), None, Unseeded);
  assert_eq!(cir.euler_spec().encode().0, 2);
  assert_eq!(
    (cir.initial_value(), cir.grid_points(), cir.horizon()),
    (0.09, 10, 2.0)
  );
}

/// Every compiled device back-end reproduces the lognormal moments and is
/// deterministic in its seed; the device kernels share one integer hash for
/// their uniforms, so two device back-ends agree seed for seed up to the
/// `f32` libm rounding of Box–Muller.
#[cfg(any(
  feature = "metal",
  feature = "cuda-native",
  feature = "gpu-cuda",
  feature = "gpu-wgpu"
))]
mod devices {
  use super::*;

  fn column_mean_var(paths: &Array2<f64>, col: usize) -> (f64, f64) {
    let column = paths.column(col);
    let m = column.len() as f64;
    let mean = column.sum() / m;
    let var = column.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (m - 1.0);
    (mean, var)
  }

  fn gbm_moments_hold<B: EulerBackend>(label: &str) -> Array2<f64> {
    let gbm = Gbm::new(0.05, 0.2, 253, Some(100.0), Some(1.0), Unseeded);
    let paths = sample_paths::<f64, B, _>(&gbm, 40_000, 7);
    assert_eq!(paths.dim(), (40_000, 253), "{label}");
    let (mean, var) = column_mean_var(&paths, 252);
    let expected_mean = 100.0 * 0.05_f64.exp();
    let expected_var = 100.0_f64.powi(2) * (0.1_f64).exp() * ((0.04_f64).exp() - 1.0);
    assert!(
      (mean / expected_mean - 1.0).abs() < 0.01,
      "{label}: mean {mean} vs {expected_mean}"
    );
    assert!(
      (var / expected_var - 1.0).abs() < 0.06,
      "{label}: var {var} vs {expected_var}"
    );
    assert_eq!(
      sample_paths::<f64, B, _>(&gbm, 8, 7),
      sample_paths::<f64, B, _>(&gbm, 8, 7),
      "{label}: seed reproducibility"
    );
    assert_ne!(
      sample_paths::<f64, B, _>(&gbm, 8, 7),
      sample_paths::<f64, B, _>(&gbm, 8, 8),
      "{label}: seed discrimination"
    );
    paths
  }

  fn cir_stays_nonnegative<B: EulerBackend>(label: &str) {
    let cir = Cir::new(1.5, 0.04, 0.3, 253, Some(0.09), Some(1.0), None, Unseeded);
    let paths = sample_paths::<f64, B, _>(&cir, 4_000, 3);
    let (mean, _) = column_mean_var(&paths, 252);
    let expected = 0.04 + (0.09 - 0.04) * (-1.5_f64).exp();
    assert!(
      (mean / expected - 1.0).abs() < 0.03,
      "{label}: mean {mean} vs {expected}"
    );
    assert!(paths.iter().all(|&x| x >= 0.0), "{label}");
  }

  #[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu"))]
  #[test]
  fn cubecl_backend_matches_the_moments() {
    gbm_moments_hold::<crate::device::CubeCl>("CubeCl");
    cir_stays_nonnegative::<crate::device::CubeCl>("CubeCl");
  }

  #[cfg(feature = "metal")]
  #[test]
  fn metal_native_backend_matches_the_moments() {
    gbm_moments_hold::<crate::device::MetalNative>("MetalNative");
    cir_stays_nonnegative::<crate::device::MetalNative>("MetalNative");
  }

  #[cfg(feature = "cuda-native")]
  #[test]
  fn cuda_native_backend_matches_the_moments() {
    gbm_moments_hold::<crate::device::CudaNative>("CudaNative");
    cir_stays_nonnegative::<crate::device::CudaNative>("CudaNative");
    // f64 kernel: the double-precision path agrees with the f32 one to float rounding.
    let gbm32 = Gbm::<f32, _>::new(0.05, 0.2, 64, Some(100.0), Some(1.0), Unseeded);
    let gbm64 = Gbm::<f64, _>::new(0.05, 0.2, 64, Some(100.0), Some(1.0), Unseeded);
    let single = sample_paths::<f32, crate::device::CudaNative, _>(&gbm32, 4, 5);
    let double = sample_paths::<f64, crate::device::CudaNative, _>(&gbm64, 4, 5);
    for (a, b) in single.iter().zip(double.iter()) {
      assert!(((*a as f64) - b).abs() < 1e-3 * b.abs().max(1.0));
    }
  }

  #[cfg(all(feature = "metal", any(feature = "gpu-cuda", feature = "gpu-wgpu")))]
  #[test]
  fn metal_native_and_cubecl_agree_seed_for_seed() {
    let gbm = Gbm::new(0.05, 0.2, 128, Some(100.0), Some(1.0), Unseeded);
    let metal = sample_paths::<f64, crate::device::MetalNative, _>(&gbm, 16, 11);
    let cubecl = sample_paths::<f64, crate::device::CubeCl, _>(&gbm, 16, 11);
    for (a, b) in metal.iter().zip(cubecl.iter()) {
      assert!((a - b).abs() < 1e-3 * b.abs().max(1.0), "{a} vs {b}");
    }
  }
}
