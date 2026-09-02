use super::*;
use crate::diffusion::cir::Cir;
use crate::diffusion::gbm::Gbm;
use crate::diffusion::ou::Ou;

fn column_mean_var(paths: &Array2<f64>, col: usize) -> (f64, f64) {
  let column = paths.column(col);
  let m = column.len() as f64;
  let mean = column.sum() / m;
  let var = column.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (m - 1.0);
  (mean, var)
}

/// `E[X_T] = x₀e^{μT}`, `Var[X_T] = x₀²e^{2μT}(e^{σ²T} − 1)` for geometric
/// Brownian motion; the Euler bias at Δt = 1/252 is far below the tolerances.
#[test]
fn cpu_gbm_paths_match_the_lognormal_moments() {
  let gbm = Gbm::new(
    0.05,
    0.2,
    253,
    Some(100.0),
    Some(1.0),
    stochastic_rs_core::simd_rng::Unseeded,
  );
  let paths = sample_paths::<f64, Cpu, _>(&gbm, 40_000, 7);
  assert_eq!(paths.dim(), (40_000, 253));
  assert!(paths.column(0).iter().all(|&x| x == 100.0));
  let (mean, var) = column_mean_var(&paths, 252);
  let expected_mean = 100.0 * 0.05_f64.exp();
  let expected_var = 100.0_f64.powi(2) * (0.1_f64).exp() * ((0.04_f64).exp() - 1.0);
  assert!(
    (mean / expected_mean - 1.0).abs() < 0.01,
    "mean {mean} vs {expected_mean}"
  );
  assert!(
    (var / expected_var - 1.0).abs() < 0.06,
    "var {var} vs {expected_var}"
  );
}

/// Ornstein–Uhlenbeck from `x₀ = μ`: the mean stays at `μ` and the variance
/// follows `σ²(1 − e^{−2θT})/(2θ)`.
#[test]
fn cpu_ou_paths_match_the_gaussian_moments() {
  let ou = Ou::new(
    2.0,
    1.0,
    0.5,
    501,
    Some(1.0),
    Some(2.0),
    stochastic_rs_core::simd_rng::Unseeded,
  );
  let paths = sample_paths::<f64, Cpu, _>(&ou, 40_000, 11);
  let (mean, var) = column_mean_var(&paths, 500);
  let expected_var = 0.25 * (1.0 - (-8.0_f64).exp()) / 4.0;
  assert!((mean - 1.0).abs() < 0.01, "mean {mean}");
  assert!(
    (var / expected_var - 1.0).abs() < 0.06,
    "var {var} vs {expected_var}"
  );
}

/// CIR: `E[X_T] = θ + (x₀ − θ)e^{−κT}` (the full-truncation bias at Δt = 1/252
/// is well inside the tolerance), and the reported path is the positive part.
#[test]
fn cpu_cir_paths_match_the_mean_and_stay_nonnegative() {
  let cir = Cir::new(
    1.5,
    0.04,
    0.3,
    253,
    Some(0.09),
    Some(1.0),
    None,
    stochastic_rs_core::simd_rng::Unseeded,
  );
  let paths = sample_paths::<f64, Cpu, _>(&cir, 40_000, 3);
  let (mean, _) = column_mean_var(&paths, 252);
  let expected = 0.04 + (0.09 - 0.04) * (-1.5_f64).exp();
  assert!(
    (mean / expected - 1.0).abs() < 0.02,
    "mean {mean} vs {expected}"
  );
  assert!(paths.iter().all(|&x| x >= 0.0));
}

#[test]
fn seeds_are_reproducible_and_discriminating() {
  let gbm = Gbm::new(
    0.05,
    0.2,
    64,
    Some(100.0),
    Some(1.0),
    stochastic_rs_core::simd_rng::Unseeded,
  );
  let a = sample_paths::<f64, Cpu, _>(&gbm, 16, 42);
  let b = sample_paths::<f64, Cpu, _>(&gbm, 16, 42);
  let c = sample_paths::<f64, Cpu, _>(&gbm, 16, 43);
  assert_eq!(a, b);
  assert_ne!(a, c);
  // Paths within one run are distinct streams, not copies of each other.
  assert_ne!(a.row(0), a.row(1));
}

#[test]
fn spec_step_matches_the_closed_form_increments() {
  let spec = EulerSpec::GeometricBrownian {
    mu: 0.1,
    sigma: 0.3,
  };
  assert!(
    (spec.step(2.0, 0.5, 0.5_f64.sqrt(), 1.0)
      - (2.0 + 0.1 * 2.0 * 0.5 + 0.3 * 2.0 * 0.5_f64.sqrt()))
    .abs()
      < 1e-15
  );
  let (code, p) = EulerSpec::SquareRoot {
    kappa: 1.0,
    theta: 2.0,
    sigma: 3.0,
  }
  .encode();
  assert_eq!((code, p), (2, [1.0, 2.0, 3.0, 0.0]));
  // Full truncation: a negative auxiliary state contributes no diffusion,
  // reverts upward and is reported as zero.
  let cir = EulerSpec::SquareRoot {
    kappa: 1.0,
    theta: 0.04,
    sigma: 0.3,
  };
  assert_eq!(cir.step(-0.01, 0.1, 0.1_f64.sqrt(), 5.0), -0.01 + 0.004);
  assert_eq!(cir.observed(-0.01), 0.0);
  assert_eq!(cir.observed(0.02), 0.02);
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

  fn gbm_moments_hold<B: EulerBackend>(label: &str) -> Array2<f64> {
    let gbm = Gbm::new(
      0.05,
      0.2,
      253,
      Some(100.0),
      Some(1.0),
      stochastic_rs_core::simd_rng::Unseeded,
    );
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
    let cir = Cir::new(
      1.5,
      0.04,
      0.3,
      253,
      Some(0.09),
      Some(1.0),
      None,
      stochastic_rs_core::simd_rng::Unseeded,
    );
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
    let gbm32 = Gbm::<f32, _>::new(
      0.05,
      0.2,
      64,
      Some(100.0),
      Some(1.0),
      stochastic_rs_core::simd_rng::Unseeded,
    );
    let gbm64 = Gbm::<f64, _>::new(
      0.05,
      0.2,
      64,
      Some(100.0),
      Some(1.0),
      stochastic_rs_core::simd_rng::Unseeded,
    );
    let single = sample_paths::<f32, crate::device::CudaNative, _>(&gbm32, 4, 5);
    let double = sample_paths::<f64, crate::device::CudaNative, _>(&gbm64, 4, 5);
    for (a, b) in single.iter().zip(double.iter()) {
      assert!(((*a as f64) - b).abs() < 1e-3 * b.abs().max(1.0));
    }
  }

  #[cfg(all(feature = "metal", any(feature = "gpu-cuda", feature = "gpu-wgpu")))]
  #[test]
  fn metal_native_and_cubecl_agree_seed_for_seed() {
    let gbm = Gbm::new(
      0.05,
      0.2,
      128,
      Some(100.0),
      Some(1.0),
      stochastic_rs_core::simd_rng::Unseeded,
    );
    let metal = sample_paths::<f64, crate::device::MetalNative, _>(&gbm, 16, 11);
    let cubecl = sample_paths::<f64, crate::device::CubeCl, _>(&gbm, 16, 11);
    for (a, b) in metal.iter().zip(cubecl.iter()) {
      assert!((a - b).abs() < 1e-3 * b.abs().max(1.0), "{a} vs {b}");
    }
  }
}
