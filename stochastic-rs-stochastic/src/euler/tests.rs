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

#[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu"))]
mod device {
  use super::*;
  use crate::device::CubeCl;

  /// The device path reproduces the lognormal moments too — same stepper,
  /// different random stream.
  #[test]
  fn gpu_gbm_paths_match_the_lognormal_moments() {
    let gbm = Gbm::new(
      0.05,
      0.2,
      253,
      Some(100.0),
      Some(1.0),
      stochastic_rs_core::simd_rng::Unseeded,
    );
    let paths = sample_paths::<f64, CubeCl, _>(&gbm, 40_000, 7);
    assert_eq!(paths.dim(), (40_000, 253));
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
    let again = sample_paths::<f64, CubeCl, _>(&gbm, 8, 7);
    assert_eq!(again, sample_paths::<f64, CubeCl, _>(&gbm, 8, 7));
  }
}
