//! Monte-Carlo FBM validation for every shipped Hurst estimator.
//!
//! For each `H ∈ {0.3, 0.5, 0.7}` we draw `m = 16` exact FBM paths
//! (Davies-Harte) of length `n = 8192` and check that every estimator
//! returns within tolerance of the true `H`.  Tolerance is per-method
//! because not all estimators are equally efficient (`Wavelet` is
//! tighter than `RescaledRange`, `Whittle` is intended for rough-vol
//! log-RV and not bare FBM).

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stats::fractal_dim::Higuchi;
use stochastic_rs_stats::fractal_dim::Variogram;
use stochastic_rs_stats::hurst::Dfa;
use stochastic_rs_stats::hurst::Gph;
use stochastic_rs_stats::hurst::HurstEstimator;
use stochastic_rs_stats::hurst::RescaledRange;
use stochastic_rs_stats::hurst::VariationKind;
use stochastic_rs_stats::hurst::Variations;
use stochastic_rs_stats::hurst::Wavelet;
use stochastic_rs_stochastic::process::fbm::Fbm;
use stochastic_rs_stochastic::traits::ProcessExt;

const H_VALUES: [f64; 3] = [0.3, 0.5, 0.7];
const N_PATHS: usize = 16;
const PATH_LEN: usize = 8192;

fn fbm_paths(h: f64) -> Vec<Array1<f64>> {
  (0..N_PATHS)
    .map(|_| Fbm::new(h, PATH_LEN, Some(1.0), Unseeded).sample())
    .collect()
}

fn mean_h_over_paths<E>(estimator: &E, paths: &[Array1<f64>]) -> f64
where
  E: HurstEstimator<f64>,
{
  let mut acc = 0.0;
  let mut count = 0;
  for p in paths {
    if let Ok(r) = estimator.estimate(p.view()) {
      acc += r.hurst;
      count += 1;
    }
  }
  acc / count.max(1) as f64
}

#[test]
fn rescaled_range_within_tolerance_on_fbm() {
  let est = RescaledRange::default();
  for &h in &H_VALUES {
    let paths = fbm_paths(h);
    let h_est = mean_h_over_paths(&est, &paths);
    assert!(
      (h_est - h).abs() < 0.10,
      "RescaledRange H_est={h_est:.3}, true={h:.3}"
    );
  }
}

#[test]
fn dfa_within_tolerance_on_fbm() {
  let est = Dfa {
    assume_integrated: true,
    ..Default::default()
  };
  for &h in &H_VALUES {
    let paths = fbm_paths(h);
    let h_est = mean_h_over_paths(&est, &paths);
    assert!(
      (h_est - h).abs() < 0.10,
      "Dfa H_est={h_est:.3}, true={h:.3}"
    );
  }
}

#[test]
fn gph_within_tolerance_on_fbm() {
  let est = Gph::default();
  for &h in &H_VALUES {
    let paths = fbm_paths(h);
    let h_est = mean_h_over_paths(&est, &paths);
    assert!(
      (h_est - h).abs() < 0.12,
      "Gph H_est={h_est:.3}, true={h:.3}"
    );
  }
}

#[test]
fn wavelet_within_tolerance_on_fbm() {
  let est = Wavelet::default();
  for &h in &H_VALUES {
    let paths = fbm_paths(h);
    let h_est = mean_h_over_paths(&est, &paths);
    assert!(
      (h_est - h).abs() < 0.12,
      "Wavelet H_est={h_est:.3}, true={h:.3}"
    );
  }
}

#[test]
fn higuchi_via_hurst_trait_within_tolerance_on_fbm() {
  let est = Higuchi { kmax: 32 };
  for &h in &H_VALUES {
    let paths = fbm_paths(h);
    let h_est = mean_h_over_paths(&est, &paths);
    assert!(
      (h_est - h).abs() < 0.10,
      "Higuchi (Hurst trait) H_est={h_est:.3}, true={h:.3}"
    );
  }
}

#[test]
fn variogram_via_hurst_trait_within_tolerance_on_fbm() {
  let est = Variogram { p: 2.0 };
  for &h in &H_VALUES {
    let paths = fbm_paths(h);
    let h_est = mean_h_over_paths(&est, &paths);
    assert!(
      (h_est - h).abs() < 0.10,
      "Variogram (Hurst trait) H_est={h_est:.3}, true={h:.3}"
    );
  }
}

#[test]
fn variations_central_diff_within_tolerance_on_fbm() {
  let est = Variations {
    kind: VariationKind::CentralDiff,
  };
  for &h in &H_VALUES {
    let paths = fbm_paths(h);
    let h_est = mean_h_over_paths(&est, &paths);
    assert!(
      (h_est - h).abs() < 0.10,
      "Variations(CentralDiff) H_est={h_est:.3}, true={h:.3}"
    );
  }
}

#[test]
fn variations_daubechies_within_tolerance_on_fbm() {
  let est = Variations {
    kind: VariationKind::Daubechies,
  };
  for &h in &H_VALUES {
    let paths = fbm_paths(h);
    let h_est = mean_h_over_paths(&est, &paths);
    assert!(
      (h_est - h).abs() < 0.10,
      "Variations(Daubechies) H_est={h_est:.3}, true={h:.3}"
    );
  }
}

#[test]
fn variations_power_variation_within_tolerance_on_fbm() {
  let est = Variations {
    kind: VariationKind::PowerVariation { k: 2, p: 2.0 },
  };
  for &h in &H_VALUES {
    let paths = fbm_paths(h);
    let h_est = mean_h_over_paths(&est, &paths);
    assert!(
      (h_est - h).abs() < 0.10,
      "Variations(PowerVariation) H_est={h_est:.3}, true={h:.3}"
    );
  }
}
