use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;

use super::Fgn;
use crate::traits::ProcessExt;

fn generate_fgn_paths(h: f64, n: usize, t: f64, m: usize) -> Vec<Vec<f64>> {
  let fgn = Fgn::<f64>::new(h, n, Some(t), Unseeded);
  let mut out = Vec::with_capacity(m);
  for _ in 0..m {
    out.push(fgn.sample_cpu().to_vec());
  }
  out
}

fn unit_lag_covariance(h: f64, k: usize) -> f64 {
  if k == 0 {
    1.0
  } else {
    0.5
      * (((k + 1) as f64).powf(2.0 * h) - 2.0 * (k as f64).powf(2.0 * h)
        + ((k - 1) as f64).powf(2.0 * h))
  }
}

fn lag_covariance(paths: &[Vec<f64>], mean: f64, lag: usize) -> f64 {
  let mut s = 0.0;
  let mut c = 0usize;
  for p in paths {
    for i in 0..(p.len() - lag) {
      s += (p[i] - mean) * (p[i + lag] - mean);
      c += 1;
    }
  }
  s / c as f64
}

fn nearest_quantile(sorted: &[f64], p: f64) -> f64 {
  let idx = (((sorted.len() - 1) as f64) * p).round() as usize;
  sorted[idx]
}

#[test]
fn dt_and_scale_use_requested_length_not_fft_padding() {
  let hs = [0.2_f64, 0.7_f64];
  let ns = [3_usize, 17, 1000, 4095];
  let ts = [0.7_f64, 2.0_f64];

  for &h in &hs {
    for &n in &ns {
      for &t in &ts {
        let fgn = Fgn::<f64>::new(h, n, Some(t), Unseeded);

        // Internal FFT length is padded, but dt/scale must follow requested n.
        assert!(fgn.padded_n >= n && fgn.padded_n.is_power_of_two());
        assert!((fgn.dt() - (t / n as f64)).abs() < 1e-15);

        let expected_scale = (n as f64).powf(-h) * t.powf(h);
        assert!((fgn.scale - expected_scale).abs() < 1e-15);
      }
    }
  }
}

#[test]
fn fgn_marginal_distribution_and_covariance_match_theory() {
  let h = 0.72_f64;
  let n = 2048_usize;
  let t = 1.0_f64;
  let m = 1024_usize;
  let paths = generate_fgn_paths(h, n, t, m);

  let mut values = Vec::with_capacity(m * n);
  for p in &paths {
    values.extend_from_slice(p);
  }

  let count = values.len() as f64;
  let mean = values.iter().sum::<f64>() / count;
  let var = values
    .iter()
    .map(|x| {
      let d = *x - mean;
      d * d
    })
    .sum::<f64>()
    / count;
  let std = var.sqrt();

  let m3 = values
    .iter()
    .map(|x| {
      let d = *x - mean;
      d * d * d
    })
    .sum::<f64>()
    / count;
  let m4 = values
    .iter()
    .map(|x| {
      let d = *x - mean;
      d * d * d * d
    })
    .sum::<f64>()
    / count;
  let skew = m3 / std.powi(3);
  let excess_kurtosis = m4 / std.powi(4) - 3.0;

  let mut sorted = values.clone();
  sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let q025 = (nearest_quantile(&sorted, 0.025) - mean) / std;
  let q975 = (nearest_quantile(&sorted, 0.975) - mean) / std;

  let dt = t / n as f64;
  let var_theory = dt.powf(2.0 * h);
  let cov1_theory = var_theory * unit_lag_covariance(h, 1);
  let cov4_theory = var_theory * unit_lag_covariance(h, 4);

  let cov1_emp = lag_covariance(&paths, mean, 1);
  let cov4_emp = lag_covariance(&paths, mean, 4);

  assert!(mean.abs() < 5e-4, "mean too far from zero: {mean}");
  assert!(
    ((var / var_theory) - 1.0).abs() < 0.05,
    "variance mismatch: emp={var}, theory={var_theory}"
  );
  assert!(
    (skew.abs() < 0.05) && (excess_kurtosis.abs() < 0.10),
    "non-Gaussian marginals: skew={skew}, excess_kurtosis={excess_kurtosis}"
  );
  assert!(
    (q025 + 1.96).abs() < 0.10 && (q975 - 1.96).abs() < 0.10,
    "quantile mismatch: q025={q025}, q975={q975}"
  );
  assert!(
    ((cov1_emp / cov1_theory) - 1.0).abs() < 0.05,
    "lag-1 covariance mismatch: emp={cov1_emp}, theory={cov1_theory}"
  );
  assert!(
    ((cov4_emp / cov4_theory) - 1.0).abs() < 0.05,
    "lag-4 covariance mismatch: emp={cov4_emp}, theory={cov4_theory}"
  );
}

/// A single-precision sample keeps its long memory on a fine grid.
///
/// The circulant kernel is a second difference of `k^{2H}`, which cancels:
/// in `f32` the three terms agree to six digits by `k ≈ 4000` and their
/// combination is then exactly zero, which flattens the kernel and leaves
/// the sampler drawing white noise. At `n = 16384` the pooled lag-one
/// autocorrelation read −0.006 against a theoretical 0.3195 before the
/// embedding was built in double precision — the sign itself was wrong.
#[test]
fn single_precision_keeps_its_covariance_on_a_fine_grid() {
  let theory = 0.5 * (2.0_f64.powf(1.4) - 2.0);
  for n in [4_096usize, 16_384] {
    let paths = Fgn::<f32, _>::new(0.7, n, Some(1.0), Deterministic::new(11)).sample_par(40);
    let (mut num, mut den) = (0.0_f64, 0.0);
    for p in &paths {
      for i in 0..p.len() - 1 {
        num += p[i] as f64 * p[i + 1] as f64;
      }
      den += p.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>();
    }
    let rho = num / den;
    assert!(
      (rho - theory).abs() < 0.02,
      "n = {n}: lag-one autocorrelation {rho:.4}, theory {theory:.4}"
    );
  }
}

#[test]
fn fgn_lag1_correlation_sign_matches_hurst_regime() {
  let n = 2048_usize;
  let t = 1.0_f64;
  let m = 192_usize;

  let low_h = 0.25_f64;
  let high_h = 0.80_f64;

  let low_paths = generate_fgn_paths(low_h, n, t, m);
  let high_paths = generate_fgn_paths(high_h, n, t, m);

  let low_vals: Vec<f64> = low_paths.iter().flatten().copied().collect();
  let high_vals: Vec<f64> = high_paths.iter().flatten().copied().collect();

  let low_mean = low_vals.iter().sum::<f64>() / low_vals.len() as f64;
  let high_mean = high_vals.iter().sum::<f64>() / high_vals.len() as f64;

  let low_var = low_vals
    .iter()
    .map(|x| {
      let d = *x - low_mean;
      d * d
    })
    .sum::<f64>()
    / low_vals.len() as f64;
  let high_var = high_vals
    .iter()
    .map(|x| {
      let d = *x - high_mean;
      d * d
    })
    .sum::<f64>()
    / high_vals.len() as f64;

  let low_cov1 = lag_covariance(&low_paths, low_mean, 1);
  let high_cov1 = lag_covariance(&high_paths, high_mean, 1);

  let low_rho1 = low_cov1 / low_var;
  let high_rho1 = high_cov1 / high_var;

  assert!(
    low_rho1 < -0.10,
    "expected negative lag-1 correlation, got {low_rho1}"
  );
  assert!(
    high_rho1 > 0.10,
    "expected positive lag-1 correlation, got {high_rho1}"
  );
}

// `fbm_hurst_and_fractal_dimension_from_fgn_increments` lives in
// `stochastic-rs-stats/tests/fractal_dim_validation.rs` because it exercises
// the `FractalDim` estimator from the stats crate.

fn cross_covariance(a: &[Vec<f64>], b: &[Vec<f64>], mean_a: f64, mean_b: f64) -> f64 {
  let mut s = 0.0;
  let mut c = 0usize;
  for (pa, pb) in a.iter().zip(b.iter()) {
    for i in 0..pa.len() {
      s += (pa[i] - mean_a) * (pb[i] - mean_b);
      c += 1;
    }
  }
  s / c as f64
}

/// Primary and secondary paths of `sample_pair` must each satisfy the
/// target marginal law and be mutually independent. The latter is the
/// Dietrich–Newsam (1997) / Kroese–Botev (2013 §2.2) claim — here we
/// check the sample cross-correlation vanishes to within Monte-Carlo SE.
#[test]
fn sample_pair_independent_paths_match_theory() {
  let h = 0.68_f64;
  let n = 2048_usize;
  let t = 1.0_f64;
  let pairs_count = 1024_usize;

  let fgn = Fgn::<f64>::new(h, n, Some(t), Unseeded);
  let mut prim = Vec::with_capacity(pairs_count);
  let mut sec = Vec::with_capacity(pairs_count);
  for _ in 0..pairs_count {
    let (a, b) = fgn.sample_pair_cpu();
    prim.push(a.to_vec());
    sec.push(b.to_vec());
  }

  let dt = t / n as f64;
  let var_theory = dt.powf(2.0 * h);
  let cov1_theory = var_theory * unit_lag_covariance(h, 1);

  for (label, paths) in [("primary", &prim), ("secondary", &sec)] {
    let values: Vec<f64> = paths.iter().flatten().copied().collect();
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count;
    let cov1 = lag_covariance(paths, mean, 1);

    assert!(mean.abs() < 5e-4, "{label}: mean drift {mean}");
    assert!(
      ((var / var_theory) - 1.0).abs() < 0.05,
      "{label}: variance mismatch emp={var} theory={var_theory}"
    );
    assert!(
      ((cov1 / cov1_theory) - 1.0).abs() < 0.05,
      "{label}: lag-1 covariance mismatch emp={cov1} theory={cov1_theory}"
    );
  }

  let mean_p = prim.iter().flatten().sum::<f64>() / (prim.len() * n) as f64;
  let mean_s = sec.iter().flatten().sum::<f64>() / (sec.len() * n) as f64;
  let var_p = prim
    .iter()
    .flatten()
    .map(|x| (*x - mean_p).powi(2))
    .sum::<f64>()
    / (prim.len() * n) as f64;

  let xcov = cross_covariance(&prim, &sec, mean_p, mean_s);
  let correlation = xcov / var_p;
  assert!(
    correlation.abs() < 0.02,
    "primary/secondary correlation {correlation} too large (expected ≈ 0)"
  );
}

/// `sample_pair` with an explicit seed must be fully deterministic and
/// bit-for-bit match two separate `sample_cpu_with_seed` invocations —
/// the second one using a seed derived by replaying the same SplitMix64
/// step the single-path variant would have produced. Here we check the
/// simpler determinism-across-identical-seed property.
#[test]
fn sample_pair_is_deterministic_with_seed() {
  let fgn = Fgn::<f64>::new(0.55, 1024, Some(1.0), Unseeded);
  let (a1, b1) = fgn.sample_pair_cpu_with_seed(7);
  let (a2, b2) = fgn.sample_pair_cpu_with_seed(7);
  assert_eq!(a1, a2);
  assert_eq!(b1, b2);
}
