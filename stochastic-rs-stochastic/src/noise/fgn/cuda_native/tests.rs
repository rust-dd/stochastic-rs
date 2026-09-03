use stochastic_rs_core::simd_rng::Unseeded;

use super::super::Fgn;

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

#[test]
fn cuda_native_single_path_shape() {
  let fgn = Fgn::<f64>::new(0.7, 1024, Some(1.0), Unseeded);
  let result = fgn
    .sample_cuda_native_impl(1)
    .expect("single path should succeed");
  assert_eq!(result.shape(), &[1, 1024]);
}

#[test]
fn cuda_native_batch_shape() {
  let fgn = Fgn::<f64>::new(0.7, 512, Some(1.0), Unseeded);
  let m = 64;
  let batch = fgn
    .sample_cuda_native_impl(m)
    .expect("batch should succeed");
  assert_eq!(batch.shape(), &[m, 512]);
}

#[test]
fn cuda_native_f32_works() {
  let fgn = Fgn::<f32>::new(0.7, 1024, Some(1.0), Unseeded);
  let batch = fgn.sample_cuda_native_impl(4).expect("f32 should succeed");
  assert_eq!(batch.shape(), &[4, 1024]);
}

#[test]
fn cuda_native_non_power_of_two_n() {
  let fgn = Fgn::<f64>::new(0.7, 3000, Some(1.0), Unseeded);
  let batch = fgn
    .sample_cuda_native_impl(8)
    .expect("non-pot n should work");
  assert_eq!(batch.shape(), &[8, 3000]);
}

#[test]
fn cuda_native_eigenvalues_structural() {
  let fgn = Fgn::<f64>::new(0.72, 2048, Some(1.0), Unseeded);
  let eigs = &*fgn.sqrt_eigenvalues;

  assert_eq!(eigs.len(), 2 * fgn.n);
  assert!(eigs.iter().all(|&v| v >= 0.0));

  for i in 1..eigs.len() / 2 {
    let diff = (eigs[i] - eigs[eigs.len() - i]).abs();
    assert!(
      diff < 1e-10,
      "eigs[{i}]={} != eigs[{}]={}",
      eigs[i],
      eigs.len() - i,
      eigs[eigs.len() - i]
    );
  }

  let energy: f64 = eigs.iter().map(|&v| v * v).sum();
  assert!(
    (energy - 1.0).abs() < 1e-6,
    "eigenvalue energy sum should be 1.0, got {energy}"
  );
}

#[test]
fn cuda_native_scale_matches_cpu() {
  for &n in &[1024_usize, 3000, 4096] {
    let fgn = Fgn::<f64>::new(0.7, n, Some(2.0), Unseeded);
    let cpu_scale = fgn.scale;

    let out_size = fgn.n - fgn.offset;
    let scale_steps = out_size.max(1);
    let cuda_scale = (scale_steps as f64).powf(-0.7) * 2.0_f64.powf(0.7);

    assert!(
      (cpu_scale - cuda_scale).abs() < 1e-14,
      "scale mismatch for n={n}: cpu={cpu_scale}, cuda={cuda_scale}"
    );
  }
}

#[test]
fn cuda_native_variance_matches_cpu() {
  let h = 0.72_f64;
  let n = 2048_usize;
  let t = 1.0_f64;
  let m = 1024_usize;
  let fgn = Fgn::<f64>::new(h, n, Some(t), Unseeded);

  let cpu_paths: Vec<Vec<f64>> = (0..m).map(|_| fgn.sample_cpu().to_vec()).collect();
  let cpu_vals: Vec<f64> = cpu_paths.iter().flatten().copied().collect();
  let cpu_mean = cpu_vals.iter().sum::<f64>() / cpu_vals.len() as f64;
  let cpu_var =
    cpu_vals.iter().map(|x| (x - cpu_mean).powi(2)).sum::<f64>() / cpu_vals.len() as f64;

  let cuda_batch = fgn
    .sample_cuda_native_impl(m)
    .expect("cuda batch should succeed");
  let cuda_vals: Vec<f64> = cuda_batch.iter().copied().collect();
  let cuda_mean = cuda_vals.iter().sum::<f64>() / cuda_vals.len() as f64;
  let cuda_var = cuda_vals
    .iter()
    .map(|x| (x - cuda_mean).powi(2))
    .sum::<f64>()
    / cuda_vals.len() as f64;

  let ratio = cuda_var / cpu_var;
  assert!(
    (ratio - 1.0).abs() < 0.15,
    "CUDA vs CPU variance ratio = {ratio} (cuda={cuda_var}, cpu={cpu_var})"
  );
}

#[test]
fn cuda_native_covariance_structure_matches_cpu() {
  let h = 0.72_f64;
  let n = 2048_usize;
  let t = 1.0_f64;
  let m = 1024_usize;
  let fgn = Fgn::<f64>::new(h, n, Some(t), Unseeded);

  let cpu_paths: Vec<Vec<f64>> = (0..m).map(|_| fgn.sample_cpu().to_vec()).collect();
  let cpu_vals: Vec<f64> = cpu_paths.iter().flatten().copied().collect();
  let cpu_mean = cpu_vals.iter().sum::<f64>() / cpu_vals.len() as f64;
  let cpu_cov1 = lag_covariance(&cpu_paths, cpu_mean, 1);
  let cpu_cov4 = lag_covariance(&cpu_paths, cpu_mean, 4);

  let cuda_batch = fgn
    .sample_cuda_native_impl(m)
    .expect("cuda batch should succeed");
  let cuda_paths: Vec<Vec<f64>> = cuda_batch.rows().into_iter().map(|r| r.to_vec()).collect();
  let cuda_vals: Vec<f64> = cuda_paths.iter().flatten().copied().collect();
  let cuda_mean = cuda_vals.iter().sum::<f64>() / cuda_vals.len() as f64;
  let cuda_cov1 = lag_covariance(&cuda_paths, cuda_mean, 1);
  let cuda_cov4 = lag_covariance(&cuda_paths, cuda_mean, 4);

  let ratio1 = cuda_cov1 / cpu_cov1;
  let ratio4 = cuda_cov4 / cpu_cov4;
  assert!(
    (ratio1 - 1.0).abs() < 0.15,
    "lag-1 cov ratio = {ratio1} (cuda={cuda_cov1}, cpu={cpu_cov1})"
  );
  assert!(
    (ratio4 - 1.0).abs() < 0.15,
    "lag-4 cov ratio = {ratio4} (cuda={cuda_cov4}, cpu={cpu_cov4})"
  );
}

/// Two `Fgn`s from the same `Deterministic` seed agree bit for bit even when
/// an unrelated device draw happens between them: the Philox offset is a
/// function of the seed, not of the process's launch history.
#[test]
fn cuda_native_same_seed_same_paths_regardless_of_history() {
  use stochastic_rs_core::simd_rng::Deterministic;
  let first = Fgn::<f64, _>::new(0.7, 256, Some(1.0), Deterministic::new(7))
    .sample_cuda_native_impl(4)
    .expect("first batch");
  let _interleaved = Fgn::<f64>::new(0.3, 128, Some(1.0), Unseeded)
    .sample_cuda_native_impl(2)
    .expect("interleaved batch");
  let second = Fgn::<f64, _>::new(0.7, 256, Some(1.0), Deterministic::new(7))
    .sample_cuda_native_impl(4)
    .expect("second batch");
  assert_eq!(first, second, "same seed must give the same device paths");
  let other = Fgn::<f64, _>::new(0.7, 256, Some(1.0), Deterministic::new(8))
    .sample_cuda_native_impl(4)
    .expect("other seed");
  assert_ne!(
    first, other,
    "a different seed must give different device paths"
  );
}

/// A batch produced in chunks equals one launch, path for path: one seed per
/// batch and an element offset per chunk, in both precisions.
#[test]
fn cuda_native_chunks_are_bit_identical_to_one_launch() {
  use stochastic_rs_core::simd_rng::Deterministic;
  let whole64 = Fgn::<f64, _>::new(0.7, 512, Some(1.0), Deterministic::new(5))
    .sample_cuda_native_impl(9)
    .expect("whole f64");
  let whole32 = Fgn::<f32, _>::new(0.7, 512, Some(1.0), Deterministic::new(5))
    .sample_cuda_native_impl(9)
    .expect("whole f32");
  // Two paths per chunk: five launches for nine paths.
  crate::device::set_batch_budget_bytes((4 * 512 + 512) * 8 * 2);
  let chunked64 = Fgn::<f64, _>::new(0.7, 512, Some(1.0), Deterministic::new(5))
    .sample_cuda_native_impl(9)
    .expect("chunked f64");
  crate::device::set_batch_budget_bytes((4 * 512 + 512) * 4 * 2);
  let chunked32 = Fgn::<f32, _>::new(0.7, 512, Some(1.0), Deterministic::new(5))
    .sample_cuda_native_impl(9)
    .expect("chunked f32");
  crate::device::set_batch_budget_bytes(crate::device::DEFAULT_BATCH_BUDGET_BYTES);
  assert_eq!(whole64, chunked64);
  assert_eq!(whole32, chunked32);
  assert_ne!(whole64.row(0), whole64.row(1));
}
