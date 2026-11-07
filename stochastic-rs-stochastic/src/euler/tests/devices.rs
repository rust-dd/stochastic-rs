use ndarray::Array2;

use super::*;

/// Rows of `sample_par` as an `m × n` matrix, widened to `f64` for the
/// statistics.
fn stack<T: FloatExt>(rows: &[Array1<T>]) -> Array2<f64> {
  let n = rows.first().map_or(0, |r| r.len());
  let mut out = Array2::<f64>::zeros((rows.len(), n));
  for (i, row) in rows.iter().enumerate() {
    for (j, x) in row.iter().enumerate() {
      out[[i, j]] = x.to_f64().unwrap();
    }
  }
  out
}

fn column_mean_var(paths: &Array2<f64>, col: usize) -> (f64, f64) {
  let column = paths.column(col);
  let m = column.len() as f64;
  let mean = column.sum() / m;
  let var = column.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (m - 1.0);
  (mean, var)
}

fn gbm<T: FloatExt>(seed: u64) -> Gbm<T, Deterministic> {
  Gbm::new(
    T::from_f64_fast(0.05),
    T::from_f64_fast(0.2),
    253,
    Some(T::from_f64_fast(100.0)),
    Some(T::one()),
    Deterministic::new(seed),
  )
}

fn gbm_moments_hold<T: FloatExt, B: EulerBackend<T> + Default>(label: &str) {
  let paths = stack(&gbm::<T>(7).on::<B>().sample_par(40_000));
  assert_eq!(paths.dim(), (40_000, 253), "{label}");
  assert!(paths.column(0).iter().all(|&x| x == 100.0), "{label}");
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
  let a = gbm::<T>(7).on::<B>().sample_par(8);
  let b = gbm::<T>(7).on::<B>().sample_par(8);
  let c = gbm::<T>(8).on::<B>().sample_par(8);
  assert!(
    a.iter().zip(&b).all(|(x, y)| x == y),
    "{label}: seed reproducibility"
  );
  assert!(
    a.iter().zip(&c).any(|(x, y)| x != y),
    "{label}: seed discrimination"
  );
  assert_ne!(a[0], a[1], "{label}: paths are distinct streams");
  let single = gbm::<T>(7).on::<B>().sample();
  assert_eq!(single.len(), 253, "{label}");
  assert_eq!(
    gbm::<T>(7).on::<B>().sample_map(3, |p| p[252]).len(),
    3,
    "{label}"
  );
}

fn cir_stays_nonnegative<T: FloatExt, B: EulerBackend<T> + Default>(label: &str) {
  let cir = Cir::new(
    T::from_f64_fast(1.5),
    T::from_f64_fast(0.04),
    T::from_f64_fast(0.3),
    253,
    Some(T::from_f64_fast(0.09)),
    Some(T::one()),
    None,
    Deterministic::new(3),
  );
  let paths = stack(&cir.on::<B>().sample_par(4_000));
  let (mean, _) = column_mean_var(&paths, 252);
  let expected = 0.04 + (0.09 - 0.04) * (-1.5_f64).exp();
  assert!(
    (mean / expected - 1.0).abs() < 0.03,
    "{label}: mean {mean} vs {expected}"
  );
  assert!(paths.iter().all(|&x| x >= 0.0), "{label}");
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_native_chunks_are_bit_identical_to_one_launch() {
  use crate::device::Metal;
  let whole = gbm::<f32>(21).on::<Metal>().sample_par(10);
  let process = gbm::<f32>(21).on::<Metal>();
  let middle = Metal::default()
    .euler_kernel(&process, 3, 4, process.device_seed())
    .expect("Metal");
  for (k, row) in middle.outer_iter().enumerate() {
    assert_eq!(whole[3 + k], row.to_owned(), "paths 3..7 of one launch");
  }
  // A budget of three paths forces four launches; the union must not move.
  let small = Metal::default().with_batch_budget(253 * 4 * 3);
  let chunked = small.euler_paths(&gbm::<f32>(21), 10);
  let mapped: Vec<f32> = small.euler_paths_map(&gbm::<f32>(21), 10, |p| p[252]);
  assert_eq!(chunked, whole);
  assert_eq!(mapped, whole.iter().map(|p| p[252]).collect::<Vec<_>>());
}

/// The chunk-invariance the engine promises, for the fractional launches
/// rather than the hashed ones. A Gaussian family hashes its noise from
/// `(first_path + path, step, seed)` and so is chunk-invariant by
/// construction; a fractional one draws from a pipeline whose counter the
/// launch has to advance itself, and a two-stream launch has to advance it
/// by `streams` rows a path. Both are places a batch can silently repeat or
/// overlap its own noise, which no statistic of a single batch reveals.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_fractional_chunks_are_bit_identical_to_one_launch() {
  use stochastic_rs_core::simd_rng::Deterministic;

  use crate::device::Metal;
  use crate::euler::EulerBackend;
  use crate::noise::cfgns::Cfgns;
  use crate::process::cfbms::Cfbms;
  use crate::process::fbm::Fbm;

  // Small enough that a modest budget forces several launches.
  const N: usize = 16;
  const M: usize = 6;
  let small = Metal::default().with_batch_budget(N * 4 * 2);

  let fbm = || Fbm::<f32, _>::new(0.7, N, Some(1.0), Deterministic::new(5)).on::<Metal>();
  let whole = fbm().sample_par(M);
  assert_eq!(
    whole,
    small.euler_paths(&fbm(), M),
    "fBM: a chunked batch is not the whole one"
  );
  assert!(
    whole.windows(2).all(|w| w[0] != w[1]),
    "fBM: two paths of one batch are identical, so a chunk repeated itself"
  );

  // The two-stream launches: each path takes two rows of the embedding, so
  // a chunk that advances by one row a path hands its own second stream to
  // the next chunk as that chunk's first.
  let cfbms = || Cfbms::<f32, _>::new(0.7, 0.0, N, Some(1.0), Deterministic::new(5)).on::<Metal>();
  let pairs = cfbms().sample_par(M);
  assert_eq!(
    pairs,
    small.system_paths(&cfbms(), M),
    "correlated fBM: chunking moved the batch"
  );
  for (i, a) in pairs.iter().enumerate() {
    for (j, b) in pairs.iter().enumerate() {
      if i != j {
        assert_ne!(a[0], b[0], "correlated fBM: paths {i} and {j} share row 0");
      }
      assert_ne!(
        a[1], b[0],
        "correlated fBM: path {i}'s second stream is path {j}'s first"
      );
    }
  }

  let cfgns = || Cfgns::<f32, _>::new(0.7, 0.0, N, Some(1.0), Deterministic::new(5)).on::<Metal>();
  assert_eq!(
    cfgns().sample_par(M),
    small.system_paths(&cfgns(), M),
    "correlated fGn: chunking moved the batch"
  );
}

/// [`metal_fractional_chunks_are_bit_identical_to_one_launch`] on CUDA. The
/// two backends advance the pipeline's counter by different means — Metal
/// offsets the seed, CUDA offsets the row index — so the invariant is
/// asserted separately on each rather than assumed to carry over.
#[cfg(feature = "cuda")]
#[test]
fn cuda_fractional_chunks_are_bit_identical_to_one_launch() {
  use stochastic_rs_core::simd_rng::Deterministic;

  use crate::device::Cuda;
  use crate::euler::EulerBackend;
  use crate::noise::cfgns::Cfgns;
  use crate::process::cfbms::Cfbms;
  use crate::process::fbm::Fbm;

  // Small enough that a modest budget forces several launches.
  const N: usize = 16;
  const M: usize = 6;
  let small = Cuda::default().with_batch_budget(N * 4 * 2);

  let fbm = || Fbm::<f32, _>::new(0.7, N, Some(1.0), Deterministic::new(5)).on::<Cuda>();
  let whole = fbm().sample_par(M);
  assert_eq!(
    whole,
    small.euler_paths(&fbm(), M),
    "fBM: a chunked batch is not the whole one"
  );
  assert!(
    whole.windows(2).all(|w| w[0] != w[1]),
    "fBM: two paths of one batch are identical, so a chunk repeated itself"
  );

  // The two-stream launches: each path takes two rows of the embedding, so
  // a chunk that advances by one row a path hands its own second stream to
  // the next chunk as that chunk's first.
  let cfbms = || Cfbms::<f32, _>::new(0.7, 0.0, N, Some(1.0), Deterministic::new(5)).on::<Cuda>();
  let pairs = cfbms().sample_par(M);
  assert_eq!(
    pairs,
    small.system_paths(&cfbms(), M),
    "correlated fBM: chunking moved the batch"
  );
  for (i, a) in pairs.iter().enumerate() {
    for (j, b) in pairs.iter().enumerate() {
      if i != j {
        assert_ne!(a[0], b[0], "correlated fBM: paths {i} and {j} share row 0");
      }
      assert_ne!(
        a[1], b[0],
        "correlated fBM: path {i}'s second stream is path {j}'s first"
      );
    }
  }

  let cfgns = || Cfgns::<f32, _>::new(0.7, 0.0, N, Some(1.0), Deterministic::new(5)).on::<Cuda>();
  assert_eq!(
    cfgns().sample_par(M),
    small.system_paths(&cfgns(), M),
    "correlated fGn: chunking moved the batch"
  );
}

/// Metal indexes in `uint`, so no launch passes that range: a plane of values, or the `2n`-point
/// fGN rows that bind a correlated pair before its planes do. Rows that already fit are kept.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_index_rows_stop_at_the_uint_range() {
  use crate::device::Metal;
  use crate::process::cfbms::Cfbms;

  let limit = u32::MAX as usize;
  let one = gbm::<f32>(1).euler_spec();
  assert_eq!(
    Metal::default().index_rows(usize::MAX, one, 253, None),
    limit / 253
  );
  assert_eq!(Metal::default().index_rows(7, one, 253, None), 7);
  assert_eq!(Metal::default().index_rows(7, one, usize::MAX, None), 1);
  assert_eq!(crate::noise::fgn::metal::index_rows(7, 2, usize::MAX), 1);
  let pair = Cfbms::<f32, _>::new(0.7, 0.0, 1024, Some(1.0), Deterministic::new(5));
  let fgn = pair.fgn_spec().expect("a fractional pair");
  let steps = pair.grid_points();
  let rows = Metal::default().index_rows(usize::MAX, pair.euler_spec(), steps, pair.fgn_spec());
  assert_eq!(rows, limit / (fgn.streams * 2 * fgn.n));
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_rejects_wrapped_path_offsets_before_opening_a_device() {
  use crate::device::{DeviceError, Metal};
  use crate::process::fbm::Fbm;

  let device = Metal::new(usize::MAX);
  let plain = device.euler_kernel(&gbm::<f32>(1), usize::MAX, 1, 0);
  assert!(matches!(plain, Err(DeviceError::Launch(_))));
  let fractional = Fbm::<f32, _>::new(0.7, 16, Some(1.0), Deterministic::new(5));
  let result = device.euler_kernel(&fractional, usize::MAX, 1, 0);
  assert!(matches!(result, Err(DeviceError::Launch(_))));
}

/// A Gaussian CUDA launch is bound only by the `unsigned int` path count; a fractional one by the
/// fGN work buffer, indexed in `int` at `4n` values a path.
#[cfg(feature = "cuda")]
#[test]
fn cuda_index_rows_bound_paths_and_the_fgn_buffer() {
  use crate::device::Cuda;
  use crate::process::fbm::Fbm;

  let device = Cuda::default();
  let one = gbm::<f32>(1);
  let rows = device.index_rows(usize::MAX, one.euler_spec(), one.grid_points(), None);
  assert_eq!(rows, u32::MAX as usize);

  let fbm = Fbm::<f32, _>::new(0.7, 1024, Some(1.0), Deterministic::new(5));
  let n = fbm.fgn_spec().expect("a fractional process").n;
  let rows = device.index_rows(
    usize::MAX,
    fbm.euler_spec(),
    fbm.grid_points(),
    fbm.fgn_spec(),
  );
  assert_eq!(rows, i32::MAX as usize / (4 * n));
}

/// A refused Metal allocation is the out-of-memory error the batch loops halve on.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_refused_allocation_is_out_of_memory() {
  use crate::device::metal::{context, device_error};

  let context = context(0).expect("this Mac has a Metal device");
  let refused = context
    .device
    .buffer_zeroed::<f32>(1usize << 48)
    .map_err(device_error);
  assert!(refused.is_err_and(|e| e.is_out_of_memory()));
  let granted = context.device.buffer_zeroed::<f32>(16).expect("64 bytes");
  assert_eq!(granted.len(), 16);
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_native_matrix_matches_the_rows_and_chunks() {
  use crate::device::Metal;
  let rows = gbm::<f32>(23).on::<Metal>().sample_par(7);
  let matrix = gbm::<f32>(23)
    .on::<Metal>()
    .try_sample_matrix(7)
    .expect("Metal");
  for (i, row) in rows.iter().enumerate() {
    assert_eq!(matrix.row(i), row.view());
  }
  let chunked = Metal::default()
    .with_batch_budget(253 * 4 * 2)
    .try_euler_matrix(&gbm::<f32>(23), 7)
    .expect("Metal");
  assert_eq!(chunked, matrix);
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_native_probe_and_try_sample_par() {
  let info = crate::device::Metal::default()
    .probe()
    .expect("this Mac has a Metal device");
  assert_eq!(info.backend, "Metal");
  assert_eq!(info.precisions, &["f32"]);
  assert!(!info.name.is_empty());
  let paths = gbm::<f32>(3)
    .on::<crate::device::Metal>()
    .try_sample_par(5)
    .expect("Metal");
  assert_eq!(paths.len(), 5);
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_native_backend_matches_the_moments() {
  gbm_moments_hold::<f32, crate::device::Metal>("Metal");
  cir_stays_nonnegative::<f32, crate::device::Metal>("Metal");
}

/// A batch above the budget runs through the two-stream pipeline; its
/// union must equal one launch path for path, in both precisions.
#[cfg(feature = "cuda")]
#[test]
fn cuda_chunks_are_bit_identical_to_one_launch() {
  use crate::device::Cuda;
  let whole64 = gbm::<f64>(21).on::<Cuda>().sample_par(10);
  let whole32 = gbm::<f32>(21).on::<Cuda>().sample_par(10);
  let process = gbm::<f64>(21).on::<Cuda>();
  let middle = Cuda::default()
    .euler_kernel(&process, 3, 4, process.device_seed())
    .expect("CUDA");
  for (k, row) in middle.outer_iter().enumerate() {
    assert_eq!(whole64[3 + k], row.to_owned(), "paths 3..7 of one launch");
  }
  // Three paths per chunk: four launches alternating between the two streams.
  let small = Cuda::default().with_batch_budget(253 * 8 * 3);
  let chunked64 = small.euler_paths(&gbm::<f64>(21), 10);
  let chunked32 = small.euler_paths(&gbm::<f32>(21), 10);
  let mapped: Vec<f64> = small.euler_paths_map(&gbm::<f64>(21), 10, |p| p[252]);
  assert_eq!(chunked64, whole64);
  assert_eq!(chunked32, whole32);
  assert_eq!(mapped, whole64.iter().map(|p| p[252]).collect::<Vec<_>>());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_backend_matches_the_moments() {
  gbm_moments_hold::<f64, crate::device::Cuda>("Cuda f64");
  gbm_moments_hold::<f32, crate::device::Cuda>("Cuda f32");
  cir_stays_nonnegative::<f64, crate::device::Cuda>("Cuda f64");
  cir_stays_nonnegative::<f32, crate::device::Cuda>("Cuda f32");
  // The f32 and f64 kernels draw the same uniforms (one integer hash) and
  // differ only by float rounding, so the same process on the same grid
  // agrees across the two precisions to well under 1e-3 relative.
  let single = gbm::<f32>(5).on::<crate::device::Cuda>().sample_par(4);
  let double = gbm::<f64>(5).on::<crate::device::Cuda>().sample_par(4);
  assert_eq!(single.len(), double.len());
  for (a, b) in single.iter().zip(&double) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
      assert!(
        ((*x as f64) - y).abs() < 1e-3 * y.abs().max(1.0),
        "f32 {x} vs f64 {y}"
      );
    }
  }
}
