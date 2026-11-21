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
    T::from_f64_fast(100.0),
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
    VarianceBoundary::Floor,
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

#[cfg(metal_backend)]
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

/// The kernel's own draws are chunk-invariant by their Philox counters; a fractional launch moves
/// its fGN counter itself, `streams` rows a path, and a slip there shows in no one-batch statistic.
#[cfg(metal_backend)]
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

  // The two-stream launches: each path takes two rows of the embedding, so a chunk that advanced
  // by one row a path would hand its second stream to the next chunk as that chunk's first.
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

/// [`metal_fractional_chunks_are_bit_identical_to_one_launch`] on CUDA, asserted separately:
/// Metal offsets the seed to advance the pipeline's counter, CUDA offsets the row index.
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

  // The two-stream launches: each path takes two rows of the embedding, so a chunk that advanced
  // by one row a path would hand its second stream to the next chunk as that chunk's first.
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
#[cfg(metal_backend)]
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

#[cfg(metal_backend)]
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
#[cfg(metal_backend)]
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

#[cfg(metal_backend)]
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

#[cfg(metal_backend)]
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

#[cfg(metal_backend)]
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
  // The two precisions read the same Philox words, an f32 draw the f64 one truncated, so the same
  // process on the same grid agrees across them to well under 1e-3 relative.
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

/// `refuse`'s answer, which must come at once: under a one-path budget an unrefused batch would
/// run four billion launches, so it runs on a thread of its own and times out.
fn promptly<R: Send + 'static>(refuse: impl FnOnce() -> R + Send + 'static) -> R {
  let (tx, rx) = std::sync::mpsc::channel();
  std::thread::spawn(move || tx.send(refuse()));
  rx.recv_timeout(std::time::Duration::from_secs(60))
    .expect("the batch is refused before its first launch")
}

/// Every batch entry refuses more paths than the kernels' 32-bit path index reaches, before it
/// allocates or launches, and draws the one seed any call draws; `sample_par` panics alike.
fn unindexable_batches_are_refused<K>(device: K)
where
  K: EulerKernel<f32> + EulerBackend<f32> + 'static,
{
  use std::panic::{AssertUnwindSafe, catch_unwind};

  use crate::device::DeviceError;
  use crate::volatility::bergomi::Bergomi;

  let m = u32::MAX as usize + 1;
  let gbm = Gbm::<f32, _>::new(0.05, 0.2, 2, 100.0, Some(1.0), Deterministic::new(1))
    .with_backend(device.clone());
  let bergomi = Bergomi::<f32, _>::new(
    0.5,
    0.04,
    100.0,
    0.0,
    -0.7,
    64,
    Some(1.0),
    Deterministic::new(2),
  )
  .with_backend(device.clone());
  let (once_gbm, once_bergomi) = (gbm.clone(), bergomi.clone());
  let _ = (once_gbm.device_seed(), once_bergomi.device_seed());
  let (g, b) = (once_gbm.seed.current(), once_bergomi.seed.current());
  let refusals: [(Result<(), DeviceError>, u64); 5] = [
    promptly({
      let p = gbm.clone();
      move || (p.try_sample_par(m).map(drop), p.seed.current())
    }),
    promptly({
      let (p, d) = (gbm.clone(), device.clone());
      move || {
        let refusal = d.try_euler_paths_map(&p, m, |path| path[0]).map(drop);
        (refusal, p.seed.current())
      }
    }),
    promptly({
      let (p, d) = (gbm.clone(), device.clone());
      move || {
        let refusal = d.try_euler_paths_map_view(&p, m, |path| path[0]).map(drop);
        (refusal, p.seed.current())
      }
    }),
    promptly({
      let p = bergomi.clone();
      move || (p.try_sample_par(m).map(drop), p.seed.current())
    }),
    promptly({
      let (p, d) = (bergomi, device);
      move || {
        let refusal = d.try_system_paths_map(&p, m, |paths| paths[0][0]).map(drop);
        (refusal, p.seed.current())
      }
    }),
  ];
  for ((refusal, seed), once) in refusals.into_iter().zip([g, g, g, b, b]) {
    assert!(
      matches!(&refusal, Err(DeviceError::Launch(why)) if why.contains("32-bit path index")),
      "{refusal:?}"
    );
    assert_eq!(seed, once, "a refusal draws exactly one seed");
  }
  let panic = promptly(move || catch_unwind(AssertUnwindSafe(move || drop(gbm.sample_par(m)))))
    .expect_err("sample_par panics");
  let message = panic.downcast_ref::<String>().expect("a formatted panic");
  assert!(message.contains("32-bit path index"), "{message}");
}

#[cfg(metal_backend)]
#[test]
fn metal_refuses_more_paths_than_the_kernels_index() {
  unindexable_batches_are_refused(crate::device::Metal::default().with_batch_budget(8));
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_refuses_more_paths_than_the_kernels_index() {
  unindexable_batches_are_refused(crate::device::Cuda::default().with_batch_budget(8));
}

/// An empty launch answers before the host encodes its family: a history grid past the kernels'
/// cap never reaches the check that would refuse it.
fn an_empty_launch_answers_first<K: EulerKernel<f32>>(device: K) {
  use crate::autoregressive::arima::Arima;

  let n = HISTORY_SLOTS + 1;
  let arima = Arima::<f32, _>::new(
    ndarray::array![0.5],
    ndarray::array![0.2],
    0,
    1.0,
    n,
    Unseeded,
  );
  let paths = device
    .euler_kernel(&arima, 0, 0, 1)
    .expect("an empty launch");
  assert_eq!(paths.dim(), (0, n));
}

#[cfg(metal_backend)]
#[test]
fn metal_empty_launch_answers_first() {
  an_empty_launch_answers_first(crate::device::Metal::default());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_launch_answers_first() {
  an_empty_launch_answers_first(crate::device::Cuda::default());
}

/// A system launch budgets every plane its family steps: Bergomi steps four for the two it returns.
fn system_rows_count_every_plane<K: EulerKernel<f32>>(device: K) {
  use crate::volatility::bergomi::Bergomi;

  let p = Bergomi::<f32, _>::new(0.5, 0.04, 100.0, 0.0, -0.7, 64, Some(1.0), Unseeded);
  let (planes, rows) = system_rows(&device, &p);
  assert_eq!(planes, 4);
  assert_eq!(
    rows,
    crate::device::chunk_rows(device.batch_budget(), 4 * 64, 4)
  );
}

#[cfg(metal_backend)]
#[test]
fn metal_system_rows_count_every_plane() {
  system_rows_count_every_plane(crate::device::Metal::default().with_batch_budget(1 << 20));
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_system_rows_count_every_plane() {
  system_rows_count_every_plane(crate::device::Cuda::default().with_batch_budget(1 << 20));
}

/// A release drops every cache the calling thread holds on the device, so the device context goes
/// with them; releasing another ordinal leaves them alone.
#[cfg(metal_backend)]
#[test]
fn metal_release_frees_every_cache_holding_the_device() {
  use crate::device::Metal;
  use crate::process::fbm::Fbm;
  use crate::sheet::fbs::Fbs;

  let _ = gbm::<f32>(3).on::<Metal>().sample_par(4);
  let _ = Fbm::<f32, _>::new(0.7, 64, Some(1.0), Deterministic::new(5))
    .on::<Metal>()
    .sample_par(4);
  let _ = Fbs::<f32, _>::new(0.7, 17, 9, 1.0, Deterministic::new(19))
    .on::<Metal>()
    .sample();
  let device = std::rc::Rc::downgrade(&crate::device::metal::context(0).expect("a Metal device"));
  Metal::new(1).release();
  assert!(
    device.upgrade().is_some(),
    "another ordinal's release kept them"
  );
  Metal::default().release();
  assert!(device.upgrade().is_none(), "a cache still holds the device");
}

/// A release frees the compiled kernels and buffers; the next launch rebuilds them and draws the
/// same paths from the same seed.
#[cfg(feature = "cuda")]
#[test]
fn cuda_release_rebuilds_on_the_next_launch() {
  use crate::device::Cuda;

  let before = gbm::<f32>(9).on::<Cuda>().sample_par(8);
  Cuda::default().release();
  assert_eq!(before, gbm::<f32>(9).on::<Cuda>().sample_par(8));
}

mod philox;

mod stable;

mod vocabulary;
