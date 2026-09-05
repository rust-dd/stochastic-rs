use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;

use super::*;
use crate::diffusion::cir::Cir;
use crate::diffusion::gbm::Gbm;
use crate::diffusion::ou::Ou;

/// `on::<Cpu>()` is the identity on a process's output, and the `Cpu`
/// backend's `euler_paths` is the process's own `sample_par` — bit for bit.
#[test]
fn cpu_backend_is_the_process_sampler() {
  // `Deterministic` advances as it samples, so every comparison starts from a
  // freshly constructed process with the same seed.
  let gbm = || {
    Gbm::new(
      0.05,
      0.2,
      64,
      Some(100.0),
      Some(1.0),
      Deterministic::new(42),
    )
  };
  let plain = gbm().sample_par(16);
  let switched = gbm().on::<Cpu>().sample_par(16);
  let through_trait = Cpu.euler_paths(&gbm(), 16);
  assert_eq!(plain.len(), 16);
  for i in 0..16 {
    assert_eq!(plain[i].to_vec(), switched[i].to_vec());
    assert_eq!(plain[i].to_vec(), through_trait[i].to_vec());
  }

  const {}
  let ou = || {
    Ou::new(
      2.0,
      1.0,
      0.5,
      33,
      Some(1.0),
      Some(2.0),
      Deterministic::new(7),
    )
  };
  assert_eq!(ou().on::<Cpu>().sample().to_vec(), ou().sample().to_vec());
  let cir = || {
    Cir::new(
      1.5,
      0.04,
      0.3,
      33,
      Some(0.09),
      Some(1.0),
      None,
      Deterministic::new(7),
    )
  };
  assert_eq!(
    cir().on::<Cpu>().sample_map(4, |p| p[32]),
    cir().sample_map(4, |p| p[32])
  );
  assert_eq!(gbm().on::<Cpu>().sample_par(0).len(), 0);
}

#[test]
fn switching_the_backend_keeps_the_parameters() {
  let gbm = Gbm::new(0.05, 0.2, 64, Some(100.0), Some(2.0), Unseeded).on::<Cpu>();
  assert_eq!(
    (gbm.mu, gbm.sigma, gbm.n, gbm.x0, gbm.t),
    (0.05, 0.2, 64, Some(100.0), Some(2.0))
  );
  let cir = Cir::new(
    1.5,
    0.04,
    0.3,
    10,
    Some(0.09),
    Some(2.0),
    Some(true),
    Unseeded,
  )
  .on::<Cpu>();
  assert_eq!(cir.use_sym, Some(true));
  assert_eq!(cir.euler_spec().encode().0, 2);
  assert_eq!(
    (cir.initial_value(), cir.grid_points(), cir.horizon()),
    (0.09, 10, 2.0)
  );
  assert_eq!(gbm.euler_spec().encode().0, 0);
  let (code, params) = Ou::new(2.0, 1.0, 0.5, 5, None, None, Unseeded)
    .euler_spec()
    .encode();
  assert_eq!(code, 1);
  assert_eq!(&params[..3], &[2.0, 1.0, 0.5]);
  assert!(
    params[3..].iter().all(|p| *p == 0.0),
    "unused slots are zero"
  );
}

#[test]
fn try_sample_par_matches_sample_par_on_the_cpu() {
  let gbm = || Gbm::new(0.05, 0.2, 64, Some(100.0), Some(1.0), Deterministic::new(9));
  assert_eq!(gbm().try_sample_par(8).expect("cpu"), gbm().sample_par(8));
  let ou = || {
    Ou::new(
      2.0,
      1.0,
      0.5,
      64,
      Some(1.0),
      Some(1.0),
      Deterministic::new(9),
    )
  };
  assert_eq!(ou().try_sample_par(8).expect("cpu"), ou().sample_par(8));
  let cir = || {
    Cir::new(
      1.5,
      0.04,
      0.3,
      64,
      Some(0.09),
      Some(1.0),
      None,
      Deterministic::new(9),
    )
  };
  assert_eq!(cir().try_sample_par(8).expect("cpu"), cir().sample_par(8));
  let fgn = || crate::noise::fgn::Fgn::<f64, _>::new(0.7, 64, Some(1.0), Deterministic::new(9));
  assert_eq!(fgn().try_sample_par(4).expect("cpu"), fgn().sample_par(4));
}

#[test]
fn host_map_and_chunk_calls_match_the_process_sampler() {
  let gbm = || Gbm::new(0.05, 0.2, 32, Some(100.0), Some(1.0), Deterministic::new(4));
  let direct: Vec<f64> = gbm().sample_map(6, |p| p[31]);
  let through: Vec<f64> = Cpu.try_euler_paths_map(&gbm(), 6, |p| p[31]).expect("cpu");
  assert_eq!(direct, through);
  assert_eq!(
    Cpu.try_euler_paths(&gbm(), 6).expect("cpu"),
    gbm().sample_par(6),
    "the host stream is the process's own"
  );
}

#[test]
fn matrix_form_matches_the_rows() {
  let gbm = || Gbm::new(0.05, 0.2, 32, Some(100.0), Some(1.0), Deterministic::new(6));
  let rows = gbm().sample_par(5);
  let matrix = gbm().try_sample_matrix(5).expect("cpu");
  assert_eq!(matrix.dim(), (5, 32));
  for (i, row) in rows.iter().enumerate() {
    assert_eq!(matrix.row(i), row.view());
  }
}

#[test]
fn device_seed_follows_the_seed_source() {
  let a = Gbm::new(0.05, 0.2, 8, None, None, Deterministic::new(3));
  let b = Gbm::new(0.05, 0.2, 8, None, None, Deterministic::new(3));
  assert_eq!(a.device_seed(), b.device_seed());
  assert_ne!(
    a.device_seed(),
    Gbm::new(0.05, 0.2, 8, None, None, Deterministic::new(4)).device_seed()
  );
  let u = Gbm::new(0.05, 0.2, 8, None, None, Unseeded);
  assert_ne!(u.device_seed(), u.device_seed());
}

/// Every compiled device back-end reproduces the lognormal moments and is
/// deterministic in its seed; the device kernels share one integer hash for
/// their uniforms, so two device back-ends agree seed for seed up to the
/// `f32` libm rounding of Box–Muller.
#[cfg(any(
  feature = "metal",
  feature = "cuda",
  feature = "cubecl-cuda",
  feature = "cubecl-wgpu"
))]
mod devices {
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

  #[cfg(any(feature = "cubecl-cuda", feature = "cubecl-wgpu"))]
  #[test]
  fn cubecl_backend_matches_the_moments() {
    #[cfg(feature = "cubecl-wgpu")]
    type Rt = crate::device::WgpuRuntime;
    #[cfg(all(feature = "cubecl-cuda", not(feature = "cubecl-wgpu")))]
    type Rt = crate::device::CudaRuntime;
    gbm_moments_hold::<f32, crate::device::Cubecl<Rt>>("Cubecl");
    cir_stays_nonnegative::<f32, crate::device::Cubecl<Rt>>("Cubecl");
  }

  #[cfg(feature = "metal")]
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
  #[cfg(feature = "metal")]
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
    let cfbms =
      || Cfbms::<f32, _>::new(0.7, 0.0, N, Some(1.0), Deterministic::new(5)).on::<Metal>();
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

    let cfgns =
      || Cfgns::<f32, _>::new(0.7, 0.0, N, Some(1.0), Deterministic::new(5)).on::<Metal>();
    assert_eq!(
      cfgns().sample_par(M),
      small.system_paths(&cfgns(), M),
      "correlated fGn: chunking moved the batch"
    );
  }

  #[cfg(feature = "metal")]
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

  #[cfg(feature = "metal")]
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

  #[cfg(feature = "metal")]
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
}
