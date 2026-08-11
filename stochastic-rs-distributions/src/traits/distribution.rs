//! Characteristic-function / pdf / cdf / moments interface and bulk samplers.

use ndarray::Array1;
use num_complex::Complex64;

/// Analytical descriptors of a distribution.
///
/// All methods are provided with default implementations that **panic** via
/// [`unimplemented!()`]. Implementors override the methods that have a known
/// closed form for that distribution. This is intentional: silently returning
/// zero (the previous default) masked missing implementations and produced
/// downstream numerical bugs in pricing / calibration code.
pub trait DistributionExt {
  fn characteristic_function(&self, _t: f64) -> Complex64 {
    unimplemented!(
      "DistributionExt::characteristic_function is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn pdf(&self, _x: f64) -> f64 {
    unimplemented!(
      "DistributionExt::pdf is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn cdf(&self, _x: f64) -> f64 {
    unimplemented!(
      "DistributionExt::cdf is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn inv_cdf(&self, _p: f64) -> f64 {
    unimplemented!(
      "DistributionExt::inv_cdf is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn mean(&self) -> f64 {
    unimplemented!(
      "DistributionExt::mean is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn median(&self) -> f64 {
    unimplemented!(
      "DistributionExt::median is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn mode(&self) -> f64 {
    unimplemented!(
      "DistributionExt::mode is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn variance(&self) -> f64 {
    unimplemented!(
      "DistributionExt::variance is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn skewness(&self) -> f64 {
    unimplemented!(
      "DistributionExt::skewness is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn kurtosis(&self) -> f64 {
    unimplemented!(
      "DistributionExt::kurtosis is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn entropy(&self) -> f64 {
    unimplemented!(
      "DistributionExt::entropy is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }

  fn moment_generating_function(&self, _t: f64) -> f64 {
    unimplemented!(
      "DistributionExt::moment_generating_function is not implemented for {}",
      std::any::type_name::<Self>()
    )
  }
}

/// Target number of `T` elements per worker in
/// [`DistributionSampler::sample_matrix`]'s chunked fan-out. Forking a
/// worker stream (one [`fork`](DistributionSampler::fork) call) and
/// dispatching one rayon task are both low-hundreds-of-nanoseconds costs;
/// 16Ki elements per worker keeps that overhead a small fraction of a
/// worker's own fill work for any matrix worth parallelizing over, while
/// scaling worker count — and hence the exposed parallelism — linearly with
/// the matrix size instead of pinning it to the machine's core count.
const MIN_PAR_CHUNK: usize = 16 * 1024;

/// Number of workers to split `total` output elements across in
/// [`DistributionSampler::sample_matrix`].
///
/// A pure function of `total` alone. **Must never read
/// `rayon::current_num_threads()`**: the worker count fixes how many times
/// [`fork`](DistributionSampler::fork) is called before any worker starts
/// filling, which fixes how many times a
/// [`Deterministic`](crate::simd_rng::Deterministic) sampler's live fork
/// state advances. If that count depended on the ambient thread-pool size,
/// the same seed and the same `(m, n)` could produce different
/// `sample_matrix` output on two machines (or two test runs) with different
/// pool sizes — exactly the defect this function fixes.
fn worker_count(total: usize) -> usize {
  total.div_ceil(MIN_PAR_CHUNK).max(1).min(total)
}

/// Rust-side bulk sampling API for distribution structs.
///
/// Implementors provide `fill_slice`; `sample_n` and `sample_matrix` are
/// lock-free convenience methods that allocate and fill contiguous buffers.
///
/// There is no `fill_slice(rng, out)` overload: per the crate's RNG policy,
/// the internal seeded stream is the only stream a sampler draws from.
/// Construct with `Deterministic::new(seed)` for reproducible output; `Self`
/// stores everything sampling needs, so these methods take no `Rng` of
/// their own.
pub trait DistributionSampler<T> {
  /// Fills `out` by drawing from this sampler's own internal SIMD RNG
  /// stream, seeded at construction.
  fn fill_slice(&self, out: &mut [T]);

  /// Builds an independent worker stream for the `stream_idx`-th chunk of a
  /// parallel `sample_matrix` fan-out.
  ///
  /// Implementors hold their fork anchor in an interior-mutable cell that
  /// this method reads *and advances* (via
  /// [`derive_seed`](crate::simd_rng::derive_seed)) before deriving the
  /// child seed — so `stream_idx` alone does not determine the output:
  /// which *call* to `sample_matrix` this is matters too. See
  /// [`sample_matrix`](Self::sample_matrix) for the resulting cross-call
  /// semantics.
  ///
  /// [`Deterministic`]: crate::simd_rng::Deterministic
  /// [`Unseeded`]: crate::simd_rng::Unseeded
  #[doc(hidden)]
  fn fork(&self, stream_idx: u64) -> Self;

  #[inline]
  fn sample_n(&self, n: usize) -> Array1<T> {
    let mut out = Array1::<T>::uninit(n);
    let flat_uninit = out
      .as_slice_mut()
      .expect("distribution sample_n output must be contiguous");
    let flat = unsafe {
      // SAFETY: `flat_uninit` points to the output storage and `fill_slice`
      // fully initializes every element before `assume_init` below.
      std::slice::from_raw_parts_mut(flat_uninit.as_mut_ptr().cast::<T>(), flat_uninit.len())
    };
    self.fill_slice(flat);
    unsafe {
      // SAFETY: all elements were initialized by `fill_slice` above.
      out.assume_init()
    }
  }

  /// Fills an `m × n` matrix, splitting the fill across a rayon `scope`
  /// when the workload is large enough to amortise the fork cost (below
  /// that threshold this runs the same single-threaded path as
  /// [`fill_slice`](Self::fill_slice)).
  ///
  /// **Parallel-fork semantics.** Each worker gets its own fresh basis:
  /// the parallel path calls [`fork`](Self::fork) once per worker
  /// (`stream_idx = 0..workers`), sequentially on the caller thread and
  /// before any worker starts filling, and every `fork` call reads *and
  /// advances* this object's live state (a private cell distinct from the
  /// stream that drives real samples) — so worker 0, worker 1, … each draw
  /// their own *different* basis value from that live state, combined with
  /// their own `stream_idx` via `derive_fork_seed(basis, stream_idx)`. It is
  /// one fresh basis per worker, never one basis drawn once per call and
  /// fanned out across workers by index. The worker count itself is a
  /// pure function of `m * n` — never of `rayon::current_num_threads()` —
  /// so how many times `fork` is called depends only on the matrix size.
  /// Consequences:
  /// - Two [`Deterministic`]-seeded objects constructed from the same seed
  ///   produce bit-identical output call-for-call: the *first*
  ///   `sample_matrix` call on each agrees, the *second* call on each
  ///   agrees, and so on — because their live states advance through the
  ///   same sequence of `fork` calls in lockstep, on any machine and under
  ///   any rayon thread-pool size.
  /// - Repeated calls on the *same* object never replay: the live state
  ///   advances every time the parallel path runs, for both
  ///   [`Deterministic`]- and [`Unseeded`]-constructed objects.
  /// - A serial call (small `m * n`) does not touch the fork state at all,
  ///   so interleaving serial and parallel calls stays deterministic
  ///   across two identically-seeded objects.
  ///
  /// [`Deterministic`]: crate::simd_rng::Deterministic
  /// [`Unseeded`]: crate::simd_rng::Unseeded
  #[inline]
  fn sample_matrix(&self, m: usize, n: usize) -> ndarray::Array2<T>
  where
    Self: Sized + Send,
    T: Send,
  {
    // `Simd*` samplers own an `UnsafeCell` buffer and are deliberately
    // `!Sync` (see the crate-level RNG policy), so the parallel branch
    // below must never capture `&self` inside the `rayon::scope` closure —
    // that would need `Self: Sync`, which no `Simd*` type can satisfy.
    // Every worker is forked to an owned value on this thread first, and
    // only owned `Self` values (already `Send`) cross into the closure.
    let mut out = ndarray::Array2::<T>::uninit((m, n));
    if m == 0 || n == 0 {
      return unsafe {
        // SAFETY: zero-length arrays have no elements to initialize.
        out.assume_init()
      };
    }
    let flat_uninit = out
      .as_slice_mut()
      .expect("distribution sample_matrix output must be contiguous");
    let flat = unsafe {
      // SAFETY: `flat_uninit` points to the output storage and each element
      // is initialized exactly once by the serial or parallel fill below.
      std::slice::from_raw_parts_mut(flat_uninit.as_mut_ptr().cast::<T>(), flat_uninit.len())
    };
    let total = flat.len();
    let workers = worker_count(total);
    if workers == 1 {
      self.fill_slice(flat);
      return unsafe {
        // SAFETY: all elements were initialized by `fill_slice`.
        out.assume_init()
      };
    }
    let chunk_len = total.div_ceil(workers);
    let forked_workers = (0..workers)
      .map(|stream_idx| self.fork(stream_idx as u64))
      .collect::<Vec<_>>();

    rayon::scope(move |scope| {
      for (worker, chunk) in forked_workers.into_iter().zip(flat.chunks_mut(chunk_len)) {
        scope.spawn(move |_| {
          worker.fill_slice(chunk);
        });
      }
    });
    unsafe {
      // SAFETY: every chunk is fully initialized by its worker.
      out.assume_init()
    }
  }
}
