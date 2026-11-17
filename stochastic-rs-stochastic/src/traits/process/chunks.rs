use super::*;

/// Upper bound on the number of chunks [`ProcessExt::sample_par`] /
/// [`ProcessExt::sample_map`] split `m` paths into.
///
/// `chunk_count`'s prologue (building every chunk's sampler, sequentially,
/// on the calling thread — see [`ProcessExt::chunked_samplers`]) costs at
/// most `MAX_CHUNKS` sampler constructions, however large `m` is. That
/// bound matters because `sampler()` is not always cheap: the worst in-tree
/// case (`Cir2F`, whose `sampler()` evaluates a user-supplied `Fn1D` — a
/// Python callback, for the `Fn1D::Py` variant — once per grid point) would
/// otherwise pay a *sequential, GIL-round-tripping* construction cost that
/// grows with `m` (the previous rule, `m.div_ceil(8)`, made `sample_par(1000)`
/// build 125 chunks — 125 sequential constructions — instead of a number
/// bounded by the machine's core count). 64 is comfortably above real-world
/// hardware-thread counts, so `m >= 64` still saturates any realistic rayon
/// pool, while `m < 64` gets one chunk per path (full parallelism for small
/// `m`, the other end `m.div_ceil(8)` got wrong by forcing `m <= 8` fully
/// serial).
const MAX_CHUNKS: usize = 64;

/// Number of chunks to split `m` paths into.
///
/// A pure function of `m` alone. **Must never read
/// `rayon::current_num_threads()`**: the chunk count fixes how many times
/// [`sampler()`](ProcessExt::sampler) (and
/// [`advance_chunk_seed()`](ProcessExt::advance_chunk_seed)) is called
/// before any chunk starts running, which fixes how many times a
/// [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic) process's
/// shared seed state advances. If that count depended on the ambient
/// thread-pool size, the same seed and the same `m` could produce different
/// output on two machines (or two test runs) with different pool sizes —
/// exactly the defect this module fixes.
///
/// `pub(crate)` rather than private: the `accelerate` feature's `Backend`
/// impl (`device/fgn_backend.rs`) reuses it verbatim for `Fgn`/`Fbm`'s own `sample_par`
/// overrides on that backend, which bypass this trait to reach the batched
/// backend path but must split `m` the same data-derived way to get the
/// same thread-count-independence guarantee. The default `Cpu` backend
/// does not reuse it — measurement showed grouping several paths per
/// `MAX_CHUNKS`-capped chunk regressed wall time roughly 2× at `m = 1000`
/// there, because each path's own FFT call is itself a nested rayon
/// parallel region (see `device/fgn_backend.rs`'s `Cpu::generate_batch` doc), so `Cpu`
/// instead derives one basis per **path**, uncapped.
pub(crate) fn chunk_count(m: usize) -> usize {
  m.min(MAX_CHUNKS)
}

/// Splits `m` into `chunks` contiguous run lengths, as even as possible (the
/// first `m % chunks` chunks get one extra path), yielded in chunk order.
///
/// `chunks == 0` only ever arises from `chunk_count(0)`; the `checked_div`/
/// `checked_rem` fall back to `0` there (rather than relying on every caller
/// to check `m` first) so this function stays total instead of panicking on
/// the `m / 0` that a plain division would perform. `pub(crate)` for the
/// same reason as `chunk_count` above.
pub(crate) fn chunk_lens(m: usize, chunks: usize) -> impl Iterator<Item = usize> {
  let base = m.checked_div(chunks).unwrap_or(0);
  let rem = m.checked_rem(chunks).unwrap_or(0);
  (0..chunks).map(move |i| base + usize::from(i < rem))
}

/// The trait's default `sample_map`: chunked host samplers on rayon, chunk order
/// preserved. A backend-parameterised process calls this on its host path.
pub(crate) fn sample_map_chunked<T: FloatExt, P: ProcessExt<T> + ?Sized, R: Send>(
  p: &P,
  m: usize,
  f: impl Fn(&P::Output) -> R + Sync,
) -> Vec<R> {
  if m == 0 {
    return Vec::new();
  }
  if m == 1 {
    return vec![f(&p.sample())];
  }
  p.chunked_samplers(m)
    .into_par_iter()
    .map(|(mut sampler, len)| {
      let mut slot: Option<P::Output> = None;
      (0..len)
        .map(|_| {
          if let Some(buf) = slot.as_mut() {
            sampler.sample_into(buf);
            return f(buf);
          }
          // First path in this chunk: sample fresh (no wasted draw) and
          // keep the buffer to reuse for the rest of the chunk.
          let buf = sampler.sample();
          let r = f(&buf);
          slot = Some(buf);
          r
        })
        .collect::<Vec<_>>()
    })
    // Chunks run on rayon, but `Vec::into_par_iter()` → `.map()` is an
    // `IndexedParallelIterator`, so `.collect()` restores chunk order
    // regardless of completion order; flattening then reproduces the
    // exact global path order (chunk 0's paths, then chunk 1's, ...).
    .collect::<Vec<_>>()
    .into_iter()
    .flatten()
    .collect()
}

/// The trait's default `sample_par`: chunked host samplers on rayon, chunk order
/// preserved. A backend-parameterised process calls this on its host path.
pub(crate) fn sample_par_chunked<T: FloatExt, P: ProcessExt<T> + ?Sized>(
  p: &P,
  m: usize,
) -> Vec<P::Output> {
  if m == 0 {
    return Vec::new();
  }
  if m == 1 {
    return vec![p.sample()];
  }
  p.chunked_samplers(m)
    .into_par_iter()
    .map(|(mut sampler, len)| (0..len).map(|_| sampler.sample()).collect::<Vec<_>>())
    // Same order-preserving collect-then-flatten as `sample_map` above.
    .collect::<Vec<_>>()
    .into_iter()
    .flatten()
    .collect()
}

/// [`sample_par_chunked`] through the sampler's fallible `try_sample`: the
/// same chunks, the same per-chunk samplers in the same order, so a host
/// process yields bit-identical paths and a device process reports its
/// failure instead of panicking.
pub(crate) fn try_sample_par_chunked<T: FloatExt, P: ProcessExt<T> + ?Sized>(
  p: &P,
  m: usize,
) -> Result<Vec<P::Output>, DeviceError> {
  if m == 0 {
    return Ok(Vec::new());
  }
  if m == 1 {
    return Ok(vec![p.try_sample()?]);
  }
  let chunks = p
    .chunked_samplers(m)
    .into_par_iter()
    .map(|(mut sampler, len)| {
      (0..len)
        .map(|_| sampler.try_sample())
        .collect::<Result<Vec<_>, DeviceError>>()
    })
    .collect::<Result<Vec<_>, DeviceError>>()?;
  Ok(chunks.into_iter().flatten().collect())
}
