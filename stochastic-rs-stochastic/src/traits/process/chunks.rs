use super::*;

/// Upper bound on the chunks of `sample_par` / `sample_map`: each costs one sequential `sampler()`
/// call, which is not cheap for `Cir2F` with a Python `Fn1D`; 64 still saturates any rayon pool.
const MAX_CHUNKS: usize = 64;

/// Number of chunks for `m` paths, a function of `m` alone: reading `rayon::current_num_threads()`
/// would make one seed give different output under different pools. `Accelerate`'s fGN reuses it.
pub(crate) fn chunk_count(m: usize) -> usize {
  m.min(MAX_CHUNKS)
}

/// Splits `m` into `chunks` contiguous lengths in chunk order, the first `m % chunks` one longer.
/// Total: `chunks == 0`, which only `chunk_count(0)` gives, yields nothing and divides nothing.
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
    // An indexed parallel iterator collects in chunk order whatever the completion order, so the
    // flattened result is the global path order.
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

/// [`sample_par_chunked`] through `try_sample`: the same chunks and samplers in the same order, so
/// a host process yields bit-identical paths and a device process reports its failure.
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
