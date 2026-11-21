use super::error::DeviceError;

/// Paths of `n` `elem`-byte scalars that fit `budget`, at least one.
pub(crate) fn chunk_rows(budget: usize, n: usize, elem: usize) -> usize {
  (budget / (n.max(1) * elem.max(1))).max(1)
}

/// Runs `chunk(first, len)` over `m` rows, halving the chunk on each out-of-memory answer down to
/// one row; safe because an engine chunk is bit-identical however the batch is cut.
pub(crate) fn over_chunks(
  m: usize,
  rows: usize,
  mut chunk: impl FnMut(usize, usize) -> Result<(), DeviceError>,
) -> Result<(), DeviceError> {
  let mut rows = rows.max(1);
  let mut first = 0;
  while first < m {
    let len = rows.min(m - first);
    match chunk(first, len) {
      Ok(()) => first += len,
      Err(e) if e.is_out_of_memory() && len > 1 => rows = len / 2,
      Err(e) => return Err(e),
    }
  }
  Ok(())
}

/// Caps a planner's rows so no chunk wraps a kernel index; the floor of one lets an oversized row
/// reach its launch, whose [`launch_len`] check refuses it rather than wrap.
#[cfg_attr(not(device_backend), allow(dead_code))]
pub(crate) fn rows_within_index_limit(rows: usize, elements_per_row: usize, limit: usize) -> usize {
  let per_row = elements_per_row.max(1);
  rows.min(limit / per_row).max(1)
}

/// `len` as the integer type a kernel takes it in, or the launch's error when
/// it does not fit — never the wrapped value an `as` cast would hand over.
#[cfg_attr(not(device_backend), allow(dead_code))]
pub(crate) fn launch_len<I: TryFrom<usize>>(len: usize, what: &str) -> Result<I, DeviceError> {
  I::try_from(len)
    .map_err(|_| DeviceError::Launch(format!("{what}: {len} exceeds the kernel index range")))
}

/// The cached state matching `matches`, moved to the most-recent slot, or a
/// freshly built one after evicting the least-recent when the cache is full.
#[cfg_attr(not(device_backend), allow(dead_code))]
pub(crate) fn lru_slot<C, E>(
  cache: &mut Vec<C>,
  matches: impl Fn(&C) -> bool,
  build: impl FnOnce() -> Result<C, E>,
) -> Result<&mut C, E> {
  if let Some(i) = cache.iter().position(&matches) {
    let hit = cache.remove(i);
    cache.push(hit);
  } else {
    // Build first: a failed build must not evict anything.
    let built = build()?;
    if cache.len() >= super::CACHE_SLOTS {
      cache.remove(0);
    }
    cache.push(built);
  }
  Ok(cache.last_mut().expect("the slot was just pushed"))
}

/// The panic a plain `sample*` call raises when its device fails.
pub(crate) fn device_panic<T>(e: DeviceError) -> T {
  panic!("{e}; probe the device handle with `Backend::probe(&device)` before sampling on it")
}
