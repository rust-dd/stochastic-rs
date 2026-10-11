use metal::Device;

use super::{context, *};

static OUT_POOL: Mutex<Vec<Buffer>> = Mutex::new(Vec::new());

fn take_out_buffer(dev: &Device, bytes: u64) -> Result<Buffer> {
  let mut pool = OUT_POOL.lock();
  match pool
    .iter()
    .position(|b| b.length() >= bytes && b.device().registry_id() == dev.registry_id())
  {
    Some(i) => Ok(pool.remove(i)),
    None => shared_buffer(dev, bytes),
  }
}

fn return_out_buffer(buf: Buffer) {
  let mut pool = OUT_POOL.lock();
  let at = pool
    .iter()
    .position(|b| b.length() < buf.length())
    .unwrap_or(pool.len());
  pool.insert(at, buf);
  pool.truncate(crate::device::CACHE_SLOTS);
}

std::thread_local! {
  // Every caller finishes consuming the buffer before requesting another on this thread.
  static LENT: std::cell::RefCell<Option<Buffer>> = const { std::cell::RefCell::new(None) };
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_f32_buffer(
  sqrt_eigs: &[f32],
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
  seed: u32,
  first_cell: u64,
  ordinal: usize,
) -> Result<(Buffer, usize)> {
  let launch = Launch {
    sqrt_eigs,
    n,
    m,
    offset,
    hurst,
    t,
    seed,
    first_cell,
    ordinal,
  };
  let dev = context::with(ordinal, |ctx| Ok(ctx.native.clone()))?;
  if let Some(done) = LENT.with_borrow_mut(Option::take) {
    return_out_buffer(done);
  }
  let out = take_out_buffer(&dev, (launch.out_len() * 4) as u64)?;
  let out_size = context::with(ordinal, |ctx| ctx.run(launch, &out))?;
  LENT.with_borrow_mut(|slot| *slot = Some(out.clone()));
  Ok((out, out_size))
}

pub(super) fn with_f32_rows<R>(
  launch: Launch<'_>,
  consume: impl FnOnce(&[f32]) -> Result<R>,
) -> Result<R> {
  let dev = context::with(launch.ordinal, |ctx| Ok(ctx.native.clone()))?;
  let len = launch.out_len();
  let out = take_out_buffer(&dev, (len * 4) as u64)?;
  let result = context::with(launch.ordinal, |ctx| ctx.run(launch, &out)).and_then(|_| {
    // SAFETY: shared storage the launch has finished writing into, of at
    // least `len` floats by construction, and held by nobody else.
    consume(unsafe { std::slice::from_raw_parts(out.contents() as *const f32, len) })
  });
  return_out_buffer(out);
  result
}
