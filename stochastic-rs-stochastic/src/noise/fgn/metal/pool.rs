use std::cell::RefCell;
use std::rc::Rc;

use metal_oxide::Buffer;

use super::{Launch, Result, context};
use crate::device::metal::{Context as DeviceContext, device_error};

struct Output {
  context: Rc<DeviceContext>,
  buffer: Buffer<f32>,
}

thread_local! {
  static OUT_POOL: RefCell<Vec<Output>> = const { RefCell::new(Vec::new()) };
}

pub(crate) struct OutputRental(Option<Output>);

impl OutputRental {
  fn new(context: Rc<DeviceContext>, len: usize) -> Result<Self> {
    let cached = OUT_POOL.with_borrow_mut(|pool| {
      pool
        .iter()
        .position(|out| out.buffer.len() >= len && Rc::ptr_eq(&out.context, &context))
        .map(|index| pool.remove(index))
    });
    let output = match cached {
      Some(output) => output,
      None => Output {
        buffer: context
          .device
          .buffer_zeroed::<f32>(len)
          .map_err(device_error)?,
        context,
      },
    };
    Ok(Self(Some(output)))
  }

  pub(crate) fn as_buffer(&self) -> &Buffer<f32> {
    &self.0.as_ref().unwrap().buffer
  }

  fn as_mut_buffer(&mut self) -> &mut Buffer<f32> {
    &mut self.0.as_mut().unwrap().buffer
  }
}

impl Drop for OutputRental {
  fn drop(&mut self) {
    let Some(output) = self.0.take() else {
      return;
    };
    // A rental may outlive the pool during thread-local destruction.
    let _ = OUT_POOL.try_with(|pool| {
      let Ok(mut pool) = pool.try_borrow_mut() else {
        return;
      };
      let index = pool
        .iter()
        .position(|cached| cached.buffer.len() < output.buffer.len())
        .unwrap_or(pool.len());
      pool.insert(index, output);
      pool.truncate(crate::device::CACHE_SLOTS);
    });
  }
}

fn sample(launch: Launch<'_>) -> Result<(OutputRental, usize)> {
  let len = launch.out_len()?;
  let common = context::with(launch.ordinal, |ctx| Ok(Rc::clone(&ctx.common)))?;
  let mut output = OutputRental::new(common, len)?;
  let out_size = context::with(launch.ordinal, |ctx| {
    ctx.run(launch, output.as_mut_buffer())
  })?;
  Ok((output, out_size))
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
) -> Result<(OutputRental, usize)> {
  sample(Launch {
    sqrt_eigs,
    n,
    m,
    offset,
    hurst,
    t,
    seed,
    first_cell,
    ordinal,
  })
}

pub(super) fn with_f32_rows<R>(
  launch: Launch<'_>,
  consume: impl FnOnce(&[f32]) -> Result<R>,
) -> Result<R> {
  let (output, size) = sample(launch)?;
  consume(&output.as_buffer().as_slice()[..launch.m * size])
}
