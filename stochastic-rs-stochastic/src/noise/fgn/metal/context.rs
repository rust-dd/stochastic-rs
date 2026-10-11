use std::cell::RefCell;

use metal::foreign_types::ForeignType;
use metal_oxide::{Argument, Buffer, Device, Dim3, DynamicLaunchConfig, Module, Pipeline};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};

use super::{DeviceError, Launch, Result, build_bit_reverse_table, launch_len, tile_plan};

pub(super) struct Context {
  pub(super) native: metal::Device,
  device: Device,
  tile: Pipeline,
  radix4: Pipeline,
  extract: Pipeline,
  single: Pipeline,
  sized: Vec<SizedMetal>,
}

struct SizedMetal {
  real: Buffer<f32>,
  imag: Buffer<f32>,
  eigs: Buffer<f32>,
  reverse: Buffer<u32>,
  n: usize,
  m: usize,
  offset: usize,
  parity: u32,
  hurst_bits: u64,
  t_bits: u64,
}

thread_local! {
  static CONTEXT: RefCell<Option<(usize, Context)>> = const { RefCell::new(None) };
}

pub(super) fn with<R>(
  ordinal: usize,
  use_context: impl FnOnce(&mut Context) -> Result<R>,
) -> Result<R> {
  CONTEXT.with_borrow_mut(|slot| {
    if slot.as_ref().is_none_or(|(index, _)| *index != ordinal) {
      *slot = Some((ordinal, Context::new(ordinal)?));
    }
    use_context(&mut slot.as_mut().unwrap().1)
  })
}

fn device_error(error: metal_oxide::Error) -> DeviceError {
  use metal_oxide::Error;
  match error {
    Error::DeviceUnavailable | Error::UnsupportedDevice(_) => {
      DeviceError::Unavailable(error.to_string())
    }
    Error::AllocationFailed { .. } | Error::BufferTooLarge { .. } | Error::OutOfMemory(_) => {
      DeviceError::OutOfMemory(error.to_string())
    }
    Error::Library(_) | Error::Pipeline(_) | Error::Artifact(_) | Error::KernelNotFound(_) => {
      DeviceError::Compile(error.to_string())
    }
    _ => DeviceError::Launch(error.to_string()),
  }
}

impl Context {
  fn new(ordinal: usize) -> Result<Self> {
    let native = crate::euler::metal::metal_device(ordinal)?;
    // SAFETY: metal::Device owns this live MTLDevice; retain adds independent ownership.
    let raw = unsafe { Retained::retain(native.as_ptr().cast::<ProtocolObject<dyn MTLDevice>>()) }
      .expect("owned Metal device");
    let device = Device::from_native(raw).map_err(device_error)?;
    let module = Module::from_bytes(
      &device,
      include_str!("artifact/manifest.json"),
      include_bytes!("artifact/kernels.metallib"),
    )
    .map_err(device_error)?;
    let pipeline = |name| Pipeline::new(&device, &module, name).map_err(device_error);
    Ok(Self {
      native,
      tile: pipeline("gen_tile_fft")?,
      radix4: pipeline("fft_radix4")?,
      extract: pipeline("fft_radix4_extract")?,
      single: pipeline("fft_butterfly_extract")?,
      device,
      sized: Vec::new(),
    })
  }

  pub(super) fn run(&mut self, launch: Launch<'_>, output: &metal::Buffer) -> Result<usize> {
    let Launch {
      sqrt_eigs,
      n,
      m,
      offset,
      hurst,
      t,
      seed,
      first_cell,
      ..
    } = launch;
    let traj_size = 2 * n;
    let out_size = n - offset;
    if m == 0 {
      return Ok(out_size);
    }
    let scale = (out_size.max(1) as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);
    let parity = ((first_cell / traj_size as u64) & 1) as u32;
    let base_cell = first_cell - u64::from(parity) * traj_size as u64;
    let transforms = (parity as usize + m).div_ceil(2);
    let total = launch_len::<u32>(transforms * traj_size, "real")?;
    launch_len::<u32>(m * out_size, "output")?;
    launch_len::<i32>(m, "rows")?;
    let size = launch_len::<u32>(out_size, "out_size")?;
    let rows = launch_len::<u32>(m, "rows")?;
    let buffers = crate::device::lru_slot(
      &mut self.sized,
      |s| {
        s.n == n
          && s.m == m
          && s.offset == offset
          && s.parity == parity
          && s.hurst_bits == hurst.to_bits()
          && s.t_bits == t.to_bits()
      },
      || -> Result<_> {
        Ok(SizedMetal {
          real: self
            .device
            .buffer_zeroed::<f32>(total as usize)
            .map_err(device_error)?,
          imag: self
            .device
            .buffer_zeroed::<f32>(total as usize)
            .map_err(device_error)?,
          eigs: self
            .device
            .buffer_from_slice(sqrt_eigs)
            .map_err(device_error)?,
          reverse: self
            .device
            .buffer_from_slice(&build_bit_reverse_table(traj_size))
            .map_err(device_error)?,
          n,
          m,
          offset,
          parity,
          hurst_bits: hurst.to_bits(),
          t_bits: t.to_bits(),
        })
      },
    )?;
    let n = launch_len::<u32>(traj_size, "traj_size")?;
    let log_n = n.trailing_zeros() as usize;
    let (tile_log, stages) = tile_plan(log_n, self.tile.max_threads_per_block());
    let tile = 1_u32 << tile_log;
    let SizedMetal {
      real,
      imag,
      eigs,
      reverse,
      ..
    } = buffers;
    // SAFETY: the caller owns the initialized output exclusively through this handoff.
    let mut out = unsafe { import_output(&self.device, output) }.map_err(device_error)?;
    // SAFETY: complete transforms partition into disjoint tiles/butterflies; every tile lane
    // reaches each barrier. The planner bounds all buffers and the readout guards partial rows.
    let result = unsafe {
      self.device.submit(|batch| {
        batch.launch(
          &self.tile,
          DynamicLaunchConfig::new(Dim3::x(total / tile), Dim3::x(tile / 2)),
          &[
            Argument::write(real),
            Argument::write(imag),
            Argument::read(eigs),
            Argument::read(reverse),
            Argument::value(n)?,
            Argument::value(seed)?,
            Argument::value(base_cell)?,
            Argument::value(stages as u32)?,
            Argument::threadgroup::<f32>(tile as usize)?,
            Argument::threadgroup::<f32>(tile as usize)?,
          ],
        )?;
        for stage in (stages..log_n.saturating_sub(2)).step_by(2) {
          let count = total / 4;
          batch.launch(
            &self.radix4,
            DynamicLaunchConfig::for_elements(count, 256)?,
            &[
              Argument::read_write(real),
              Argument::read_write(imag),
              Argument::value(n)?,
              Argument::value(stage as u32)?,
              Argument::value(count)?,
            ],
          )?;
        }
        let (pipeline, count) = if log_n - stages == 1 {
          (&self.single, total / 2)
        } else {
          (&self.extract, total / 4)
        };
        batch.launch(
          pipeline,
          DynamicLaunchConfig::for_elements(count, 256)?,
          &[
            Argument::read(real),
            Argument::read(imag),
            Argument::value(n)?,
            Argument::write(&mut out),
            Argument::value(size)?,
            Argument::value(scale)?,
            Argument::value(rows)?,
            Argument::value(parity)?,
            Argument::value(count)?,
          ],
        )
      })
    }
    .and_then(|submission| submission.wait());
    drop(out.into_native());
    result.map(|_| out_size).map_err(device_error)
  }
}

// SAFETY: the caller excludes external access until the consuming native handoff.
unsafe fn import_output(
  device: &Device,
  buffer: &metal::Buffer,
) -> metal_oxide::Result<Buffer<f32>> {
  // SAFETY: the owned buffer is initialized shared storage; native aliases remain idle.
  let raw = unsafe { Retained::retain(buffer.as_ptr().cast::<ProtocolObject<dyn MTLBuffer>>()) }
    .expect("owned Metal buffer");
  // SAFETY: preceding GPU work is complete; the caller keeps native aliases idle.
  unsafe { Buffer::from_native(device, raw, buffer.length() as usize / size_of::<f32>()) }
}
