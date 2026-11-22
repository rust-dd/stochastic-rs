use super::*;

/// What the driver reports about the compiled engine kernel: registers and local memory per
/// thread bound occupancy, and only the device can say them, so an example prints them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelProfile {
  /// Registers per thread.
  pub registers: i32,
  /// Local (per-thread, off-chip) memory in bytes.
  pub local_bytes: i32,
  /// Statically allocated shared memory in bytes.
  pub shared_bytes: i32,
  /// The largest block the driver will launch this kernel with.
  pub max_threads_per_block: i32,
  /// Concurrent blocks per multiprocessor at [`block`](Self::block) threads.
  pub blocks_per_multiprocessor: u32,
  /// The block size the occupancy above was computed for.
  pub block: u32,
  /// Threads one multiprocessor can hold resident, read from the device: Turing holds 1024 where
  /// most generations hold 2048, so a constant would report a T4's full multiprocessor half empty.
  pub threads_per_multiprocessor: i32,
}

impl KernelProfile {
  /// The percentage of a multiprocessor's thread slots the kernel fills at [`block`](Self::block)
  /// threads; at 100 the device cannot hold another warp of this kernel.
  pub fn occupancy_percent(&self) -> u32 {
    let slots = self.threads_per_multiprocessor.max(1) as u32;
    self.blocks_per_multiprocessor * self.block * 100 / slots
  }
}

/// [`KernelProfile`] of the kernel a plain diffusion compiles at `real`, the shape every
/// specialised launch is measured against, on the device at `ordinal` with `block` threads.
pub fn kernel_profile(ordinal: usize, real: &'static str, block: u32) -> Result<KernelProfile> {
  let shape = Shape::new(super::families::Family::GeometricBrownian, false, false);
  let Launcher { context, func, .. } = ensure_kernels(ordinal, shape, real)?;
  let attr = |what: &str, v: std::result::Result<i32, DriverError>| -> Result<i32> {
    v.map_err(|e| driver_error(what, e))
  };
  Ok(KernelProfile {
    registers: attr("num_regs", func.num_regs())?,
    local_bytes: attr("local_size_bytes", func.local_size_bytes())?,
    shared_bytes: attr("shared_size_bytes", func.shared_size_bytes())?,
    max_threads_per_block: attr("max_threads_per_block", func.max_threads_per_block())?,
    blocks_per_multiprocessor: func
      .occupancy_max_active_blocks_per_multiprocessor(block, 0, None)
      .map_err(|e| driver_error("occupancy", e))?,
    block,
    threads_per_multiprocessor: attr(
      "max_threads_per_multiprocessor",
      context.attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
      ),
    )?,
  })
}
