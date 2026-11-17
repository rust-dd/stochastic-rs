use super::*;

/// What the driver reports about the compiled engine kernel.
///
/// Registers and local memory per thread are what bound occupancy, and a
/// kernel rendered for one launch shape is the engine's answer to both: it
/// carries that family's step alone and declares only the scratch it can
/// reach, where the monolithic body carried every family's step and every
/// optional frame block's scratch whether the launch used them or not. These
/// numbers say whether that worked on a given card, and they cannot be read
/// on a machine without the device — which is why they are printed by an
/// example rather than assumed.
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
  /// Threads one multiprocessor of this device can hold resident, read from
  /// the device rather than assumed.
  ///
  /// It is the denominator of [`occupancy_percent`](Self::occupancy_percent)
  /// and it is not a constant across architectures: Turing holds 1024 where
  /// most generations hold 2048, so a hard-coded 2048 reports a T4's full
  /// multiprocessor as half empty.
  pub threads_per_multiprocessor: i32,
}

impl KernelProfile {
  /// The share of a multiprocessor's thread slots the kernel fills at
  /// [`block`](Self::block) threads, in percent — 100 meaning the device
  /// cannot hold another warp of this kernel.
  pub fn occupancy_percent(&self) -> u32 {
    let slots = self.threads_per_multiprocessor.max(1) as u32;
    self.blocks_per_multiprocessor * self.block * 100 / slots
  }
}

/// [`KernelProfile`] for the kernel a plain diffusion compiles at `real` —
/// the shape every specialised launch is measured against — on the device at
/// `ordinal`, at a block size of `block` threads.
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
