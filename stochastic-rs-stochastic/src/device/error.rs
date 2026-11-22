use std::fmt;

/// Why a device could not serve a request: what [`Backend::probe`](super::Backend::probe) and the
/// `try_*` calls return, and the message the plain `sample*` calls panic with.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceError {
  /// No usable device behind the marker, or its runtime failed to initialise.
  Unavailable(String),
  /// The kernel library did not compile or load for this device.
  Compile(String),
  /// A kernel launch, allocation or copy failed at run time.
  Launch(String),
  /// The device ran out of memory: the one failure a caller can act on. The batch loops halve
  /// their chunk and retry, and a chunk is bit-identical however the batch is cut.
  OutOfMemory(String),
  /// An environment variable the handle reads holds a value it cannot use.
  Config(String),
}

impl fmt::Display for DeviceError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      DeviceError::Unavailable(msg) => write!(f, "device unavailable: {msg}"),
      DeviceError::Compile(msg) => write!(f, "kernel compilation failed: {msg}"),
      DeviceError::Launch(msg) => write!(f, "device operation failed: {msg}"),
      DeviceError::OutOfMemory(msg) => write!(f, "device out of memory: {msg}"),
      DeviceError::Config(msg) => write!(f, "device configuration: {msg}"),
    }
  }
}

impl DeviceError {
  /// Whether the device failed for want of memory, which a smaller chunk may
  /// survive. Every other failure is the same however the batch is cut.
  pub fn is_out_of_memory(&self) -> bool {
    matches!(self, DeviceError::OutOfMemory(_))
  }
}

impl std::error::Error for DeviceError {}

/// What [`Backend::probe`](super::Backend::probe) found behind a marker.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceInfo {
  /// The marker's name, e.g. `"Cuda"`.
  pub backend: &'static str,
  /// The device's own name, e.g. `"NVIDIA A100-SXM4-40GB"` or `"Apple M2 Max"`.
  pub name: String,
  /// Scalars the device computes in, e.g. `["f32"]` for an Apple GPU.
  pub precisions: &'static [&'static str],
  /// Device ordinal for back-ends that enumerate devices, `None` otherwise.
  pub ordinal: Option<usize>,
}

impl DeviceInfo {
  pub(crate) fn new(
    backend: &'static str,
    name: String,
    precisions: &'static [&'static str],
    ordinal: Option<usize>,
  ) -> Self {
    Self {
      backend,
      name,
      precisions,
      ordinal,
    }
  }

  pub(super) fn host(backend: &'static str, what: &str) -> Self {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    Self::new(
      backend,
      format!("{what}, {threads} threads"),
      &["f32", "f64"],
      None,
    )
  }
}
