//! Rust-compiled Euler paths with device-resident fractional increments.

mod context;
mod launch;
mod sampling;

use metal_oxide::Buffer;

use crate::device::DeviceError;
use launch::{device_paths, launch_paths};

type Result<T> = std::result::Result<T, DeviceError>;

enum Increments<'a> {
  Hashed,
  /// `streams` adjacent rows of `row` increments per path.
  Device {
    buffer: &'a Buffer<f32>,
    streams: u32,
    row: usize,
  },
}
