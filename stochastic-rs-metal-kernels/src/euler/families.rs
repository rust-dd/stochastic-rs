//! The host and device consume these declaration tokens independently.

use metal_oxide_device::ReadBuffer;

#[macro_use]
mod codegen;

#[macro_use]
#[path = "../../../stochastic-rs-stochastic/src/euler/families/definitions/mod.rs"]
mod definitions;

#[derive(Clone, Copy)]
pub(crate) struct Inputs {
  pub dt: f32,
  pub curve: [f32; 8],
  pub count: f32,
  pub jump: f32,
  pub gamma: [f32; 2],
  pub uniform: [f32; 2],
  pub lift: f32,
  pub convolution: f32,
  pub series: f32,
  pub arrival: f32,
  pub exponential: f32,
  pub series_uniform: [f32; 2],
  pub inverse: f32,
  pub spacing: f32,
  pub program: [f32; 2],
  pub noise: [f32; 4],
}

pub(crate) trait Family {
  const COMPONENTS: u32;
  const NOISES: u32;
  const HAS_LIFT: bool;
  const HAS_HISTORY: bool;
  const HAS_SERIES: bool;
  const HAS_TABLE: bool;
  const HAS_PROGRAM: bool;
  const HAS_UNIFORMS: bool;
  const READS_COUNT: bool;
  const READS_JUMP: bool;
  const READS_GAMMA: bool;
  const LIFT_CAPACITY: usize = if Self::HAS_LIFT { 176 } else { 1 };
  const BLOCK_CAPACITY: usize = if Self::HAS_HISTORY || Self::HAS_SERIES || Self::HAS_TABLE {
    512
  } else {
    1
  };

  /// # Safety
  /// `params` contains the family's parameters in declaration order.
  unsafe fn step(params: ReadBuffer<f32>, state: [f32; 4], input: Inputs) -> [f32; 4];

  /// # Safety
  /// `params` contains the family's parameters in declaration order.
  unsafe fn report(params: ReadBuffer<f32>, state: [f32; 4], input: Inputs) -> [f32; 4];

  /// # Safety
  /// `params` contains the family's parameters in declaration order.
  unsafe fn lift(_params: ReadBuffer<f32>, _state: [f32; 4], _input: Inputs) -> [f32; 4] {
    [0.0; 4]
  }

  /// # Safety
  /// `params` contains the family's parameters in declaration order.
  unsafe fn history(_params: ReadBuffer<f32>, _state: [f32; 4], _input: Inputs) -> [f32; 4] {
    [0.0; 4]
  }

  /// # Safety
  /// `params` contains the family's parameters in declaration order.
  unsafe fn series(_params: ReadBuffer<f32>, _state: [f32; 4], _input: Inputs) -> [f32; 4] {
    [0.0; 4]
  }

  /// # Safety
  /// `params` contains the family's parameters in declaration order.
  unsafe fn table(_params: ReadBuffer<f32>, _state: [f32; 4], _input: Inputs) -> [f32; 4] {
    [0.0; 4]
  }
}

device_families!(definitions);
