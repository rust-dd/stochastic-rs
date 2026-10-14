use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use metal_oxide::{Argument, Buffer, DynamicLaunchConfig, Launch, Operation, Pipeline};

use super::{Increments, Result};
use crate::device::metal::{bindings::EulerArgs, device_error};
use crate::device::{launch_len, metal};
use crate::euler::families::Family;

struct Context {
  shared: Rc<metal::Context>,
  pipelines: HashMap<(u32, bool, bool), Pipeline>,
  output: Option<Buffer<f32>>,
  params: Buffer<f32>,
  curves: Buffer<f32>,
  lift: [Buffer<f32>; 3],
  program: Buffer<f32>,
  empty: Buffer<f32>,
}

thread_local! {
  static CONTEXT: RefCell<Option<(usize, Context)>> = const { RefCell::new(None) };
}

impl Context {
  fn new(ordinal: usize) -> Result<Self> {
    let shared = metal::context(ordinal)?;
    let floats = |len| {
      shared
        .device
        .buffer_zeroed::<f32>(len)
        .map_err(device_error)
    };
    Ok(Self {
      pipelines: HashMap::new(),
      output: None,
      params: floats(crate::euler::PARAM_SLOTS)?,
      curves: floats(1)?,
      lift: [
        floats(crate::euler::LIFT_SLOTS)?,
        floats(crate::euler::LIFT_SLOTS)?,
        floats(crate::euler::LIFT_SLOTS)?,
      ],
      program: floats(1)?,
      empty: floats(1)?,
      shared,
    })
  }

  #[allow(clippy::too_many_arguments)]
  fn launch(
    &mut self,
    params: [f32; crate::euler::PARAM_SLOTS],
    args: EulerArgs,
    increments: Increments<'_>,
    curves: &[f32],
    lift: [&[f32]; 3],
    program: &[f32],
    total: usize,
  ) -> Result<Buffer<f32>> {
    let shape = kernel(
      args.family,
      args.has_jumps != 0 || args.jump_law != 0,
      args.gamma_law != 0,
    );
    if let std::collections::hash_map::Entry::Vacant(slot) = self.pipelines.entry(shape) {
      slot.insert(
        Pipeline::new(
          &self.shared.device,
          &self.shared.module,
          &kernel_name(shape),
        )
        .map_err(device_error)?,
      );
    }
    self.params.write().wait().copy_from_slice(&params);
    update(&self.shared, &mut self.curves, curves)?;
    for (buffer, values) in self.lift.iter_mut().zip(lift) {
      update(&self.shared, buffer, values)?;
    }
    update(&self.shared, &mut self.program, program)?;
    if self
      .output
      .as_ref()
      .is_none_or(|buffer| buffer.len() < total)
    {
      self.output = Some(
        self
          .shared
          .device
          .buffer_zeroed::<f32>(total)
          .map_err(device_error)?,
      );
    }
    let mut output = self.output.take().expect("sized output");
    let increments = match increments {
      Increments::Hashed => &self.empty,
      Increments::Device { buffer, .. } => buffer,
    };
    let pipeline = &self.pipelines[&shape];
    let width = pipeline.max_threads_per_block().min(256) as u32;
    let config = DynamicLaunchConfig::for_elements(args.paths, width).map_err(device_error)?;
    // SAFETY: launch_paths validates shape counts and every input's extent. Each
    // guarded thread owns one complete path in every output plane. All buffers
    // belong to the same ordered context and no CPU view survives the launch.
    unsafe {
      Launch::new(
        Ok(pipeline),
        config,
        (
          &mut output,
          &self.params,
          args,
          increments,
          &self.curves,
          &self.lift,
          &self.program,
        ),
        |(output, params, args, increments, curves, lift, program)| {
          Ok([
            Argument::write(output),
            Argument::read(params),
            Argument::value(*args)?,
            Argument::read(increments),
            Argument::read(curves),
            Argument::read(&lift[0]),
            Argument::read(&lift[1]),
            Argument::read(&lift[2]),
            Argument::read(program),
          ])
        },
      )
      .wait()
    }
    .map_err(device_error)?;
    Ok(output)
  }
}

/// Launch shapes that differ only in draws the family never reads share one kernel.
fn kernel(family: u32, jumps: bool, gamma: bool) -> (u32, bool, bool) {
  let (reads_jumps, reads_gamma) = Family::from_code(family)
    .expect("a declared family")
    .draws();
  (family, jumps && reads_jumps, gamma && reads_gamma)
}

fn kernel_name((family, jumps, gamma): (u32, bool, bool)) -> String {
  format!("euler_{family}_{}_{}", u8::from(jumps), u8::from(gamma))
}

fn update(context: &metal::Context, buffer: &mut Buffer<f32>, values: &[f32]) -> Result<()> {
  if buffer.len() < values.len() {
    *buffer = context
      .device
      .buffer_from_slice(values)
      .map_err(device_error)?;
  } else {
    buffer.write().wait()[..values.len()].copy_from_slice(values);
  }
  Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run<O>(
  ordinal: usize,
  params: [f32; crate::euler::PARAM_SLOTS],
  args: EulerArgs,
  increments: Increments<'_>,
  curves: &[f32],
  lift: [&[f32]; 3],
  program: &[f32],
  finish: impl FnOnce(&[f32]) -> O,
) -> Result<O> {
  let cells = args.paths as usize * args.steps as usize;
  let total = cells
    .checked_mul(args.components as usize)
    .ok_or_else(|| crate::device::DeviceError::Launch("Euler output dimensions overflow".into()))?;
  launch_len::<u32>(total, "out")?;
  let inc_len = cells.checked_mul(args.increments as usize).ok_or_else(|| {
    crate::device::DeviceError::Launch("Euler increment dimensions overflow".into())
  })?;
  launch_len::<u32>(inc_len, "incs")?;
  let output = CONTEXT.with_borrow_mut(|slot| -> Result<_> {
    if slot.as_ref().is_none_or(|(index, _)| *index != ordinal) {
      *slot = Some((ordinal, Context::new(ordinal)?));
    }
    slot
      .as_mut()
      .unwrap()
      .1
      .launch(params, args, increments, curves, lift, program, total)
  })?;
  let result = finish(&output.read().wait()[..total]);
  CONTEXT.with_borrow_mut(|slot| {
    if let Some((index, context)) = slot
      && *index == ordinal
      && context
        .output
        .as_ref()
        .is_none_or(|buffer| buffer.len() < output.len())
    {
      context.output = Some(output);
    }
  });
  Ok(result)
}

#[cfg(test)]
mod tests {
  use std::collections::BTreeSet;

  use super::{Family, kernel, kernel_name};

  #[test]
  fn launch_shapes_name_exactly_the_embedded_euler_kernels() {
    let manifest = include_str!("../../../metal/src/device/metal.rs/bindings-fast/manifest.json");
    let embedded = manifest
      .split("\"name\": \"")
      .skip(1)
      .filter_map(|rest| rest.split('"').next())
      .filter(|name| name.starts_with("euler_"))
      .map(str::to_owned)
      .collect::<BTreeSet<_>>();
    let launched = Family::ALL
      .iter()
      .flat_map(|family| {
        [(false, false), (false, true), (true, false), (true, true)]
          .map(|(jumps, gamma)| kernel_name(kernel(family.code(), jumps, gamma)))
      })
      .collect::<BTreeSet<_>>();
    assert_eq!(launched, embedded);
  }
}
