use std::cell::RefCell;
use std::rc::Rc;

use metal_oxide::{Device, Module};

use super::{DeviceError, DeviceInfo};

pub(crate) mod bindings {
  include!("metal/artifact/bindings.rs");
}

type Result<T> = std::result::Result<T, DeviceError>;

pub(crate) struct Context {
  pub(crate) device: Device,
  pub(crate) module: Module,
}

thread_local! {
  static CONTEXTS: RefCell<Vec<(usize, Rc<Context>)>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn context(ordinal: usize) -> Result<Rc<Context>> {
  CONTEXTS.with_borrow_mut(|contexts| {
    if let Some((_, context)) = contexts.iter().find(|(index, _)| *index == ordinal) {
      return Ok(Rc::clone(context));
    }
    let device = create_device(ordinal)?;
    let module = Module::from_bytes(
      &device,
      include_str!("metal/artifact/manifest.json"),
      include_bytes!("metal/artifact/kernels.metallib"),
    )
    .map_err(device_error)?;
    let context = Rc::new(Context { device, module });
    contexts.push((ordinal, Rc::clone(&context)));
    Ok(context)
  })
}

fn create_device(ordinal: usize) -> Result<Device> {
  if ordinal == 0 {
    return Device::system_default().map_err(device_error);
  }
  let mut devices = Device::all().map_err(device_error)?;
  if ordinal < devices.len() {
    Ok(devices.swap_remove(ordinal))
  } else {
    Err(DeviceError::Unavailable(format!(
      "no Metal device at index {ordinal}; this machine has {}",
      devices.len()
    )))
  }
}

/// Half the recommended working set, since launches also hold parameters and tables;
/// `usize::MAX` when the device will not open, which the following launch reports.
pub(crate) fn working_set(ordinal: usize) -> usize {
  context(ordinal)
    .map(|context| (context.device.recommended_max_working_set_size() / 2) as usize)
    .unwrap_or(usize::MAX)
}

pub(crate) fn probe(ordinal: usize) -> Result<DeviceInfo> {
  let device = create_device(ordinal)?;
  Ok(DeviceInfo::new(
    "Metal",
    device.name(),
    &["f32"],
    Some(ordinal),
  ))
}

pub(crate) fn device_error(error: metal_oxide::Error) -> DeviceError {
  use metal_oxide::Error;
  match error {
    Error::DeviceUnavailable | Error::UnsupportedDevice(_) => {
      DeviceError::Unavailable(error.to_string())
    }
    Error::AllocationFailed { .. } | Error::BufferTooLarge { .. } | Error::OutOfMemory(_) => {
      DeviceError::OutOfMemory(error.to_string())
    }
    Error::Library(_) | Error::Artifact(_) => DeviceError::Compile(error.to_string()),
    _ => DeviceError::Launch(error.to_string()),
  }
}

#[cfg(test)]
mod tests {
  use super::{DeviceError, device_error};
  use metal_oxide::Error;

  #[test]
  fn pipeline_and_kernel_lookup_failures_use_the_launch_channel() {
    for error in [
      Error::Pipeline("pipeline creation failed".into()),
      Error::KernelNotFound("missing_kernel".into()),
    ] {
      let message = error.to_string();
      assert!(matches!(device_error(error), DeviceError::Launch(reason) if reason == message));
    }
    assert!(matches!(
      device_error(Error::Library("invalid library".into())),
      DeviceError::Compile(_)
    ));
  }
}
