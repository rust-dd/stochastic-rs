#[cfg(feature = "python")]
#[macro_export]
macro_rules! py_dispatch_f64 {
  ($self:expr, |$inner:ident| $body:expr) => {
    if let Some(ref $inner) = $self.inner {
      $body
    } else if let Some(ref $inner) = $self.seeded {
      $body
    } else {
      unreachable!()
    }
  };
}

/// `py_dispatch_f64!` on the class's device. A class whose parameters
/// include a Python callable has no `float32` slot, so only the devices that
/// compute in double precision can appear here — which is exactly the set
/// `py_on_device_f64!` handles.
#[cfg(feature = "python")]
#[macro_export]
macro_rules! py_device_dispatch_f64 {
  ($self:expr, |$p:ident| $body:expr) => {
    if let Some(ref inner) = $self.inner {
      $crate::py_on_device_f64!($self.device, inner, |$p| $body)
    } else if let Some(ref inner) = $self.seeded {
      $crate::py_on_device_f64!($self.device, inner, |$p| $body)
    } else {
      unreachable!()
    }
  };
}

#[cfg(feature = "python")]
#[macro_export]
macro_rules! py_dispatch {
  ($self:expr, |$inner:ident| $body:expr) => {
    if let Some(ref $inner) = $self.inner_f64 {
      $body
    } else if let Some(ref $inner) = $self.seeded_f64 {
      $body
    } else if let Some(ref $inner) = $self.inner_f32 {
      $body
    } else if let Some(ref $inner) = $self.seeded_f32 {
      $body
    } else {
      unreachable!()
    }
  };
}

/// One `sample*` body over a `float64` slot on the class's device: the host
/// slot as is, a device slot re-typed on a clone sharing its seed state. The
/// single-precision devices never reach here — `Device::parse` rejects them
/// for a `float64` process.
#[cfg(feature = "python")]
#[macro_export]
macro_rules! py_on_device_f64 {
  ($device:expr, $inner:ident, |$p:ident| $body:expr) => {
    match $device {
      $crate::python_device::Device::Cpu => {
        let $p = $inner;
        $body
      }
      #[cfg(feature = "accelerate")]
      $crate::python_device::Device::Accelerate => {
        let owned = $inner.clone().with_backend($crate::device::Accelerate);
        let $p = &owned;
        $body
      }
      #[cfg(feature = "cuda")]
      $crate::python_device::Device::Cuda(ordinal) => {
        let owned = $inner
          .clone()
          .with_backend($crate::device::Cuda::new(ordinal));
        let $p = &owned;
        $body
      }
      #[allow(unreachable_patterns)]
      _ => unreachable!("single-precision devices are rejected for float64 at construction"),
    }
  };
}

/// `py_on_device_f64!` for a `float32` slot, which every compiled device
/// accepts.
#[cfg(feature = "python")]
#[macro_export]
macro_rules! py_on_device_f32 {
  ($device:expr, $inner:ident, |$p:ident| $body:expr) => {
    match $device {
      $crate::python_device::Device::Cpu => {
        let $p = $inner;
        $body
      }
      #[cfg(feature = "accelerate")]
      $crate::python_device::Device::Accelerate => {
        let owned = $inner.clone().with_backend($crate::device::Accelerate);
        let $p = &owned;
        $body
      }
      #[cfg(feature = "cuda")]
      $crate::python_device::Device::Cuda(ordinal) => {
        let owned = $inner
          .clone()
          .with_backend($crate::device::Cuda::new(ordinal));
        let $p = &owned;
        $body
      }
      #[cfg(feature = "metal")]
      $crate::python_device::Device::Metal(ordinal) => {
        let owned = $inner
          .clone()
          .with_backend($crate::device::Metal::new(ordinal));
        let $p = &owned;
        $body
      }
      #[allow(unreachable_patterns)]
      _ => unreachable!("devices this build lacks are rejected at construction"),
    }
  };
}

/// `py_dispatch!` for a class with a `device` field: the four slots, each
/// routed through `py_on_device_f64!` / `py_on_device_f32!`.
#[cfg(feature = "python")]
#[macro_export]
macro_rules! py_device_dispatch {
  ($self:expr, |$p:ident| $body:expr) => {
    if let Some(ref inner) = $self.inner_f64 {
      $crate::py_on_device_f64!($self.device, inner, |$p| $body)
    } else if let Some(ref inner) = $self.seeded_f64 {
      $crate::py_on_device_f64!($self.device, inner, |$p| $body)
    } else if let Some(ref inner) = $self.inner_f32 {
      $crate::py_on_device_f32!($self.device, inner, |$p| $body)
    } else if let Some(ref inner) = $self.seeded_f32 {
      $crate::py_on_device_f32!($self.device, inner, |$p| $body)
    } else {
      unreachable!()
    }
  };
}
