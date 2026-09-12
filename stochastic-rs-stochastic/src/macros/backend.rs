/// Compile-time backend switch: generates the whole `impl` block whose
/// `on::<B2>()` re-types a process to sample on another
/// [`crate::device::Backend`] with zero runtime cost. The marker params `B`
/// (source) and `B2` (target) are appended automatically — pass the remaining
/// generics in `[..]`, the type without `B`, the moved fields, the storage
/// form, then any trailing `where` bounds.
///
/// Storage form:
/// - `via fgn`     — backend carried through an inner `fgn: Fgn<_, _, B>` field.
/// - `via phantom` — backend carried through a `backend: Cpu<B>` field.
/// - `via host`    — a process with a host sampler only (`B2: HostBackend`), backend carried
///   through its public `backend: Cpu<B>` field.
/// - `via euler`   — a process with Euler-engine device kernels (`B2: EulerBackend`), same field.
macro_rules! backend_switch {
  (
    [$($gen:tt)*] $ty:ident<$t:ident $(, $targ:ident)* $(,)?> { $($field:ident),* $(,)? } via fgn
    $(where $($wc:tt)*)?
  ) => {
    impl<$($gen)*, B> $ty<$t $(, $targ)*, B> $(where $($wc)*)? {
      /// The same process on backend `B2`, using that backend's default handle
      /// (the device ordinal and batch budget from the environment).
      pub fn on<B2: $crate::device::FgnBackend<$t> + Default>(self) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          fgn: self.fgn.on::<B2>(),
        }
      }

      /// The same process on an explicit backend handle: a device picked by
      /// ordinal, `with_backend(Cuda::new(1))`, or a handle with its own batch
      /// budget, `with_backend(Metal::default().with_batch_budget(256 << 20))`.
      /// [`on`](Self::on) is the same move with the backend's default handle,
      /// which reads `STOCHASTIC_RS_DEVICE` and `STOCHASTIC_RS_DEVICE_BATCH_BYTES`.
      pub fn with_backend<B2: $crate::device::FgnBackend<$t>>(self, device: B2) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          fgn: self.fgn.with_backend(device),
        }
      }

      /// The handle this process samples on: the value `on` or
      /// `with_backend` put there, with the device ordinal and batch budget
      /// it carries. What [`probe`](Self::probe) opens, and the one way to
      /// read back a choice the type alone does not record.
      pub fn backend(&self) -> B
      where
        B: Copy,
      {
        self.fgn.backend
      }

      /// What the device behind this process reports — its name, the scalars
      /// it computes in, its ordinal — or why it cannot be used. The CPU
      /// devices always answer `Ok`.
      pub fn probe(&self) -> ::core::result::Result<$crate::device::DeviceInfo, $crate::device::DeviceError>
      where
        B: $crate::device::Backend,
      {
        $crate::device::Backend::probe(&self.fgn.backend)
      }
    }
  };
  (
    [$($gen:tt)*] $ty:ident<$t:ident $(, $targ:ident)* $(,)?> { $($field:ident),* $(,)? } via fgn euler
    $(where $($wc:tt)*)?
  ) => {
    impl<$($gen)*, B> $ty<$t $(, $targ)*, B> $(where $($wc)*)? {
      /// The same process on backend `B2`, using that backend's default handle
      /// (the device ordinal and batch budget from the environment).
      pub fn on<B2: $crate::device::FgnBackend<$t> + $crate::euler::EulerBackend<$t> + Default>(self) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          fgn: self.fgn.on::<B2>(),
        }
      }

      /// The same process on an explicit backend handle: a device picked by
      /// ordinal, `with_backend(Cuda::new(1))`, or a handle with its own batch
      /// budget, `with_backend(Metal::default().with_batch_budget(256 << 20))`.
      /// [`on`](Self::on) is the same move with the backend's default handle,
      /// which reads `STOCHASTIC_RS_DEVICE` and `STOCHASTIC_RS_DEVICE_BATCH_BYTES`.
      pub fn with_backend<B2: $crate::device::FgnBackend<$t> + $crate::euler::EulerBackend<$t>>(self, device: B2) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          fgn: self.fgn.with_backend(device),
        }
      }

      /// The handle this process samples on: the value `on` or
      /// `with_backend` put there, with the device ordinal and batch budget
      /// it carries. What [`probe`](Self::probe) opens, and the one way to
      /// read back a choice the type alone does not record.
      pub fn backend(&self) -> B
      where
        B: Copy,
      {
        self.fgn.backend
      }

      /// What the device behind this process reports — its name, the scalars
      /// it computes in, its ordinal — or why it cannot be used. The CPU
      /// devices always answer `Ok`.
      pub fn probe(&self) -> ::core::result::Result<$crate::device::DeviceInfo, $crate::device::DeviceError>
      where
        B: $crate::device::Backend,
      {
        $crate::device::Backend::probe(&self.fgn.backend)
      }
    }
  };
  (
    [$($gen:tt)*] $ty:ident<$t:ident $(, $targ:ident)* $(,)?> { $($field:ident),* $(,)? } via phantom
    $(where $($wc:tt)*)?
  ) => {
    impl<$($gen)*, B> $ty<$t $(, $targ)*, B> $(where $($wc)*)? {
      /// The same process on backend `B2`, using that backend's default handle
      /// (the device ordinal and batch budget from the environment).
      pub fn on<B2: $crate::device::FgnBackend<$t> + Default>(self) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          backend: B2::default(),
        }
      }

      /// The same process on an explicit backend handle: a device picked by
      /// ordinal, `with_backend(Cuda::new(1))`, or a handle with its own batch
      /// budget, `with_backend(Metal::default().with_batch_budget(256 << 20))`.
      /// [`on`](Self::on) is the same move with the backend's default handle,
      /// which reads `STOCHASTIC_RS_DEVICE` and `STOCHASTIC_RS_DEVICE_BATCH_BYTES`.
      pub fn with_backend<B2: $crate::device::FgnBackend<$t>>(self, device: B2) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          backend: device,
        }
      }

      /// The handle this process samples on: the value `on` or
      /// `with_backend` put there, with the device ordinal and batch budget
      /// it carries. What [`probe`](Self::probe) opens, and the one way to
      /// read back a choice the type alone does not record.
      pub fn backend(&self) -> B
      where
        B: Copy,
      {
        self.backend
      }

      /// What the device behind this process reports — its name, the scalars
      /// it computes in, its ordinal — or why it cannot be used. The CPU
      /// devices always answer `Ok`.
      pub fn probe(&self) -> ::core::result::Result<$crate::device::DeviceInfo, $crate::device::DeviceError>
      where
        B: $crate::device::Backend,
      {
        $crate::device::Backend::probe(&self.backend)
      }
    }
  };
  (
    [$($gen:tt)*] $ty:ident<$t:ident $(, $targ:ident)* $(,)?> { $($field:ident),* $(,)? } via host
    $(where $($wc:tt)*)?
  ) => {
    impl<$($gen)*, B> $ty<$t $(, $targ)*, B> $(where $($wc)*)? {
      /// The same process on backend `B2`, using that backend's default handle
      /// (the device ordinal and batch budget from the environment).
      pub fn on<B2: $crate::device::HostBackend + Default>(self) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          backend: B2::default(),
        }
      }

      /// The same process on an explicit backend handle: a device picked by
      /// ordinal, `with_backend(Cuda::new(1))`, or a handle with its own batch
      /// budget, `with_backend(Metal::default().with_batch_budget(256 << 20))`.
      /// [`on`](Self::on) is the same move with the backend's default handle,
      /// which reads `STOCHASTIC_RS_DEVICE` and `STOCHASTIC_RS_DEVICE_BATCH_BYTES`.
      pub fn with_backend<B2: $crate::device::HostBackend>(self, device: B2) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          backend: device,
        }
      }

      /// The handle this process samples on: the value `on` or
      /// `with_backend` put there, with the device ordinal and batch budget
      /// it carries. What [`probe`](Self::probe) opens, and the one way to
      /// read back a choice the type alone does not record.
      pub fn backend(&self) -> B
      where
        B: Copy,
      {
        self.backend
      }

      /// What the device behind this process reports — its name, the scalars
      /// it computes in, its ordinal — or why it cannot be used. The CPU
      /// devices always answer `Ok`.
      pub fn probe(&self) -> ::core::result::Result<$crate::device::DeviceInfo, $crate::device::DeviceError>
      where
        B: $crate::device::Backend,
      {
        $crate::device::Backend::probe(&self.backend)
      }
    }
  };
  (
    [$($gen:tt)*] $ty:ident<$t:ident $(, $targ:ident)* $(,)?> { $($field:ident),* $(,)? } via sheet
    $(where $($wc:tt)*)?
  ) => {
    impl<$($gen)*, B> $ty<$t $(, $targ)*, B> $(where $($wc)*)? {
      /// The same process on backend `B2`, using that backend's default handle
      /// (the device ordinal and batch budget from the environment).
      pub fn on<B2: $crate::device::SheetBackend<$t> + Default>(self) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          backend: B2::default(),
        }
      }

      /// The same process on an explicit backend handle: a device picked by
      /// ordinal, `with_backend(Cuda::new(1))`, or a handle with its own batch
      /// budget, `with_backend(Metal::default().with_batch_budget(256 << 20))`.
      /// [`on`](Self::on) is the same move with the backend's default handle,
      /// which reads `STOCHASTIC_RS_DEVICE` and `STOCHASTIC_RS_DEVICE_BATCH_BYTES`.
      pub fn with_backend<B2: $crate::device::SheetBackend<$t>>(self, device: B2) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          backend: device,
        }
      }

      /// The handle this process samples on: the value `on` or
      /// `with_backend` put there, with the device ordinal and batch budget
      /// it carries. What [`probe`](Self::probe) opens, and the one way to
      /// read back a choice the type alone does not record.
      pub fn backend(&self) -> B
      where
        B: Copy,
      {
        self.backend
      }

      /// What the device behind this process reports — its name, the scalars
      /// it computes in, its ordinal — or why it cannot be used. The CPU
      /// devices always answer `Ok`.
      pub fn probe(&self) -> ::core::result::Result<$crate::device::DeviceInfo, $crate::device::DeviceError>
      where
        B: $crate::device::Backend,
      {
        $crate::device::Backend::probe(&self.backend)
      }
    }
  };
  (
    [$($gen:tt)*] $ty:ident<$t:ident $(, $targ:ident)* $(,)?> { $($field:ident),* $(,)? } via euler
    $(where $($wc:tt)*)?
  ) => {
    impl<$($gen)*, B> $ty<$t $(, $targ)*, B> $(where $($wc)*)? {
      /// The same process on backend `B2`, using that backend's default handle
      /// (the device ordinal and batch budget from the environment).
      pub fn on<B2: $crate::euler::EulerBackend<$t> + Default>(self) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          backend: B2::default(),
        }
      }

      /// The same process on an explicit backend handle: a device picked by
      /// ordinal, `with_backend(Cuda::new(1))`, or a handle with its own batch
      /// budget, `with_backend(Metal::default().with_batch_budget(256 << 20))`.
      /// [`on`](Self::on) is the same move with the backend's default handle,
      /// which reads `STOCHASTIC_RS_DEVICE` and `STOCHASTIC_RS_DEVICE_BATCH_BYTES`.
      pub fn with_backend<B2: $crate::euler::EulerBackend<$t>>(self, device: B2) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          backend: device,
        }
      }

      /// The handle this process samples on: the value `on` or
      /// `with_backend` put there, with the device ordinal and batch budget
      /// it carries. What [`probe`](Self::probe) opens, and the one way to
      /// read back a choice the type alone does not record.
      pub fn backend(&self) -> B
      where
        B: Copy,
      {
        self.backend
      }

      /// What the device behind this process reports — its name, the scalars
      /// it computes in, its ordinal — or why it cannot be used. The CPU
      /// devices always answer `Ok`.
      pub fn probe(&self) -> ::core::result::Result<$crate::device::DeviceInfo, $crate::device::DeviceError>
      where
        B: $crate::device::Backend,
      {
        $crate::device::Backend::probe(&self.backend)
      }
    }
  };
}
