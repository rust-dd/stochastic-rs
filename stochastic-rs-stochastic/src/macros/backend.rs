/// A process's `on::<B2>()`, `with_backend`, `backend` and `probe`, from its generics without `B`,
/// its type, every field but the backend's and the form that holds the backend (`via euler`, …).
macro_rules! backend_switch {
  (
    [$($gen:tt)*] $ty:ident<$t:ident $(, $targ:ident)* $(,)?> { $($field:ident),* $(,)? } via fgn euler
    $(where $($wc:tt)*)?
  ) => {
    impl<$($gen)*, B> $ty<$t $(, $targ)*, B> $(where $($wc)*)? {
      /// The same process on backend `B2`, using that backend's default handle
      /// (ordinal `0`, the default batch budget).
      pub fn on<B2: $crate::device::FgnBackend<$t> + $crate::euler::EulerBackend<$t> + Default>(self) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          fgn: self.fgn.on::<B2>(),
        }
      }

      /// The same process on an explicit handle: `Cuda::new(1)`, `Metal::default().with_batch_budget(256 << 20)`,
      /// or `Cuda::from_env()?` for the device and budget of `STOCHASTIC_RS_DEVICE` / `STOCHASTIC_RS_DEVICE_BATCH_BYTES`.
      pub fn with_backend<B2: $crate::device::FgnBackend<$t> + $crate::euler::EulerBackend<$t>>(self, device: B2) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          fgn: self.fgn.with_backend(device),
        }
      }

      /// The handle `on` or `with_backend` put here, ordinal and batch budget included: what [`probe`](Self::probe)
      /// opens, and the one way to read back a choice the type alone does not record.
      pub fn backend(&self) -> &B {
        &self.fgn.backend
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
      /// (ordinal `0`, the default batch budget).
      pub fn on<B2: $crate::device::FgnBackend<$t> + Default>(self) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          backend: B2::default(),
        }
      }

      /// The same process on an explicit handle: `Cuda::new(1)`, `Metal::default().with_batch_budget(256 << 20)`,
      /// or `Cuda::from_env()?` for the device and budget of `STOCHASTIC_RS_DEVICE` / `STOCHASTIC_RS_DEVICE_BATCH_BYTES`.
      pub fn with_backend<B2: $crate::device::FgnBackend<$t>>(self, device: B2) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          backend: device,
        }
      }

      /// The handle `on` or `with_backend` put here, ordinal and batch budget included: what [`probe`](Self::probe)
      /// opens, and the one way to read back a choice the type alone does not record.
      pub fn backend(&self) -> &B {
        &self.backend
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
      /// (ordinal `0`, the default batch budget).
      pub fn on<B2: $crate::device::SheetBackend<$t> + Default>(self) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          backend: B2::default(),
        }
      }

      /// The same process on an explicit handle: `Cuda::new(1)`, `Metal::default().with_batch_budget(256 << 20)`,
      /// or `Cuda::from_env()?` for the device and budget of `STOCHASTIC_RS_DEVICE` / `STOCHASTIC_RS_DEVICE_BATCH_BYTES`.
      pub fn with_backend<B2: $crate::device::SheetBackend<$t>>(self, device: B2) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          backend: device,
        }
      }

      /// The handle `on` or `with_backend` put here, ordinal and batch budget included: what [`probe`](Self::probe)
      /// opens, and the one way to read back a choice the type alone does not record.
      pub fn backend(&self) -> &B {
        &self.backend
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
      /// (ordinal `0`, the default batch budget).
      pub fn on<B2: $crate::euler::EulerBackend<$t> + Default>(self) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          backend: B2::default(),
        }
      }

      /// The same process on an explicit handle: `Cuda::new(1)`, `Metal::default().with_batch_budget(256 << 20)`,
      /// or `Cuda::from_env()?` for the device and budget of `STOCHASTIC_RS_DEVICE` / `STOCHASTIC_RS_DEVICE_BATCH_BYTES`.
      pub fn with_backend<B2: $crate::euler::EulerBackend<$t>>(self, device: B2) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          backend: device,
        }
      }

      /// The handle `on` or `with_backend` put here, ordinal and batch budget included: what [`probe`](Self::probe)
      /// opens, and the one way to read back a choice the type alone does not record.
      pub fn backend(&self) -> &B {
        &self.backend
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
