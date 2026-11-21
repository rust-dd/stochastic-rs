/// A device handle: the ordinal it opens and the bytes of path data one launch may hold.
macro_rules! device_handle {
  ($(#[$meta:meta])* $name:ident, $ordinal_doc:literal) => {
    $(#[$meta])*
    #[derive(Clone, Debug, PartialEq, Eq)]
    #[non_exhaustive]
    pub struct $name {
      #[doc = $ordinal_doc]
      pub ordinal: usize,
      /// Bytes of path data one launch may hold; a larger batch runs as chunks
      /// whose union is bit-identical to one launch.
      pub batch_budget: usize,
    }

    impl Default for $name {
      /// Ordinal `0` with [`DEFAULT_BATCH_BUDGET_BYTES`]; [`from_env`](Self::from_env) reads the
      /// environment.
      fn default() -> Self {
        Self {
          ordinal: 0,
          batch_budget: DEFAULT_BATCH_BUDGET_BYTES,
        }
      }
    }

    impl $name {
      /// The device at `ordinal` with the default batch budget.
      pub fn new(ordinal: usize) -> Self {
        Self {
          ordinal,
          ..Self::default()
        }
      }

      /// The device of `STOCHASTIC_RS_DEVICE`, the budget of `STOCHASTIC_RS_DEVICE_BATCH_BYTES`;
      /// an unset variable takes the default, an unusable one is a [`DeviceError::Config`].
      pub fn from_env() -> Result<Self, DeviceError> {
        Ok(Self {
          ordinal: env_ordinal()?,
          batch_budget: env_budget()?,
        })
      }

      /// The same device with `bytes` of path data per launch, at least one.
      pub fn with_batch_budget(self, bytes: usize) -> Self {
        Self {
          batch_budget: bytes.max(1),
          ..self
        }
      }
    }

    impl sealed::Sealed for $name {}
  };
}
