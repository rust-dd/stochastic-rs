use super::error::DeviceError;

#[cfg_attr(not(any(device_backend, feature = "python")), allow(dead_code))]
/// `STOCHASTIC_RS_DEVICE` as an ordinal: unset is `0`; only a non-negative integer is accepted.
pub(crate) fn device_from_env(value: Option<&str>) -> Result<usize, DeviceError> {
  let Some(raw) = value else {
    return Ok(0);
  };
  raw.trim().parse().map_err(|_| {
    DeviceError::Config(format!(
      "STOCHASTIC_RS_DEVICE must be a non-negative integer, got {raw:?}"
    ))
  })
}

#[cfg_attr(not(any(device_backend, feature = "python")), allow(dead_code))]
pub(crate) fn env_ordinal() -> Result<usize, DeviceError> {
  device_from_env(env_text("STOCHASTIC_RS_DEVICE")?.as_deref())
}

/// Default cap on the path data one device launch materialises: 1 GiB.
pub const DEFAULT_BATCH_BUDGET_BYTES: usize = 1 << 30;

#[cfg_attr(not(any(device_backend, feature = "python")), allow(dead_code))]
/// `STOCHASTIC_RS_DEVICE_BATCH_BYTES` as a budget: unset is the default, else a positive integer.
pub(crate) fn budget_from_env(value: Option<&str>) -> Result<usize, DeviceError> {
  let Some(raw) = value else {
    return Ok(DEFAULT_BATCH_BUDGET_BYTES);
  };
  match raw.trim().parse::<usize>() {
    Ok(bytes) if bytes > 0 => Ok(bytes),
    _ => Err(DeviceError::Config(format!(
      "STOCHASTIC_RS_DEVICE_BATCH_BYTES must be a positive integer, got {raw:?}"
    ))),
  }
}

#[cfg_attr(not(any(device_backend, feature = "python")), allow(dead_code))]
pub(crate) fn env_budget() -> Result<usize, DeviceError> {
  budget_from_env(env_text("STOCHASTIC_RS_DEVICE_BATCH_BYTES")?.as_deref())
}

#[cfg_attr(not(any(device_backend, feature = "python")), allow(dead_code))]
/// `name`'s value, `None` when unset; a non-unicode value is a `Config` error, never read as unset.
fn env_text(name: &str) -> Result<Option<String>, DeviceError> {
  match std::env::var(name) {
    Ok(value) => Ok(Some(value)),
    Err(std::env::VarError::NotPresent) => Ok(None),
    Err(std::env::VarError::NotUnicode(raw)) => Err(DeviceError::Config(format!(
      "{name} must be valid unicode, got {raw:?}"
    ))),
  }
}
