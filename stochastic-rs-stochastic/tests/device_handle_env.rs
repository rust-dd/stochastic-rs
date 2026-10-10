//! A device handle reads `STOCHASTIC_RS_DEVICE` / `STOCHASTIC_RS_DEVICE_BATCH_BYTES` in `from_env` alone. One
//! test in a binary of its own, so no other thread touches the environment while `set_var` runs.

#![cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]

use std::ffi::OsStr;

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::device::DEFAULT_BATCH_BUDGET_BYTES;
use stochastic_rs_stochastic::device::DeviceError;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;

const DEVICE: &str = "STOCHASTIC_RS_DEVICE";
const BUDGET: &str = "STOCHASTIC_RS_DEVICE_BATCH_BYTES";

fn set(name: &str, value: Option<&OsStr>) {
  // SAFETY: the binary's only test is the only thread that reads or writes the environment.
  unsafe {
    match value {
      Some(value) => std::env::set_var(name, value),
      None => std::env::remove_var(name),
    }
  }
}

macro_rules! env_contract {
  ($handle:ident, $t:ty) => {{
    use stochastic_rs_stochastic::device::$handle;

    set(DEVICE, Some(OsStr::new("2")));
    set(BUDGET, Some(OsStr::new("4096")));
    let name = stringify!($handle);
    let default = $handle::default();
    assert_eq!(
      (default.ordinal, default.batch_budget),
      (0, DEFAULT_BATCH_BUDGET_BYTES),
      "{name}::default() read the environment"
    );
    let explicit = $handle::new(3);
    assert_eq!(
      (explicit.ordinal, explicit.batch_budget),
      (3, DEFAULT_BATCH_BUDGET_BYTES),
      "{name}::new(3) read the environment"
    );
    let gbm = Gbm::<$t, _>::new(0.05, 0.2, 16, Some(1.0), None, Deterministic::new(1));
    assert_eq!(
      gbm.on::<$handle>().backend(),
      &$handle::default(),
      "on::<{name}>() read the environment"
    );
    assert_eq!(
      $handle::from_env(),
      Ok($handle::new(2).with_batch_budget(4096)),
      "{name}::from_env() missed a variable"
    );

    let refused = |what: &str| {
      assert!(
        matches!($handle::from_env(), Err(DeviceError::Config(_))),
        "{name}::from_env() accepted {what}"
      )
    };
    set(DEVICE, Some(OsStr::new("gpu1")));
    refused("a malformed ordinal");
    set(DEVICE, None);
    set(BUDGET, Some(OsStr::new("0")));
    refused("a zero budget");

    #[cfg(unix)]
    {
      use std::os::unix::ffi::OsStrExt;
      set(BUDGET, Some(OsStr::from_bytes(b"\xff")));
      refused("a budget that is not unicode");
      set(BUDGET, None);
      set(DEVICE, Some(OsStr::from_bytes(b"\xff")));
      refused("an ordinal that is not unicode");
    }

    set(DEVICE, None);
    set(BUDGET, None);
    assert_eq!(
      $handle::from_env(),
      Ok($handle::default()),
      "{name}::from_env() without the variables"
    );
  }};
}

#[test]
fn a_handle_reads_the_environment_only_in_from_env() {
  #[cfg(feature = "cuda")]
  env_contract!(Cuda, f64);
  #[cfg(all(feature = "metal", target_os = "macos"))]
  env_contract!(Metal, f32);
}
