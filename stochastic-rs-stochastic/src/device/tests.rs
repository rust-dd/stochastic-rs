use super::chunks::lru_slot;
use super::chunks::rows_within_index_limit;
use super::env::budget_from_env;
use super::env::device_from_env;
use super::*;

#[test]
fn cpu_marker_is_a_backend() {
  fn assert_backend<B: Backend>() {}
  assert_backend::<Cpu>();
}

/// The marker trait alone must stay algorithm-free: this compiles because
/// `Cpu` has the fGN capability, and a future device that lacks it can
/// still be a [`Backend`] for the capabilities it does have.
#[test]
fn lru_slot_keeps_four_and_promotes_hits() {
  let mut cache: Vec<usize> = Vec::new();
  for k in 0..4 {
    lru_slot::<usize, ()>(&mut cache, |c| *c == k, || Ok(k)).unwrap();
  }
  assert_eq!(cache, vec![0, 1, 2, 3]);
  lru_slot::<usize, ()>(&mut cache, |c| *c == 1, || Ok(1)).unwrap();
  assert_eq!(
    cache,
    vec![0, 2, 3, 1],
    "a hit moves to the most-recent slot"
  );
  lru_slot::<usize, ()>(&mut cache, |c| *c == 9, || Ok(9)).unwrap();
  assert_eq!(
    cache,
    vec![2, 3, 1, 9],
    "a miss evicts the least-recent slot"
  );
  assert!(lru_slot::<usize, ()>(&mut cache, |c| *c == 7, || Err(())).is_err());
  assert_eq!(
    cache,
    vec![2, 3, 1, 9],
    "a failed build leaves the cache alone"
  );
}

#[test]
fn the_batch_budget_variable_is_a_positive_integer_or_an_error() {
  assert_eq!(budget_from_env(None), Ok(DEFAULT_BATCH_BUDGET_BYTES));
  assert_eq!(budget_from_env(Some(" 4096 ")), Ok(4096));
  assert!(matches!(
    budget_from_env(Some("0")),
    Err(DeviceError::Config(_))
  ));
  assert!(matches!(
    budget_from_env(Some("x")),
    Err(DeviceError::Config(_))
  ));
}

#[test]
fn the_device_variable_is_an_ordinal_or_an_error() {
  assert_eq!(device_from_env(None), Ok(0));
  assert_eq!(device_from_env(Some(" 2 ")), Ok(2));
  assert!(matches!(
    device_from_env(Some("gpu1")),
    Err(DeviceError::Config(_))
  ));
  assert!(matches!(
    device_from_env(Some("")),
    Err(DeviceError::Config(_))
  ));
}

#[test]
fn cpu_probe_reports_both_precisions() {
  let info = Cpu.probe().expect("the host is always available");
  assert_eq!(info.backend, "Cpu");
  assert_eq!(info.precisions, &["f32", "f64"]);
  assert_eq!(info.ordinal, None);
}

#[test]
fn device_error_names_its_kind() {
  assert_eq!(
    DeviceError::Unavailable("no Metal device".into()).to_string(),
    "device unavailable: no Metal device"
  );
  assert_eq!(
    DeviceError::Compile("NVRTC euler_paths_float: x".into()).to_string(),
    "kernel compilation failed: NVRTC euler_paths_float: x"
  );
  assert_eq!(
    DeviceError::Launch("alloc out: y".into()).to_string(),
    "device operation failed: alloc out: y"
  );
}

#[test]
fn cpu_marker_has_the_fgn_capability() {
  fn assert_fgn<B: FgnBackend<f64>>() {}
  assert_fgn::<Cpu>();
}

#[test]
fn cpu_marker_has_the_sheet_capability() {
  fn assert_sheet<B: SheetBackend<f64>>() {}
  assert_sheet::<Cpu>();
}

mod index_limits {
  use super::*;

  #[test]
  fn rows_shrink_to_fit_the_index_range() {
    assert_eq!(rows_within_index_limit(10, 100, 1_000), 10);
    assert_eq!(rows_within_index_limit(10, 300, 1_000), 3);
    assert_eq!(rows_within_index_limit(10, 5_000, 1_000), 1);
  }

  #[test]
  fn a_launch_length_past_i32_is_an_error() {
    assert!(launch_len::<i32>(i32::MAX as usize + 1, "gen_scale").is_err());
    assert_eq!(launch_len::<i32>(42, "gen_scale").unwrap(), 42);
  }
}
