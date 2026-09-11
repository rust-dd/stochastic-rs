#[path = "common/lm_calibration.rs"]
mod fixtures;

#[test]
fn calibration_convergence_regression() {
  for case in fixtures::cases() {
    let (converged, rmse) = (case.run)();
    println!("{}: converged={converged:?}, rmse={rmse:.12e}", case.name);
    assert_ne!(converged, Some(false), "{} did not converge", case.name);
    assert!(
      rmse < case.max_rmse,
      "{}: RMSE {rmse} exceeds {}",
      case.name,
      case.max_rmse
    );
  }
}
