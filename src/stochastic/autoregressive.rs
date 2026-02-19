//! # Autoregressive
//!
//! $$
//! X_t = f(X_{t-1}, X_{t-2}, \ldots, \varepsilon_t, \varepsilon_{t-1}, \ldots)
//! $$
//!
pub mod agrach;
pub mod ar;
pub mod arch;
pub mod arima;
pub mod egarch;
pub mod garch;
pub mod ma;
pub mod sarima;
pub mod tgarch;

#[cfg(test)]
mod tests {
  use ndarray::Array1;

  use super::agrach::AGARCH;
  use super::ar::ARp;
  use super::egarch::EGARCH;
  use super::garch::GARCH;
  use super::sarima::SARIMA;
  use super::tgarch::TGARCH;
  use crate::traits::ProcessExt;

  #[test]
  fn ar_rejects_short_initial_state() {
    let result = std::panic::catch_unwind(|| {
      let _ = ARp::<f64>::new(
        Array1::from_vec(vec![0.4, -0.1]),
        0.1,
        16,
        Some(Array1::from_vec(vec![0.0])),
      );
    });
    assert!(result.is_err());
  }

  #[test]
  fn egarch_rejects_mismatched_alpha_gamma_lengths() {
    let result = std::panic::catch_unwind(|| {
      let _ = EGARCH::<f64>::new(
        0.0,
        Array1::from_vec(vec![0.1, 0.2]),
        Array1::from_vec(vec![0.1]),
        Array1::from_vec(vec![0.9]),
        32,
      );
    });
    assert!(result.is_err());
  }

  #[test]
  fn tgarch_rejects_mismatched_alpha_gamma_lengths() {
    let result = std::panic::catch_unwind(|| {
      let _ = TGARCH::<f64>::new(
        0.05,
        Array1::from_vec(vec![0.1, 0.05]),
        Array1::from_vec(vec![0.1]),
        Array1::from_vec(vec![0.8]),
        32,
      );
    });
    assert!(result.is_err());
  }

  #[test]
  fn agarch_rejects_mismatched_alpha_delta_lengths() {
    let result = std::panic::catch_unwind(|| {
      let _ = AGARCH::<f64>::new(
        0.05,
        Array1::from_vec(vec![0.1, 0.05]),
        Array1::from_vec(vec![0.1]),
        Array1::from_vec(vec![0.8]),
        32,
      );
    });
    assert!(result.is_err());
  }

  #[test]
  fn sarima_rejects_zero_season_length() {
    let result = std::panic::catch_unwind(|| {
      let _ = SARIMA::<f64>::new(
        Array1::from_vec(vec![0.3]),
        Array1::from_vec(vec![0.1]),
        Array1::from_vec(vec![0.2]),
        Array1::from_vec(vec![0.1]),
        0,
        1,
        0,
        0.1,
        64,
      );
    });
    assert!(result.is_err());
  }

  #[test]
  fn garch_rejects_non_stationary_parameters_at_sampling() {
    let result = std::panic::catch_unwind(|| {
      let model = GARCH::<f64>::new(
        0.05,
        Array1::from_vec(vec![0.7]),
        Array1::from_vec(vec![0.4]),
        128,
      );
      let _ = model.sample();
    });
    assert!(result.is_err());
  }

  #[test]
  fn garch_valid_parameters_produce_finite_sample() {
    let model = GARCH::<f64>::new(
      0.05,
      Array1::from_vec(vec![0.08]),
      Array1::from_vec(vec![0.9]),
      256,
    );
    let path = model.sample();
    assert_eq!(path.len(), 256);
    assert!(path.iter().all(|x| x.is_finite()));
  }
}
