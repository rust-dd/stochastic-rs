//! Discrete-time series models: the ARMA family and the GARCH family of conditional variances.
//!
//! $$
//! X_t = f(X_{t-1}, X_{t-2}, \ldots, \varepsilon_t, \varepsilon_{t-1}, \ldots)
//! $$
//!
pub mod agarch;
pub mod ar;
pub mod arch;
pub mod arima;
pub mod egarch;
pub mod garch;
pub mod gjr_garch;
pub mod ma;
pub mod sarima;
mod threshold;

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::agarch::Agarch;
  use super::ar::Ar;
  use super::arch::Arch;
  use super::egarch::Egarch;
  use super::garch::Garch;
  use super::gjr_garch::GjrGarch;
  use super::sarima::Sarima;
  use crate::traits::ProcessExt;

  #[test]
  fn ar_rejects_short_initial_state() {
    let result = std::panic::catch_unwind(|| {
      let _ = Ar::<f64>::new(
        Array1::from_vec(vec![0.4, -0.1]),
        0.1,
        16,
        Some(Array1::from_vec(vec![0.0])),
        Unseeded,
      );
    });
    assert!(result.is_err());
  }

  #[test]
  fn egarch_rejects_mismatched_alpha_gamma_lengths() {
    let result = std::panic::catch_unwind(|| {
      let _ = Egarch::<f64>::new(
        0.0,
        Array1::from_vec(vec![0.1, 0.2]),
        Array1::from_vec(vec![0.1]),
        Array1::from_vec(vec![0.9]),
        32,
        Unseeded,
      );
    });
    assert!(result.is_err());
  }

  #[test]
  fn gjr_garch_rejects_mismatched_alpha_gamma_lengths() {
    let result = std::panic::catch_unwind(|| {
      let _ = GjrGarch::<f64>::new(
        0.05,
        Array1::from_vec(vec![0.1, 0.05]),
        Array1::from_vec(vec![0.1]),
        Array1::from_vec(vec![0.8]),
        32,
        Unseeded,
      );
    });
    assert!(result.is_err());
  }

  #[test]
  fn agarch_rejects_mismatched_alpha_delta_lengths() {
    let result = std::panic::catch_unwind(|| {
      let _ = Agarch::<f64>::new(
        0.05,
        Array1::from_vec(vec![0.1, 0.05]),
        Array1::from_vec(vec![0.1]),
        Array1::from_vec(vec![0.8]),
        32,
        Unseeded,
      );
    });
    assert!(result.is_err());
  }

  #[test]
  fn sarima_rejects_zero_season_length() {
    let result = std::panic::catch_unwind(|| {
      let _ = Sarima::<f64>::new(
        Array1::from_vec(vec![0.3]),
        Array1::from_vec(vec![0.1]),
        Array1::from_vec(vec![0.2]),
        Array1::from_vec(vec![0.1]),
        0,
        1,
        0,
        0.1,
        64,
        Unseeded,
      );
    });
    assert!(result.is_err());
  }

  #[test]
  fn an_egarch_past_the_exponential_range_samples_without_a_panic() {
    let model = Egarch::<f64, _>::new(
      800.0,
      Array1::from_vec(vec![0.1]),
      Array1::from_vec(vec![0.0]),
      Array1::from_vec(vec![0.5]),
      64,
      Deterministic::new(7),
    );
    let path = model.sample();
    assert_eq!(path.len(), 64);
    assert!(path.iter().any(|x| !x.is_finite()));
  }

  #[test]
  fn a_stationary_second_order_egarch_beyond_unit_absolute_sum_samples_finite() {
    let model = Egarch::<f64, _>::new(
      -0.1,
      Array1::from_vec(vec![0.1]),
      Array1::from_vec(vec![-0.05]),
      Array1::from_vec(vec![1.2, -0.3]),
      512,
      Deterministic::new(7),
    );
    assert!(model.sample().iter().all(|x| x.is_finite()));
  }

  #[test]
  #[should_panic(expected = "alpha[0] must satisfy `alpha[i] >= T::zero()`, got alpha[0] = -0.5")]
  fn an_arch_loading_written_negative_after_new_is_refused_at_sampling() {
    let mut model =
      Arch::<f64, _>::new(0.1, Array1::from_vec(vec![0.2]), 64, Deterministic::new(7));
    model.alpha[0] = -0.5;
    let _ = model.sample();
  }

  #[test]
  fn garch_valid_parameters_produce_finite_sample() {
    let model = Garch::<f64>::new(
      0.05,
      Array1::from_vec(vec![0.08]),
      Array1::from_vec(vec![0.9]),
      256,
      Unseeded,
    );
    let path = model.sample();
    assert_eq!(path.len(), 256);
    assert!(path.iter().all(|x| x.is_finite()));
  }
}
