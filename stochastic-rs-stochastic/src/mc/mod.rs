//! $$
//! \hat{\mu}_N = \frac{1}{N}\sum_{i=1}^{N} f(X_i),\quad
//! \operatorname{Var}[\hat{\mu}_N] = \frac{\sigma^2}{N}
//! $$
//!
//! Variance reduction, Brownian-bridge QMC paths over the `stochastic_rs_distributions::qmc`
//! sequences and multi-level MC; the Longstaff–Schwartz pricer is quant's `BermudanLsmPricer`.
//!
//! Reference: Glasserman (2003), Springer, DOI 10.1007/978-0-387-21617-1

pub mod antithetic;
pub mod brownian_bridge_qmc;
pub mod control_variates;
pub mod importance_sampling;
pub mod mlmc;
pub mod stratified;
pub mod welford;

pub use welford::Welford;

use crate::traits::FloatExt;

/// Result of a Monte Carlo estimation.
#[derive(Debug, Clone)]
pub struct McEstimate<T: FloatExt> {
  /// Estimated mean.
  pub mean: T,
  /// Standard error of the estimate: the sample standard deviation (`n − 1`
  /// denominator) over `√n`. `NaN` when fewer than two samples were used.
  pub std_err: T,
  /// Number of samples used.
  pub n_samples: usize,
}

impl<T: FloatExt> McEstimate<T> {
  /// Symmetric confidence interval `[mean ± z · std_err]`.
  pub fn confidence_interval(&self, z: T) -> (T, T) {
    (self.mean - z * self.std_err, self.mean + z * self.std_err)
  }

  /// 95% confidence interval (z = 1.96).
  pub fn ci_95(&self) -> (T, T) {
    self.confidence_interval(T::from_f64_fast(1.96))
  }
}

impl<T: FloatExt + std::fmt::Display> std::fmt::Display for McEstimate<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{:.4} ± {:.4}", self.mean, self.std_err)
  }
}

/// The estimate the accumulated samples give: their mean, its standard error
/// and their count.
impl<T: FloatExt> From<&Welford> for McEstimate<T> {
  fn from(acc: &Welford) -> Self {
    McEstimate {
      mean: T::from_f64_fast(acc.mean()),
      std_err: T::from_f64_fast(acc.std_err()),
      n_samples: acc.count(),
    }
  }
}

/// Mean and standard error of i.i.d. samples through [`Welford`]: `sqrt(s² / n)` on the `n − 1`
/// sample variance, `NaN` with fewer than two samples or a non-finite one.
///
/// Reference: Welford (1962), DOI 10.1080/00401706.1962.10490022
pub(crate) fn estimate_from_samples<T: FloatExt>(
  samples: impl IntoIterator<Item = T>,
) -> McEstimate<T> {
  let mut acc = Welford::default();
  acc.extend(samples);
  McEstimate::from(&acc)
}

#[cfg(test)]
mod estimator_numerics {
  use ndarray::Array1;
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  const PATHS: usize = 10_000;

  /// A constant with no exact binary form, so its sums round. `3.0` is exact in
  /// binary, which hides a variance computed as `sum_sq / n − mean²`.
  const NEAR_CONSTANT: f64 = 0.1;

  fn assert_zero_error(est: McEstimate<f64>) {
    assert!(
      (est.mean - NEAR_CONSTANT).abs() < 1e-12,
      "mean {}",
      est.mean
    );
    assert_eq!(est.std_err, 0.0);
    assert_eq!(est.n_samples, PATHS);
  }

  #[test]
  fn a_constant_payoff_has_zero_standard_error() {
    let est = antithetic::estimate::<f64, _, _>(10_000, 1, |_| 3.0, &Deterministic::new(1));
    assert_eq!(est.mean, 3.0);
    assert_eq!(est.std_err, 0.0);
  }

  #[test]
  fn antithetic_reports_zero_error_for_a_near_constant_payoff() {
    assert_zero_error(antithetic::estimate(
      PATHS,
      1,
      |_| NEAR_CONSTANT,
      &Deterministic::new(1),
    ));
  }

  #[test]
  fn antithetic_par_reports_zero_error_for_a_near_constant_payoff() {
    assert_zero_error(antithetic::estimate_par(
      PATHS,
      1,
      |_| NEAR_CONSTANT,
      &Deterministic::new(1),
    ));
  }

  #[test]
  fn importance_sampling_reports_zero_error_for_a_near_constant_payoff() {
    let shift = Array1::<f64>::zeros(1);
    assert_zero_error(importance_sampling::estimate(
      PATHS,
      1,
      |_| NEAR_CONSTANT,
      &shift,
      &Deterministic::new(1),
    ));
  }

  #[test]
  fn stratified_reports_zero_error_for_a_near_constant_payoff() {
    assert_zero_error(stratified::estimate(
      PATHS,
      1,
      |_| NEAR_CONSTANT,
      &Deterministic::new(1),
    ));
  }

  #[test]
  fn control_variates_report_zero_error_for_a_near_constant_payoff() {
    assert_zero_error(control_variates::estimate(
      PATHS,
      1,
      |_| NEAR_CONSTANT,
      |z: &Array1<f64>| z[0],
      0.0,
      &Deterministic::new(1),
    ));
  }

  #[test]
  fn every_one_shot_estimator_is_a_function_of_its_seed() {
    let payoff = |z: &Array1<f64>| z[0].max(0.0);
    let shift = Array1::from_elem(1, 0.5);
    let control = |seed: u64| {
      control_variates::estimate(1_000, 1, payoff, |z| z[0], 0.0, &Deterministic::new(seed))
    };
    let shifted = |seed: u64| {
      importance_sampling::estimate(1_000, 1, payoff, &shift, &Deterministic::new(seed))
    };
    let strata = |seed: u64| stratified::estimate(1_000, 1, payoff, &Deterministic::new(seed));
    let runs: [&dyn Fn(u64) -> McEstimate<f64>; 3] = [&control, &shifted, &strata];
    for (k, run) in runs.iter().enumerate() {
      assert_eq!(
        run(7).mean.to_bits(),
        run(7).mean.to_bits(),
        "estimator {k}"
      );
      assert_ne!(
        run(7).mean.to_bits(),
        run(8).mean.to_bits(),
        "estimator {k}"
      );
    }
  }

  #[test]
  fn f32_means_do_not_saturate() {
    let est = estimate_from_samples(std::iter::repeat_n(1.0f32, 20_000_000));
    assert!((est.mean - 1.0).abs() < 1e-6, "mean {}", est.mean);
    assert_eq!(est.n_samples, 20_000_000);
  }

  #[test]
  fn the_helper_uses_the_sample_variance() {
    let est = estimate_from_samples([1.0f64, 2.0, 3.0, 4.0]);
    assert!((est.mean - 2.5).abs() < 1e-15);
    let var = 5.0 / 3.0;
    assert!((est.std_err - (var / 4.0f64).sqrt()).abs() < 1e-15);
  }

  #[test]
  fn a_large_offset_does_not_cancel() {
    let est = estimate_from_samples([1.0e9 + 1.0, 1.0e9 + 2.0, 1.0e9 + 3.0, 1.0e9 + 4.0]);
    assert_eq!(est.mean, 1.0e9 + 2.5);
    assert!((est.std_err - (5.0 / 3.0 / 4.0f64).sqrt()).abs() < 1e-12);
  }

  #[test]
  fn a_non_finite_sample_poisons_the_standard_error() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
      let est = estimate_from_samples([1.0, bad, 3.0]);
      assert!(
        est.std_err.is_nan(),
        "sample {bad} gave std_err {}",
        est.std_err
      );
    }
  }

  #[test]
  fn fewer_than_two_samples_have_no_standard_error() {
    let none = estimate_from_samples(std::iter::empty::<f64>());
    assert!(none.mean.is_nan() && none.std_err.is_nan());
    assert_eq!(none.n_samples, 0);

    let one = estimate_from_samples([2.5f64]);
    assert_eq!(one.mean, 2.5);
    assert!(one.std_err.is_nan());
    assert_eq!(one.n_samples, 1);
  }
}
