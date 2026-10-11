//! # Goodness of fit
//!
//! $$
//! S_n = \sum_{i=1}^n \bigl(C_n(\mathbf R_i) - R_{i1}R_{i2}\bigr)^2,\qquad
//! \mathbf R_i = \bigl(U_{i2},\ \partial_v C(U_{i1}, U_{i2})\bigr)
//! $$
//!
//! Cramér–von Mises test of a fitted bivariate copula on the Rosenblatt
//! transform of the pseudo-observations: under the null the transformed
//! sample is uniform on the square with independent coordinates, so the
//! empirical copula of the transformed sample is compared with the product
//! copula. The p-value comes from a parametric bootstrap (sample from the
//! fitted copula, refit, recompute the statistic), which is the valid
//! procedure when the copula parameters are estimated.
//!
//! Reference: Genest, C., Rémillard, B. & Beaudoin, D. (2009), "Goodness-of-fit tests for copulas: A review and a power study", Insurance: Mathematics and Economics 44(2), 199–213, doi:10.1016/j.insmatheco.2007.10.005 (§3.3 Rosenblatt transform; §4 and Appendix D `S_n^{(C)}` and its parametric bootstrap).
//! Reference: Rosenblatt, M. (1952), "Remarks on a multivariate transformation", Annals of Mathematical Statistics 23(3), 470–472, doi:10.1214/aoms/1177729394.

use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray::Axis;
use ndarray::stack;
use stochastic_rs_core::simd_rng::SeedExt;

use crate::error::CopulaError;
use crate::traits::BivariateExt;

/// Rosenblatt transform `(v, ∂_v C(u, v))` of the rows of `x` (Rosenblatt 1952).
/// Uniform only for a continuous conditional, which `MarshallOlkin`'s singular part is not.
pub fn rosenblatt<C: BivariateExt + ?Sized>(
  copula: &C,
  x: ArrayView2<f64>,
) -> Result<Array2<f64>, CopulaError> {
  let conditional = copula.partial_derivative(x)?;
  Ok(stack![Axis(1), x.column(1), conditional.view()])
}

/// Cramér–von Mises distance between the empirical copula of `r` and the
/// product copula, `Σ_i (C_n(R_i) − R_{i1} R_{i2})²`.
pub fn cramer_von_mises_independence(r: ArrayView2<f64>) -> f64 {
  let n = r.nrows();
  let mut statistic = 0.0;
  for i in 0..n {
    let (a, b) = (r[(i, 0)], r[(i, 1)]);
    let empirical = r
      .rows()
      .into_iter()
      .filter(|row| row[0] <= a && row[1] <= b)
      .count() as f64
      / n as f64;
    statistic += (empirical - a * b).powi(2);
  }
  statistic
}

/// Settings of the parametric-bootstrap test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GofConfig {
  /// Bootstrap replications behind the p-value.
  pub replications: usize,
  /// Significance level of the rejection decision.
  pub alpha: f64,
}

impl Default for GofConfig {
  fn default() -> Self {
    Self {
      replications: 200,
      alpha: 0.05,
    }
  }
}

/// Result of the bootstrap goodness-of-fit test.
#[derive(Clone, Debug, PartialEq)]
pub struct GofResult {
  /// Statistic on the data.
  pub statistic: f64,
  /// Bootstrap p-value: share of bootstrap statistics at or above the data's.
  pub p_value: f64,
  /// Number of bootstrap replications.
  pub replications: usize,
  /// Significance level the decision was taken at.
  pub alpha: f64,
  /// `p_value < alpha`.
  pub rejects_null: bool,
}

impl stochastic_rs_distributions::traits::HypothesisTest for GofResult {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.rejects_null)
  }
}

/// Parametric-bootstrap Cramér–von Mises test of the fitted `copula` on the pseudo-observations `x`, one stream of `seed` per
/// replication (Genest, Rémillard & Beaudoin 2009, §4 and Appendix D); panics unless `0 < alpha < 1` and `replications > 0`.
pub fn gof_cramer_von_mises<C: BivariateExt + Clone, S: SeedExt>(
  copula: &C,
  x: ArrayView2<f64>,
  config: GofConfig,
  seed: &S,
  refit: impl Fn(&mut C, ArrayView2<f64>) -> Result<(), CopulaError>,
) -> Result<GofResult, CopulaError> {
  assert!(
    config.replications > 0,
    "replications must satisfy `replications > 0`, got replications = {:?}",
    config.replications
  );
  assert!(
    config.alpha > 0.0 && config.alpha < 1.0,
    "alpha must satisfy `0.0 < alpha < 1.0`, got alpha = {:?}",
    config.alpha
  );
  let statistic = cramer_von_mises_independence(rosenblatt(copula, x)?.view());
  let n = x.nrows();
  let mut exceed = 0usize;
  for _ in 0..config.replications {
    let sample = pseudo_observations(copula.sample(n, seed)?.view());
    let mut refitted = copula.clone();
    refit(&mut refitted, sample.view())?;
    if cramer_von_mises_independence(rosenblatt(&refitted, sample.view())?.view()) >= statistic {
      exceed += 1;
    }
  }
  let p_value = exceed as f64 / config.replications as f64;
  Ok(GofResult {
    statistic,
    p_value,
    replications: config.replications,
    alpha: config.alpha,
    rejects_null: p_value < config.alpha,
  })
}

/// Pseudo-observations (normalised ranks `rank / (n + 1)`) of raw data.
pub fn pseudo_observations(data: ArrayView2<f64>) -> Array2<f64> {
  let n = data.nrows();
  let mut out = Array2::<f64>::zeros(data.raw_dim());
  for (j, column) in data.columns().into_iter().enumerate() {
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
      column[a]
        .partial_cmp(&column[b])
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (rank, &i) in order.iter().enumerate() {
      out[(i, j)] = (rank + 1) as f64 / (n + 1) as f64;
    }
  }
  out
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::traits::HypothesisTest;

  use super::*;
  use crate::bivariate::clayton::Clayton;
  use crate::bivariate::gaussian::GaussianCopula;

  fn data_for(seed: u64) -> Array2<f64> {
    let mut truth = Clayton::new();
    truth.set_tau(0.6); // θ = 3
    truth.set_theta(truth.compute_theta());
    pseudo_observations(truth.sample(600, &Deterministic::new(seed)).unwrap().view())
  }

  #[test]
  fn rosenblatt_of_the_true_copula_is_uniform_and_independent() {
    let mut copula = Clayton::new();
    copula.set_tau(0.5); // θ = 2τ/(1 − τ) = 2
    copula.set_theta(copula.compute_theta());
    let sample = copula.sample(3000, &Deterministic::new(3)).unwrap();
    let r = rosenblatt(&copula, sample.view()).unwrap();
    let (mean_a, mean_b) = (r.column(0).mean().unwrap(), r.column(1).mean().unwrap());
    assert!(
      (mean_a - 0.5).abs() < 0.03 && (mean_b - 0.5).abs() < 0.03,
      "means {mean_a} {mean_b}"
    );
    let (u, v): (Vec<f64>, Vec<f64>) = (r.column(0).to_vec(), r.column(1).to_vec());
    let (tau, ..) =
      kendalls::tau_b_with_comparator(&u, &v, |a: &f64, b: &f64| a.partial_cmp(b).unwrap())
        .unwrap();
    assert!(tau.abs() < 0.05, "tau {tau}");
    assert!(cramer_von_mises_independence(r.view()) < 0.2);
  }

  /// A Gaussian copula fitted to Clayton data is rejected; the true family
  /// is not. The true-family p-value is uniform under the null, so three
  /// pinned data seeds are tried and the best is asserted on.
  #[test]
  fn bootstrap_test_rejects_the_wrong_family() {
    let refit = |c: &mut Clayton, x: ArrayView2<f64>| c.fit(x);
    let best_right = [5u64, 6, 7]
      .into_iter()
      .map(|seed| {
        let data = data_for(seed);
        let mut fitted = Clayton::new();
        fitted.fit(data.view()).unwrap();
        gof_cramer_von_mises(
          &fitted,
          data.view(),
          GofConfig {
            replications: 60,
            alpha: 0.05,
          },
          &Deterministic::new(100),
          refit,
        )
        .unwrap()
        .p_value
      })
      .fold(0.0_f64, f64::max);
    assert!(
      best_right > 0.05,
      "true family rejected on every seed (best p {best_right})"
    );
    let data = data_for(5);
    let mut fitted = Clayton::new();
    fitted.fit(data.view()).unwrap();
    let right = gof_cramer_von_mises(
      &fitted,
      data.view(),
      GofConfig {
        replications: 1,
        alpha: 0.05,
      },
      &Deterministic::new(100),
      refit,
    )
    .unwrap();
    let mut gaussian = GaussianCopula::new();
    gaussian.fit(data.view()).unwrap();
    let wrong = gof_cramer_von_mises(
      &gaussian,
      data.view(),
      GofConfig {
        replications: 60,
        alpha: 0.05,
      },
      &Deterministic::new(100),
      |c: &mut GaussianCopula, x| c.fit(x),
    )
    .unwrap();
    assert!(wrong.p_value < 0.05, "wrong family p {}", wrong.p_value);
    assert!(wrong.statistic > right.statistic);
  }

  /// The transform is `(v, ∂_v C(u, v))`, the trait's h-function.
  #[test]
  fn rosenblatt_conditions_on_the_second_coordinate() {
    let copula = Clayton {
      theta: Some(2.0),
      ..Clayton::new()
    };
    let x = ndarray::array![[0.3, 0.7], [0.5, 0.2], [0.9, 0.4]];
    let r = rosenblatt(&copula, x.view()).unwrap();
    assert_eq!(r.column(0), x.column(1));
    assert_eq!(r.column(1), copula.partial_derivative(x.view()).unwrap());
  }

  #[test]
  fn the_decision_is_the_p_value_against_alpha() {
    let data = data_for(5);
    let mut gaussian = GaussianCopula::new();
    gaussian.fit(data.view()).unwrap();
    let config = GofConfig {
      replications: 60,
      alpha: 0.05,
    };
    let wrong = gof_cramer_von_mises(
      &gaussian,
      data.view(),
      config,
      &Deterministic::new(100),
      |c: &mut GaussianCopula, x| c.fit(x),
    )
    .unwrap();
    assert_eq!(
      wrong.rejects_null,
      wrong.p_value < 0.05,
      "p {}",
      wrong.p_value
    );
    assert_eq!(wrong.null_rejected(), Some(wrong.rejects_null));
    assert_eq!((wrong.replications, wrong.alpha), (60, 0.05));
    let loose = gof_cramer_von_mises(
      &gaussian,
      data.view(),
      GofConfig {
        replications: 60,
        alpha: 1e-6,
      },
      &Deterministic::new(100),
      |c: &mut GaussianCopula, x| c.fit(x),
    )
    .unwrap();
    assert_eq!(loose.rejects_null, loose.p_value < 1e-6);
    let kept = [5u64, 6, 7].into_iter().any(|seed| {
      let data = data_for(seed);
      let mut fitted = Clayton::new();
      fitted.fit(data.view()).unwrap();
      let config = GofConfig {
        replications: 60,
        alpha: 0.05,
      };
      !gof_cramer_von_mises(
        &fitted,
        data.view(),
        config,
        &Deterministic::new(100),
        |c: &mut Clayton, x| c.fit(x),
      )
      .unwrap()
      .rejects_null
    });
    assert!(kept, "the true family was rejected on every seed");
  }

  #[test]
  #[should_panic(expected = "replications must satisfy `replications > 0`")]
  fn zero_replications_is_a_precondition() {
    let copula = Clayton {
      theta: Some(2.0),
      ..Clayton::new()
    };
    let data = data_for(5);
    let config = GofConfig {
      replications: 0,
      alpha: 0.05,
    };
    let _ = gof_cramer_von_mises(
      &copula,
      data.view(),
      config,
      &Deterministic::new(1),
      |c: &mut Clayton, x| c.fit(x),
    );
  }

  #[test]
  #[should_panic(expected = "alpha must satisfy `0.0 < alpha < 1.0`")]
  fn alpha_outside_the_unit_interval_is_a_precondition() {
    let copula = Clayton {
      theta: Some(2.0),
      ..Clayton::new()
    };
    let data = data_for(5);
    let config = GofConfig {
      replications: 10,
      alpha: 1.0,
    };
    let _ = gof_cramer_von_mises(
      &copula,
      data.view(),
      config,
      &Deterministic::new(1),
      |c: &mut Clayton, x| c.fit(x),
    );
  }

  #[test]
  fn pseudo_observations_are_normalised_ranks() {
    let data = ndarray::array![[3.0, 10.0], [1.0, 30.0], [2.0, 20.0]];
    let u = pseudo_observations(data.view());
    assert_eq!(u[(0, 0)], 0.75);
    assert_eq!(u[(1, 0)], 0.25);
    assert_eq!(u[(1, 1)], 0.75);
  }

  #[test]
  fn a_gof_result_is_a_hypothesis_test() {
    let result = GofResult {
      statistic: 0.042,
      p_value: 0.31,
      replications: 200,
      alpha: 0.05,
      rejects_null: false,
    };
    assert_eq!(result.statistic(), 0.042);
    assert_eq!(result.null_rejected(), Some(false));
  }
}
