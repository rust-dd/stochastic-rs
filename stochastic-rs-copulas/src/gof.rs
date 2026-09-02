//! # Goodness of fit
//!
//! $$
//! S_n = \sum_{i=1}^n \bigl(C_n(\mathbf R_i) - R_{i1}R_{i2}\bigr)^2,\qquad
//! \mathbf R_i = \bigl(U_{i1},\ \partial_u C(U_{i1}, U_{i2})\bigr)
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
//! References: Genest, C., Rémillard, B. & Beaudoin, D. (2009),
//! *Goodness-of-fit tests for copulas: a review and a power study*,
//! Insurance: Mathematics and Economics 44(2), 199–213, §3.3 (`S_n^{(C)}`)
//! and §4 (parametric bootstrap); Rosenblatt, M. (1952), *Remarks on a
//! multivariate transformation*, Ann. Math. Statist. 23, 470–472.

use std::error::Error;

use ndarray::Array2;
use ndarray::Axis;
use ndarray::stack;

use crate::traits::BivariateExt;

/// Rosenblatt transform `(u, ∂_u C(u, v))` of the rows of `x`. The crate's
/// bivariate families are exchangeable, so `∂_u C(u, v) = ∂_v C(v, u)` is
/// read off the trait's `∂_v` h-function with the arguments swapped.
pub fn rosenblatt<C: BivariateExt + ?Sized>(
  copula: &C,
  x: &Array2<f64>,
) -> Result<Array2<f64>, Box<dyn Error>> {
  let u = x.column(0).to_owned();
  let v = x.column(1).to_owned();
  let swapped = stack![Axis(1), v, u];
  let conditional = copula.partial_derivative(&swapped)?;
  Ok(stack![Axis(1), u, conditional])
}

/// Cramér–von Mises distance between the empirical copula of `r` and the
/// product copula, `Σ_i (C_n(R_i) − R_{i1} R_{i2})²`.
pub fn cramer_von_mises_independence(r: &Array2<f64>) -> f64 {
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

/// Result of the bootstrap goodness-of-fit test.
#[derive(Clone, Debug, PartialEq)]
pub struct GofResult {
  /// Statistic on the data.
  pub statistic: f64,
  /// Bootstrap p-value: share of bootstrap statistics at or above the data's.
  pub p_value: f64,
  /// Number of bootstrap replications.
  pub replications: usize,
}

/// Parametric-bootstrap Cramér–von Mises test of `copula` (already fitted)
/// on the pseudo-observations `x` (Genest, Rémillard & Beaudoin 2009, §4):
/// `replications` samples of size `n` are drawn from the fitted copula with
/// the seeds `seed, seed + 1, …`, each is rank-transformed to
/// pseudo-observations (their Step 2(a) — the statistic is a rank
/// statistic, so the bootstrap must see ranks as the data did), refitted
/// with `refit` and transformed, and the p-value is the share of bootstrap
/// statistics at or above the data's.
pub fn gof_cramer_von_mises<C: BivariateExt + Clone>(
  copula: &C,
  x: &Array2<f64>,
  replications: usize,
  seed: u64,
  refit: impl Fn(&mut C, &Array2<f64>) -> Result<(), Box<dyn Error>>,
) -> Result<GofResult, Box<dyn Error>> {
  let statistic = cramer_von_mises_independence(&rosenblatt(copula, x)?);
  let n = x.nrows();
  let mut exceed = 0usize;
  for b in 0..replications {
    let sample = pseudo_observations(&copula.sample_with_seed(n, seed.wrapping_add(b as u64))?);
    let mut refitted = copula.clone();
    refit(&mut refitted, &sample)?;
    let s_b = cramer_von_mises_independence(&rosenblatt(&refitted, &sample)?);
    if s_b >= statistic {
      exceed += 1;
    }
  }
  Ok(GofResult {
    statistic,
    p_value: exceed as f64 / replications as f64,
    replications,
  })
}

/// Pseudo-observations (normalised ranks `rank / (n + 1)`) of raw data.
pub fn pseudo_observations(data: &Array2<f64>) -> Array2<f64> {
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
  use super::*;
  use crate::bivariate::clayton::Clayton;
  use crate::bivariate::gaussian::GaussianCopula;

  #[test]
  fn rosenblatt_of_the_true_copula_is_uniform_and_independent() {
    let mut copula = Clayton::new();
    copula.set_tau(0.5); // θ = 2τ/(1 − τ) = 2
    copula.set_theta(copula.compute_theta());
    let sample = copula.sample_with_seed(3000, 3).unwrap();
    let r = rosenblatt(&copula, &sample).unwrap();
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
    assert!(cramer_von_mises_independence(&r) < 0.2);
  }

  /// A Gaussian copula fitted to Clayton data is rejected; the true family
  /// is not. The true-family p-value is uniform under the null, so three
  /// pinned data seeds are tried and the best is asserted on.
  #[test]
  fn bootstrap_test_rejects_the_wrong_family() {
    let data_for = |seed: u64| {
      let mut truth = Clayton::new();
      truth.set_tau(0.6); // θ = 3
      truth.set_theta(truth.compute_theta());
      pseudo_observations(&truth.sample_with_seed(600, seed).unwrap())
    };
    let refit = |c: &mut Clayton, x: &Array2<f64>| c.fit(x);
    let best_right = [5u64, 6, 7]
      .into_iter()
      .map(|seed| {
        let data = data_for(seed);
        let mut fitted = Clayton::new();
        fitted.fit(&data).unwrap();
        gof_cramer_von_mises(&fitted, &data, 60, 100, refit)
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
    fitted.fit(&data).unwrap();
    let right = gof_cramer_von_mises(&fitted, &data, 1, 100, refit).unwrap();
    let mut gaussian = GaussianCopula::new();
    gaussian.fit(&data).unwrap();
    let wrong = gof_cramer_von_mises(&gaussian, &data, 60, 100, |c: &mut GaussianCopula, x| {
      c.fit(x)
    })
    .unwrap();
    assert!(wrong.p_value < 0.05, "wrong family p {}", wrong.p_value);
    assert!(wrong.statistic > right.statistic);
  }

  #[test]
  fn pseudo_observations_are_normalised_ranks() {
    let data = ndarray::array![[3.0, 10.0], [1.0, 30.0], [2.0, 20.0]];
    let u = pseudo_observations(&data);
    assert_eq!(u[(0, 0)], 0.75);
    assert_eq!(u[(1, 0)], 0.25);
    assert_eq!(u[(1, 1)], 0.75);
  }
}
