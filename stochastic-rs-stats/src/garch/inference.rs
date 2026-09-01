//! Finite-difference Hessian and outer-product-of-scores at the optimum,
//! combined into the inverse-Hessian and Bollerslev–Wooldridge sandwich
//! covariances.

use ndarray::Array2;

use super::GarchSpec;
use super::recursion;
use crate::linalg::inverse;

pub(super) struct Inference {
  /// $(-H)^{-1}$.
  pub covariance: Array2<f64>,
  /// $H^{-1} J H^{-1}$.
  pub robust_covariance: Array2<f64>,
}

/// Central-difference Hessian of the summed log-likelihood and per-observation
/// scores, with steps $10^{-4}\max(|\theta_i|, \text{scale}_i)$.
pub(super) fn sandwich(
  spec: &GarchSpec,
  natural: &[f64],
  returns: &[f64],
  backcast: f64,
  scales: &[f64],
) -> Inference {
  let k = natural.len();
  let n = returns.len();
  let steps: Vec<f64> = (0..k)
    .map(|i| 1e-4 * natural[i].abs().max(scales[i]))
    .collect();
  let shifted = |i: usize, si: f64, j: usize, sj: f64| {
    let mut v = natural.to_vec();
    v[i] += si;
    v[j] += sj;
    v
  };
  let ll = |v: &[f64]| recursion::total_log_likelihood(spec, v, returns, backcast);

  let f0 = ll(natural);
  let mut hessian = Array2::<f64>::zeros((k, k));
  for i in 0..k {
    let h = steps[i];
    hessian[[i, i]] =
      (ll(&shifted(i, h, i, 0.0)) - 2.0 * f0 + ll(&shifted(i, -h, i, 0.0))) / (h * h);
    for j in (i + 1)..k {
      let g = steps[j];
      let mixed =
        (ll(&shifted(i, h, j, g)) - ll(&shifted(i, h, j, -g)) - ll(&shifted(i, -h, j, g))
          + ll(&shifted(i, -h, j, -g)))
          / (4.0 * h * g);
      hessian[[i, j]] = mixed;
      hessian[[j, i]] = mixed;
    }
  }

  let mut scores = Array2::<f64>::zeros((n, k));
  for i in 0..k {
    let h = steps[i];
    let (_, plus) =
      recursion::log_likelihood_terms(spec, &shifted(i, h, i, 0.0), returns, backcast);
    let (_, minus) =
      recursion::log_likelihood_terms(spec, &shifted(i, -h, i, 0.0), returns, backcast);
    for t in 0..n {
      scores[[t, i]] = (plus[t] - minus[t]) / (2.0 * h);
    }
  }
  let opg = scores.t().dot(&scores);

  let neg_hessian = -&hessian;
  let covariance = inverse(&neg_hessian).unwrap_or_else(|| Array2::from_elem((k, k), f64::NAN));
  let robust_covariance = covariance.dot(&opg).dot(&covariance);
  Inference {
    covariance,
    robust_covariance,
  }
}
