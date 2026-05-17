//! Sample covariance and Ledoit-Wolf shrinkage to scaled identity.
//!
//! $$
//! \hat\Sigma_{LW} = \alpha\,\mu I + (1-\alpha)\,S,\qquad
//! \mu = \mathrm{tr}(S)/p,\quad
//! d^2 = \|S - \mu I\|_F^2 / p,\quad
//! b^2 = \min\!\bigl(d^2,\, \tfrac{1}{p T^2}\sum_t \|X_t X_t^\top - S\|_F^2\bigr),
//! $$
//! with shrinkage intensity $\alpha = b^2 / d^2$.
//!
//! Reference: Ledoit, Wolf, "A Well-Conditioned Estimator for
//! Large-Dimensional Covariance Matrices", Journal of Multivariate Analysis,
//! 88(2), 365-411 (2004). DOI: 10.1016/S0047-259X(03)00096-4

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;

use crate::traits::FloatExt;

/// Result of the Ledoit-Wolf shrinkage estimator.
#[derive(Debug, Clone)]
pub struct LedoitWolfResult<T: FloatExt> {
  /// Shrinkage estimator $\hat\Sigma_{LW}$.
  pub covariance: Array2<T>,
  /// Sample covariance $S$.
  pub sample: Array2<T>,
  /// Optimal shrinkage intensity $\alpha\in[0,1]$.
  pub alpha: T,
  /// Average diagonal $\mu = \mathrm{tr}(S)/p$.
  pub mu: T,
}

/// Sample covariance matrix from an `(observations × assets)` returns matrix.
/// Uses the unbiased divisor `n - 1`.
pub fn sample_covariance<T: FloatExt>(returns: ArrayView2<T>) -> Array2<T> {
  let (t, p) = returns.dim();
  assert!(t >= 2, "need at least two observations for covariance");
  let mut means = Array1::<T>::zeros(p);
  for j in 0..p {
    means[j] = returns.column(j).iter().fold(T::zero(), |a, &v| a + v) / T::from_usize_(t);
  }
  let mut cov = Array2::<T>::zeros((p, p));
  let nm1 = T::from_usize_(t - 1);
  for k in 0..t {
    let row = returns.row(k);
    for i in 0..p {
      let xi = row[i] - means[i];
      for j in i..p {
        let xj = row[j] - means[j];
        cov[[i, j]] += xi * xj;
      }
    }
  }
  for i in 0..p {
    for j in i..p {
      cov[[i, j]] = cov[[i, j]] / nm1;
      if i != j {
        cov[[j, i]] = cov[[i, j]];
      }
    }
  }
  cov
}

/// Ledoit-Wolf (2004) shrinkage of the sample covariance toward the scaled
/// identity target $\mu I$.
///
/// Optimal shrinkage intensity (LW 2004, eq. 14):
/// $\hat\delta = \max(0, \min(1, (\hat\pi - \hat\rho)/(T\hat\gamma)))$
/// with $\hat\pi_{ij} = \tfrac{1}{T}\sum_t (y_{ti}y_{tj} - s_{ij})^2$,
/// $\hat\pi = \sum_{i,j}\hat\pi_{ij}$, $\hat\rho = \sum_i \hat\pi_{ii}$,
/// $\hat\gamma = \|S - \mu I\|_F^2$.
pub fn ledoit_wolf_shrinkage<T: FloatExt>(returns: ArrayView2<T>) -> LedoitWolfResult<T> {
  let (t, p) = returns.dim();
  assert!(
    t >= 2 && p >= 1,
    "need at least two observations and one asset"
  );
  let s = sample_covariance(returns);
  let mut means = Array1::<T>::zeros(p);
  for j in 0..p {
    means[j] = returns.column(j).iter().fold(T::zero(), |a, &v| a + v) / T::from_usize_(t);
  }
  let mut x = Array2::<T>::zeros((t, p));
  for k in 0..t {
    for j in 0..p {
      x[[k, j]] = returns[[k, j]] - means[j];
    }
  }

  let mu = (0..p).fold(T::zero(), |acc, i| acc + s[[i, i]]) / T::from_usize_(p);

  let mut gamma = T::zero();
  for i in 0..p {
    for j in 0..p {
      let target = if i == j { mu } else { T::zero() };
      let d = s[[i, j]] - target;
      gamma += d * d;
    }
  }

  let mut pi_hat = T::zero();
  let mut rho_hat = T::zero();
  let t_inv = T::one() / T::from_usize_(t);
  for i in 0..p {
    for j in 0..p {
      let mut acc = T::zero();
      for k in 0..t {
        let yij = x[[k, i]] * x[[k, j]];
        let d = yij - s[[i, j]];
        acc += d * d;
      }
      let pi_ij = acc * t_inv;
      pi_hat += pi_ij;
      if i == j {
        rho_hat += pi_ij;
      }
    }
  }

  let kappa_over_t = if gamma > T::zero() {
    (pi_hat - rho_hat) / (T::from_usize_(t) * gamma)
  } else {
    T::zero()
  };
  let zero = T::zero();
  let one = T::one();
  let alpha = if kappa_over_t < zero {
    zero
  } else if kappa_over_t > one {
    one
  } else {
    kappa_over_t
  };

  let mut cov = Array2::<T>::zeros((p, p));
  let one_minus = T::one() - alpha;
  for i in 0..p {
    for j in 0..p {
      let target = if i == j { mu } else { T::zero() };
      cov[[i, j]] = alpha * target + one_minus * s[[i, j]];
    }
  }
  LedoitWolfResult {
    covariance: cov,
    sample: s,
    alpha,
    mu,
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  #[test]
  fn sample_covariance_matches_hand_computation() {
    let r = array![[1.0_f64, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]];
    let s = sample_covariance(r.view());
    assert!(approx(s[[0, 0]], 5.0 / 3.0, 1e-12));
    assert!(approx(s[[1, 1]], 20.0 / 3.0, 1e-12));
    assert!(approx(s[[0, 1]], 10.0 / 3.0, 1e-12));
    assert!(approx(s[[1, 0]], s[[0, 1]], 1e-12));
  }

  #[test]
  fn ledoit_wolf_alpha_in_unit_interval() {
    let dist = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(41));
    let mut buf = vec![0.0_f64; 200 * 5];
    dist.fill_slice_fast(&mut buf);
    let r = Array2::from_shape_vec((200, 5), buf).unwrap();
    let lw = ledoit_wolf_shrinkage(r.view());
    assert!(lw.alpha >= 0.0 && lw.alpha <= 1.0);
    assert!(lw.mu > 0.0);
    let trace_lw: f64 = (0..5).map(|i| lw.covariance[[i, i]]).sum();
    let trace_s: f64 = (0..5).map(|i| lw.sample[[i, i]]).sum();
    assert!((trace_lw - trace_s).abs() < 1e-9);
  }

  #[test]
  fn ledoit_wolf_off_diagonals_shrunk_toward_zero() {
    // LW should make off-diagonals smaller in magnitude than the corresponding
    // sample-covariance off-diagonals, by exactly the factor (1 - alpha).
    let dist = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(17));
    let mut buf = vec![0.0_f64; 5 * 20];
    dist.fill_slice_fast(&mut buf);
    let r = Array2::from_shape_vec((5, 20), buf).unwrap();
    let lw = ledoit_wolf_shrinkage(r.view());
    assert!((0.0..=1.0).contains(&lw.alpha));
    for i in 0..20 {
      for j in 0..20 {
        if i != j {
          let expected = (1.0 - lw.alpha) * lw.sample[[i, j]];
          assert!(
            (lw.covariance[[i, j]] - expected).abs() < 1e-12,
            "off-diagonal mismatch at ({i},{j})"
          );
        }
      }
    }
  }

  #[test]
  fn ledoit_wolf_high_shrinkage_when_sample_size_small() {
    // With p >> T, the data-driven shrinkage should be close to 1.
    let dist = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(19));
    let mut buf = vec![0.0_f64; 8 * 30];
    dist.fill_slice_fast(&mut buf);
    let r = Array2::from_shape_vec((8, 30), buf).unwrap();
    let lw = ledoit_wolf_shrinkage(r.view());
    assert!(lw.alpha > 0.3);
  }
}
