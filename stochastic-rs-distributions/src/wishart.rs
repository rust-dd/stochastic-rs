//! # Wishart distribution
//!
//! $$
//! X \sim \mathcal{W}_p(\nu, V)
//! \iff X = \sum_{i=1}^{\nu} Z_i Z_i^\top,\ Z_i \sim \mathcal{N}_p(0, V),
//! $$
//!
//! for $\nu > p - 1$ degrees of freedom and a $p \times p$ positive-definite
//! scale matrix $V$. The PDF on the cone of $p \times p$ SPD matrices is
//!
//! $$
//! f(X) = \frac{|X|^{(\nu - p - 1)/2}\,
//!              \exp(-\tfrac{1}{2} \mathrm{tr}(V^{-1} X))}
//!             {2^{\nu p / 2}\, |V|^{\nu/2}\, \Gamma_p(\nu/2)},
//! $$
//!
//! with the multivariate gamma $\Gamma_p(a) = \pi^{p(p-1)/4} \prod_{j=1}^{p} \Gamma(a - (j-1)/2)$.
//!
//! ## Sampling — Bartlett (1933) decomposition
//!
//! For integer $\nu$, the **Bartlett decomposition** avoids the
//! $\nu$-fold outer product. Let $L$ be the Cholesky factor of $V$. Build
//! a lower-triangular $A$ with
//!
//! - diagonal $A_{jj} \sim \chi(\nu - j + 1)$, i.e. $\sqrt{\chi^2_{\nu - j + 1}}$;
//! - sub-diagonal $A_{ij} \sim \mathcal{N}(0, 1)$ for $i > j$.
//!
//! Then $X = L A A^\top L^\top \sim \mathcal{W}_p(\nu, V)$.
//!
//! ## Inverse-Wishart
//!
//! If $X \sim \mathcal{W}_p(\nu, V)$ then $X^{-1} \sim
//! \mathcal{IW}_p(\nu, V^{-1})$. Use [`SimdWishart::sample_inverse`] for
//! the Bayesian-prior variant.
//!
//! References:
//! - Bartlett, M.S. (1933), "On the theory of statistical regression",
//!   *Proceedings of the Royal Society of Edinburgh* 53, 260-283.
//! - Smith, W.B., Hocking, R.R. (1972), "Algorithm AS 53: Wishart
//!   variate generator", *Applied Statistics* 21(3), 341-345.

use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::chi_square::SimdChiSquared;
use crate::normal::SimdNormal;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;
use crate::traits::SimdFloatExt;

pub struct SimdWishart<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  /// Degrees of freedom $\nu > p - 1$.
  nu: f64,
  /// Dimensionality $p$.
  p: usize,
  /// Lower-triangular Cholesky factor of the scale matrix $V$.
  chol: Array2<f64>,
  /// Per-row diagonal $\chi^2_{\nu - j + 1}$ samplers (one per dim).
  diag_chi: Vec<SimdChiSquared<T, R>>,
  /// Auxiliary standard normal for off-diagonal Bartlett entries.
  normal: SimdNormal<T, 64, R>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdWishart<T, R> {
  /// Construct a Wishart$(\nu, V)$ generator.
  ///
  /// `scale` is the $p \times p$ positive-definite scale matrix; the
  /// constructor Cholesky-factorises it eagerly so subsequent draws are
  /// cheap. Returns a panic on a non-SPD scale matrix or on $\nu \le p - 1$.
  pub fn new<S: SeedExt>(nu: f64, scale: Array2<f64>, seed: &S) -> Self {
    let p = scale.nrows();
    assert_eq!(scale.ncols(), p, "scale must be square");
    assert!(nu > (p - 1) as f64, "Wishart needs ν > p − 1");

    let chol = cholesky_lower(&scale).expect("scale matrix must be positive definite");

    let diag_chi: Vec<SimdChiSquared<T, R>> = (0..p)
      .map(|j| SimdChiSquared::<T, R>::new(T::from_f64_fast(nu - j as f64), seed))
      .collect();
    let normal = SimdNormal::<T, 64, R>::new(T::zero(), T::one(), seed);

    Self {
      nu,
      p,
      chol,
      diag_chi,
      normal,
    }
  }

  pub fn dim(&self) -> usize {
    self.p
  }

  /// Bartlett-decomposition sample: returns a $p \times p$ Wishart matrix.
  pub fn sample_fast(&self) -> Array2<f64> {
    // Build A: lower-triangular with χ² diagonal and N(0, 1) sub-diagonal.
    let mut a = Array2::<f64>::zeros((self.p, self.p));
    for j in 0..self.p {
      let chi2_j = self.diag_chi[j].sample_fast().to_f64().unwrap();
      a[[j, j]] = chi2_j.max(0.0).sqrt();
      for i in (j + 1)..self.p {
        a[[i, j]] = self.normal.sample_fast().to_f64().unwrap();
      }
    }
    // X = L · A · Aᵀ · Lᵀ. Compute step by step: M = L · A, then X = M · Mᵀ.
    let m = self.chol.dot(&a);
    m.dot(&m.t())
  }

  /// Inverse-Wishart sample: if $X \sim \mathcal{W}_p(\nu, V)$ then
  /// $X^{-1} \sim \mathcal{IW}_p(\nu, V^{-1})$. We sample $X$ and invert.
  pub fn sample_inverse(&self) -> Option<Array2<f64>> {
    let x = self.sample_fast();
    invert_spd(&x)
  }

  /// Log-density of `X` (must be SPD). NaN if `X` is not positive definite
  /// (Cholesky fails); otherwise the closed-form Wishart log-PDF.
  pub fn log_pdf(&self, x: &Array2<f64>) -> f64 {
    let nu = self.nu;
    let p = self.p as f64;
    let det_x_log = match cholesky_lower(x) {
      Some(l) => 2.0 * (0..self.p).map(|i| l[[i, i]].ln()).sum::<f64>(),
      None => return f64::NAN,
    };
    let v = self.chol.dot(&self.chol.t());
    let det_v_log = 2.0 * (0..self.p).map(|i| self.chol[[i, i]].ln()).sum::<f64>();
    let v_inv = invert_spd(&v).expect("scale matrix should be invertible");
    let tr_vinv_x: f64 = (0..self.p)
      .map(|i| (0..self.p).map(|j| v_inv[[i, j]] * x[[j, i]]).sum::<f64>())
      .sum();
    let log_gamma_p: f64 = (0..self.p)
      .map(|j| crate::special::ln_gamma(0.5 * (nu - j as f64)))
      .sum();
    let log_norm = -0.5 * nu * p * std::f64::consts::LN_2
      - 0.5 * nu * det_v_log
      - 0.25 * p * (p - 1.0) * std::f64::consts::PI.ln()
      - log_gamma_p;
    log_norm + 0.5 * (nu - p - 1.0) * det_x_log - 0.5 * tr_vinv_x
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdWishart<T, R> {
  fn clone(&self) -> Self {
    let v = self.chol.dot(&self.chol.t());
    Self::new(self.nu, v, &Unseeded)
  }
}

/// Plain in-place Cholesky decomposition (no external linalg dependency
/// needed for this 2.3.0 implementation; dim ≤ 10 in practice for
/// portfolio / factor-model use cases).
fn cholesky_lower(a: &Array2<f64>) -> Option<Array2<f64>> {
  let n = a.nrows();
  let mut l = Array2::<f64>::zeros((n, n));
  for i in 0..n {
    for j in 0..=i {
      let mut sum = 0.0;
      for k in 0..j {
        sum += l[[i, k]] * l[[j, k]];
      }
      if i == j {
        let diag = a[[i, i]] - sum;
        if diag <= 0.0 {
          return None;
        }
        l[[i, i]] = diag.sqrt();
      } else {
        l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
      }
    }
  }
  Some(l)
}

/// Invert a positive-definite matrix via L · Lᵀ Cholesky → forward / back
/// substitution. Returns `None` if `a` is not SPD.
fn invert_spd(a: &Array2<f64>) -> Option<Array2<f64>> {
  let n = a.nrows();
  let l = cholesky_lower(a)?;
  let mut inv = Array2::<f64>::zeros((n, n));
  for col in 0..n {
    let mut y = vec![0.0_f64; n];
    let mut x = vec![0.0_f64; n];
    // L · y = e_col
    for i in 0..n {
      let mut s = if i == col { 1.0 } else { 0.0 };
      for k in 0..i {
        s -= l[[i, k]] * y[k];
      }
      y[i] = s / l[[i, i]];
    }
    // Lᵀ · x = y  (back substitution).
    for i in (0..n).rev() {
      let mut s = y[i];
      for k in (i + 1)..n {
        s -= l[[k, i]] * x[k];
      }
      x[i] = s / l[[i, i]];
    }
    for i in 0..n {
      inv[[i, col]] = x[i];
    }
  }
  Some(inv)
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  /// Wishart samples are symmetric positive semi-definite.
  #[test]
  fn wishart_samples_are_spd() {
    let v = array![[2.0, 0.5], [0.5, 1.0]];
    let w = SimdWishart::<f64>::new(5.0, v, &Unseeded);
    for _ in 0..200 {
      let x = w.sample_fast();
      // Symmetry.
      for i in 0..2 {
        for j in 0..2 {
          assert!(
            (x[[i, j]] - x[[j, i]]).abs() < 1e-12,
            "asymmetry at ({i}, {j})"
          );
        }
      }
      // SPD via Cholesky.
      assert!(cholesky_lower(&x).is_some(), "non-SPD sample");
    }
  }

  /// Sample mean equals $\nu V$ (Wishart first moment).
  #[test]
  fn wishart_sample_mean_matches_nu_v() {
    let v = array![[2.0, 0.5], [0.5, 1.0]];
    let nu = 8.0;
    let w = SimdWishart::<f64>::new(nu, v.clone(), &Unseeded);
    let n = 5_000;
    let mut acc = Array2::<f64>::zeros((2, 2));
    for _ in 0..n {
      acc += &w.sample_fast();
    }
    let mean = acc.mapv(|x| x / n as f64);
    for i in 0..2 {
      for j in 0..2 {
        let expected = nu * v[[i, j]];
        let err = (mean[[i, j]] - expected).abs();
        assert!(
          err < 0.4,
          "Wishart mean[{i}, {j}] = {} vs expected {} (err {err})",
          mean[[i, j]],
          expected
        );
      }
    }
  }

  /// Inverse-Wishart sampling produces SPD inverses.
  #[test]
  fn inverse_wishart_samples_are_spd() {
    let v = array![[2.0, 0.5], [0.5, 1.0]];
    let w = SimdWishart::<f64>::new(6.0, v, &Unseeded);
    for _ in 0..100 {
      let x_inv = w.sample_inverse().expect("invertible");
      assert!(cholesky_lower(&x_inv).is_some(), "non-SPD inverse");
    }
  }

  /// Log-pdf returns NaN on non-SPD inputs.
  #[test]
  fn wishart_log_pdf_nan_on_non_spd() {
    let v = array![[1.0, 0.0], [0.0, 1.0]];
    let w = SimdWishart::<f64>::new(3.0, v, &Unseeded);
    // Non-SPD test matrix (negative eigenvalue).
    let bad = array![[1.0, 2.0], [2.0, 1.0]];
    assert!(w.log_pdf(&bad).is_nan());
  }
}
