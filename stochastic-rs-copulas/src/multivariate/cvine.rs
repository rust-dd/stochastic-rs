//! # Canonical vine (C-vine) pair-copula construction
//!
//! $$
//! c(u_1, \dots, u_d) \;=\; \prod_{m=1}^{d-1} \prod_{i=1}^{d-m}
//!     c_{m,\,m+i \mid 1,\dots,m-1}\!\bigl(
//!         F(u_m \mid u_1, \dots, u_{m-1}),\;
//!         F(u_{m+i} \mid u_1, \dots, u_{m-1})
//!     \bigr).
//! $$
//!
//! In the C-vine, tree $T_m$ has variable $m$ as its **single root**: all
//! edges of $T_m$ emanate from node $m$. This contrasts with the D-vine's
//! path-shaped trees ([`super::dvine::DVine`]).
//!
//! The pair-copula tree is stored identically to [`super::dvine::DVine`]:
//! `pair_copulas[m][i]` is the $(m+1, m+i+2 \mid 1, \dots, m)$ edge of
//! tree $T_{m+1}$ (1-based on paper; 0-based in storage).
//!
//! ## Sampling — Aas-Czado-Frigessi-Bakken (2009), Algorithm 3
//!
//! Independent uniforms $w_1, \dots, w_d$ are inverted through nested
//! $h^{-1}$ calls anchored at the root variable of each tree.
//!
//! ## Density — direct chain
//!
//! Apply `PairCopula::log_density` at every edge after the conditional
//! pseudo-observations are propagated via `h(\cdot, \cdot)` along the
//! root chain.
//!
//! References: Aas-Czado-Frigessi-Bakken (2009), §4 and Algorithm 3.

use std::error::Error;

use ndarray::Array1;
use ndarray::Array2;
use rand::Rng;

use super::CopulaType;
use super::dvine::PairCopula;
use crate::traits::MultivariateExt;

/// Canonical vine (C-vine) pair-copula construction over `dim` marginals.
///
/// `pair_copulas[m][i]` is the bivariate pair-copula on edge
/// $(m+1,\,m+i+2 \mid 1,\dots,m)$ of tree $T_{m+1}$. Outer length = $d-1$;
/// inner length of tree $m$ is $d-1-m$.
#[derive(Debug, Clone)]
pub struct CVine {
  dim: usize,
  pair_copulas: Vec<Vec<PairCopula>>,
}

impl CVine {
  /// Build a C-vine from a precomputed tree of pair copulas. Identical
  /// shape contract to [`super::dvine::DVine::new`].
  pub fn new(dim: usize, pair_copulas: Vec<Vec<PairCopula>>) -> Result<Self, Box<dyn Error>> {
    if dim < 2 {
      return Err(format!("C-vine requires dim ≥ 2, got {dim}").into());
    }
    if pair_copulas.len() != dim - 1 {
      return Err(
        format!(
          "Expected {} trees for dim={dim}, got {}",
          dim - 1,
          pair_copulas.len()
        )
        .into(),
      );
    }
    for (m, tree) in pair_copulas.iter().enumerate() {
      let expected = dim - 1 - m;
      if tree.len() != expected {
        return Err(
          format!(
            "Tree T_{} should have {expected} edges (got {})",
            m + 1,
            tree.len()
          )
          .into(),
        );
      }
    }
    Ok(Self { dim, pair_copulas })
  }

  /// All-Independence C-vine of the given dimension.
  pub fn independence(dim: usize) -> Result<Self, Box<dyn Error>> {
    let pc: Vec<Vec<PairCopula>> = (0..dim - 1)
      .map(|m| vec![PairCopula::Independence; dim - 1 - m])
      .collect();
    Self::new(dim, pc)
  }

  /// Dimensionality (number of marginals).
  pub fn dim(&self) -> usize {
    self.dim
  }

  /// Borrowed pair-copula tree.
  pub fn pair_copulas(&self) -> &[Vec<PairCopula>] {
    &self.pair_copulas
  }

  /// Aas-Czado (2009) Algorithm 3: invert independent uniforms $w$ to a
  /// single C-vine observation $u$. The pyramid `v[i][j]` stores
  /// conditional pseudo-observations at increasing conditioning depth.
  fn sample_one<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
    let d = self.dim;
    let w: Vec<f64> = (0..d).map(|_| rng.random::<f64>()).collect();
    let mut u = Array1::<f64>::zeros(d);
    let mut v = vec![vec![0.0_f64; d + 1]; d + 1];

    u[0] = w[0];
    v[0][0] = w[0];

    for i in 1..d {
      v[i][0] = w[i];
      // Invert through all preceding trees from T_i down to T_1.
      for k in (0..i).rev() {
        let cop = self.pair_copulas[k][i - k - 1];
        v[i][0] = cop.h_inverse(v[i][0], v[k][k]);
      }
      u[i] = v[i][0];
      if i == d - 1 {
        break;
      }
      // Forward h-pass through trees T_1, ..., T_i for the conditioning
      // values at column i.
      for j in 0..i {
        let cop = self.pair_copulas[j][i - j - 1];
        v[i][j + 1] = cop.h(v[i][j], v[j][j]);
      }
    }
    u
  }

  /// Log-density at a single observation. Recursive structure mirrors the
  /// sampling pyramid: traverse trees $T_1, \dots, T_{d-1}$, summing the
  /// edge log-densities while propagating conditional pseudo-observations.
  fn log_density_one(&self, u: &[f64]) -> f64 {
    let d = self.dim;
    let mut v = vec![vec![0.0_f64; d + 1]; d + 1];
    let mut log_c = 0.0_f64;
    for (i, &ui) in u.iter().enumerate() {
      v[i][0] = ui;
    }
    // Tree T_m has root variable m+1 (0-based: index m).
    for m in 0..d - 1 {
      for i in 0..d - 1 - m {
        let cop = self.pair_copulas[m][i];
        log_c += cop.log_density(v[m][m], v[m + i + 1][m]);
      }
      if m + 1 == d - 1 {
        break;
      }
      // Forward h-pass to produce the conditional pseudo-observations
      // needed by the next tree T_{m+2}.
      for i in 0..d - 1 - m {
        let cop = self.pair_copulas[m][i];
        v[m + i + 1][m + 1] = cop.h(v[m + i + 1][m], v[m][m]);
      }
    }
    log_c
  }
}

impl MultivariateExt for CVine {
  fn r#type(&self) -> CopulaType {
    CopulaType::CVine
  }

  fn sample(&self, n: usize) -> Result<Array2<f64>, Box<dyn Error>> {
    let mut rng = rand::rng();
    let mut out = Array2::<f64>::zeros((n, self.dim));
    for r in 0..n {
      let row = self.sample_one(&mut rng);
      for c in 0..self.dim {
        out[[r, c]] = row[c].clamp(1e-12, 1.0 - 1e-12);
      }
    }
    Ok(out)
  }

  fn fit(&mut self, _X: Array2<f64>) -> Result<(), Box<dyn Error>> {
    Err(
      "CVine::fit not implemented — supply the tree explicitly via CVine::new \
       and seed each PairCopula parameter from pairwise Kendall τ. Sequential MLE + \
       AIC/BIC family selection (Dißmann 2013) is not yet implemented."
        .into(),
    )
  }

  fn check_fit(&self, X: &Array2<f64>) -> Result<(), Box<dyn Error>> {
    if X.ncols() != self.dim {
      return Err(
        format!(
          "Dimension mismatch: X has {} columns, C-vine has dim {}",
          X.ncols(),
          self.dim
        )
        .into(),
      );
    }
    if X.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
      return Err("Input X must be in [0,1] for the C-vine".into());
    }
    Ok(())
  }

  fn pdf(&self, X: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit(&X)?;
    let mut out = Array1::<f64>::zeros(X.nrows());
    for (i, row) in X.rows().into_iter().enumerate() {
      let u: Vec<f64> = row.iter().copied().collect();
      out[i] = self.log_density_one(&u).exp();
    }
    Ok(out)
  }

  fn log_pdf(&self, X: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit(&X)?;
    let mut out = Array1::<f64>::zeros(X.nrows());
    for (i, row) in X.rows().into_iter().enumerate() {
      let u: Vec<f64> = row.iter().copied().collect();
      out[i] = self.log_density_one(&u);
    }
    Ok(out)
  }

  fn cdf(&self, X: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    // MC estimator on 4000 samples per query (same approach as D-vine and
    // multivariate t-copula): closed-form C-vine CDF requires nested
    // numerical integration in d ≥ 3 dims.
    self.check_fit(&X)?;
    let m = 4_000usize;
    let sample = self.sample(m)?;
    let mut out = Array1::<f64>::zeros(X.nrows());
    for (i, row) in X.rows().into_iter().enumerate() {
      let u: Vec<f64> = row.iter().copied().collect();
      let mut count = 0usize;
      for r in 0..m {
        let mut all_le = true;
        for c in 0..self.dim {
          if sample[[r, c]] > u[c] {
            all_le = false;
            break;
          }
        }
        if all_le {
          count += 1;
        }
      }
      out[i] = count as f64 / m as f64;
    }
    Ok(out)
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  /// All-Independence C-vine on d=3: marginals uniform, log-density ≡ 0.
  #[test]
  fn cvine_independence_three_dim_uniform_marginals() {
    let cv = CVine::independence(3).unwrap();
    let s = cv.sample(10_000).unwrap();
    for j in 0..3 {
      let col = s.column(j);
      let m: f64 = col.iter().sum::<f64>() / col.len() as f64;
      assert!(
        (m - 0.5).abs() < 0.02,
        "marginal {j} mean = {m}, expected ~0.5"
      );
    }
    let lp = cv.log_pdf(array![[0.2, 0.4, 0.7]]).unwrap();
    assert!(lp[0].abs() < 1e-12);
  }

  /// 2-dim C-vine reduces to its single edge, matching the chosen
  /// PairCopula on the bivariate dependence.
  #[test]
  fn cvine_two_dim_clayton_matches_bivariate() {
    let cv = CVine::new(2, vec![vec![PairCopula::Clayton { theta: 2.0 }]]).unwrap();
    let s = cv.sample(10_000).unwrap();
    use crate::correlation::kendall_tau;
    let tau = kendall_tau(&s);
    // Clayton(θ=2): τ = θ/(θ+2) = 0.5
    assert!(
      (tau[[0, 1]] - 0.5).abs() < 0.04,
      "2-dim Clayton(θ=2) C-vine τ_(0,1) = {}, expected ~0.5",
      tau[[0, 1]]
    );
  }

  /// 3-dim C-vine with Gaussian pair-copulas T_1 = [ρ_12, ρ_13] and
  /// independence T_2 = [(2,3|1)]: variables 2 and 3 are conditionally
  /// independent given 1, and both share strong dependence with the root.
  #[test]
  fn cvine_three_dim_gaussian_star() {
    let t1 = vec![
      PairCopula::Gaussian { rho: 0.6 }, // (1, 2)
      PairCopula::Gaussian { rho: 0.6 }, // (1, 3)
    ];
    let t2 = vec![PairCopula::Independence]; // (2, 3 | 1)
    let cv = CVine::new(3, vec![t1, t2]).unwrap();
    let s = cv.sample(10_000).unwrap();
    use crate::correlation::kendall_tau;
    let tau = kendall_tau(&s);
    let expected_tau = (2.0 / std::f64::consts::PI) * 0.6_f64.asin();
    // (1, 2) and (1, 3) should both match the Gaussian τ identity.
    assert!(
      (tau[[0, 1]] - expected_tau).abs() < 0.03,
      "τ_(0,1) = {} vs expected {expected_tau}",
      tau[[0, 1]]
    );
    assert!(
      (tau[[0, 2]] - expected_tau).abs() < 0.03,
      "τ_(0,2) = {} vs expected {expected_tau}",
      tau[[0, 2]]
    );
    // (2, 3) has the implied "shared root" dependence ≈ ρ² in Gaussian.
    // ρ = 0.36 → τ ≈ 0.234. The conditional copula is independence so
    // the unconditional correlation comes from the shared common factor.
    assert!(
      tau[[1, 2]] > 0.15 && tau[[1, 2]] < 0.32,
      "τ_(1,2) = {} should sit between [0.15, 0.32] (Gaussian common-factor)",
      tau[[1, 2]]
    );
    for j in 0..3 {
      let col = s.column(j);
      let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
      assert!((mean - 0.5).abs() < 0.02);
    }
  }

  /// Shape validation: wrong tree sizes must be rejected.
  #[test]
  fn cvine_shape_validation() {
    assert!(CVine::new(3, vec![vec![PairCopula::Independence]]).is_err());
    assert!(CVine::new(1, vec![]).is_err());
  }

  /// `fit` returns a descriptive not-implemented error.
  #[test]
  fn cvine_fit_rejects_with_descriptive_error() {
    let mut cv = CVine::independence(3).unwrap();
    let data = ndarray::Array2::<f64>::from_elem((10, 3), 0.5);
    let res = cv.fit(data);
    assert!(res.is_err());
    let msg = res.unwrap_err().to_string();
    assert!(
      msg.contains("not implemented") || msg.contains("MLE"),
      "fit error should point at the unimplemented sequential MLE; got: {msg}"
    );
  }
}
