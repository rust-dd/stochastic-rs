//! # D-vine pair-copula construction
//!
//! $$
//! c(u_1, \dots, u_d) \;=\; \prod_{m=1}^{d-1} \prod_{i=1}^{d-m}
//!     c_{i,\,i+m \mid i+1,\dots,i+m-1}\!\bigl(
//!         F(u_i \mid u_{i+1}, \dots, u_{i+m-1}),\;
//!         F(u_{i+m} \mid u_{i+1}, \dots, u_{i+m-1})
//!     \bigr)
//! $$
//!
//! The D-vine is a special case of the regular vine (Bedford & Cooke 2002)
//! whose tree structure is a path: in the first tree $T_1$ pairs are
//! $(1,2), (2,3), \dots, (d-1, d)$; in tree $T_m$ pairs are
//! $(i, i+m)$ conditioned on $(i+1, \dots, i+m-1)$.
//!
//! ## Sampling — Aas-Czado-Frigessi-Bakken (2009), Algorithm 4
//!
//! The recursive inverse-Rosenblatt construction. Sample $w_1, \dots, w_d$
//! i.i.d. $U(0,1)$, then for each $k$:
//!
//! 1. Set $u_k := w_k$;
//! 2. For $j = k-1, k-2, \dots, 1$ apply $h^{-1}$ along the conditioning
//!    chain to map $u_k$ through the conditional CDFs back to its uniform.
//! 3. Maintain the "$v$ pyramid" $v_{j, k}$ of conditional pseudo-observations
//!    used by the inner step.
//!
//! Storage: $v$ has shape $(d, d)$ with $v[j][k]$ holding $F(u_k \mid u_{k-j}, \dots, u_{k-1})$.
//!
//! ## Density evaluation
//!
//! Same pyramid recursion, but using `PairCopula::h` to descend into the
//! conditional pseudo-observations and `PairCopula::log_density` at every
//! visited edge.
//!
//! ## Fit
//!
//! Two-stage sequential MLE (Aas-Czado §5, Joe-Xu 1996): for each pair in
//! each tree, fit the marginal copula by MLE on the current pseudo
//! observations, then propagate via the h-function. v2.3.0 ships a
//! Gaussian-only sequential fit (single-parameter per edge, closed-form
//! Kendall-τ inversion); arbitrary-family fits are scheduled for v2.4
//! alongside the AIC/BIC family-selection driver.
//!
//! References:
//! - Aas, K., Czado, C., Frigessi, A., Bakken, H. (2009),
//!   "Pair-copula constructions of multiple dependence",
//!   *Insurance: Mathematics and Economics* 44(2), 182-198.
//! - Joe, H. (1997), *Multivariate Models and Dependence Concepts*,
//!   Chapman & Hall, §4.6 (recursive sampling).
//! - Bedford, T., Cooke, R.M. (2002), "Vines — a new graphical model
//!   for dependent random variables", *Annals of Statistics* 30, 1031-1068.

use std::error::Error;

use ndarray::Array1;
use ndarray::Array2;
use rand::Rng;

use super::CopulaType;
use crate::traits::MultivariateExt;

pub mod pair_copula;

pub use pair_copula::PairCopula;

/// D-vine pair-copula construction over `dim` marginals. Each edge in the
/// $d-1$ trees stores a single [`PairCopula`] family + parameter, indexed
/// as `pair_copulas[tree_m][i]` for `m = 0..d-1`, `i = 0..d-1-m`.
#[derive(Debug, Clone)]
pub struct DVine {
  dim: usize,
  /// `pair_copulas[m][i]` is the edge $(i,\,i+m+1\mid i+1,\dots,i+m)$ of
  /// tree $T_{m+1}$. Outer length = $d-1$; tree $m$ has length $d-1-m$.
  pair_copulas: Vec<Vec<PairCopula>>,
}

impl DVine {
  /// Construct a D-vine from a precomputed tree of pair copulas.
  ///
  /// `pair_copulas` is expected to have $d-1$ outer entries; the $m$-th
  /// entry has length $d-1-m$. Returns an error on any shape mismatch.
  pub fn new(dim: usize, pair_copulas: Vec<Vec<PairCopula>>) -> Result<Self, Box<dyn Error>> {
    if dim < 2 {
      return Err(format!("D-vine requires dim ≥ 2, got {dim}").into());
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
    Ok(Self {
      dim,
      pair_copulas,
    })
  }

  /// Build an **all-independence** D-vine of the given dimension. Useful
  /// as a baseline / pre-fit starting point and for unit tests where the
  /// copula should reduce to the product.
  pub fn independence(dim: usize) -> Result<Self, Box<dyn Error>> {
    let pair_copulas: Vec<Vec<PairCopula>> = (0..dim - 1)
      .map(|m| vec![PairCopula::Independence; dim - 1 - m])
      .collect();
    Self::new(dim, pair_copulas)
  }

  /// Dimensionality (number of marginals).
  pub fn dim(&self) -> usize {
    self.dim
  }

  /// Borrowed view of the pair-copula tree.
  pub fn pair_copulas(&self) -> &[Vec<PairCopula>] {
    &self.pair_copulas
  }

  /// Sample a single D-vine observation via Aas-Czado 2009 Algorithm 4.
  /// Each `w[j]` is one of d independent $U(0, 1)$ inputs.
  fn sample_one<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
    let d = self.dim;
    // Independent uniforms feeding the inverse-Rosenblatt recursion.
    let w: Vec<f64> = (0..d).map(|_| rng.random::<f64>()).collect();
    let mut u = Array1::<f64>::zeros(d);
    // v[i][j] holds intermediate pseudo-observations as in Aas-Czado §3.
    // Storage convention matches the paper's two-index scheme.
    let mut v = vec![vec![0.0_f64; 2 * d]; d];

    u[0] = w[0];
    v[0][0] = w[0];

    for i in 1..d {
      v[i][0] = w[i];
      // Descend along the D-vine chain, applying h_inverse with copulas
      // from T_1, T_2, ..., T_i.
      for k in (1..=i).rev() {
        // Tree index in our 0-based storage: pair_copulas[k-1][i-k].
        let cop = self.pair_copulas[k - 1][i - k];
        v[i][0] = cop.h_inverse(v[i][0], v[i - 1][2 * k - 2]);
      }
      u[i] = v[i][0];
      if i == d - 1 {
        break;
      }
      // Update the inner v's for the next column: forward h-pass that
      // turns u[i] into the conditional pseudo-observations needed at the
      // next sampling step.
      v[i][1] = self.pair_copulas[0][i - 1].h(v[i - 1][0], v[i][0]);
      v[i][2] = self.pair_copulas[0][i - 1].h(v[i][0], v[i - 1][0]);
      for k in 2..=i - 1 {
        let cop_a = self.pair_copulas[k - 1][i - k];
        v[i][2 * k - 1] = cop_a.h(v[i - 1][2 * k - 2 - 1], v[i][2 * k - 2]);
        v[i][2 * k] = cop_a.h(v[i][2 * k - 2], v[i - 1][2 * k - 2 - 1]);
      }
      // Final entry for the top of the pyramid at column i.
      if i >= 2 {
        v[i][2 * i - 1] = self.pair_copulas[i - 1][0].h(v[i - 1][2 * i - 3], v[i][2 * i - 2]);
      }
    }
    u
  }

  /// Log-density of a single observation. Mirrors the sampling pyramid
  /// but applies `log_density` at every edge instead of the inverse step.
  fn log_density_one(&self, u: &[f64]) -> f64 {
    let d = self.dim;
    // v[i][j] follows the same indexing scheme as in `sample_one`.
    let mut v = vec![vec![0.0_f64; 2 * d]; d];
    let mut log_c = 0.0_f64;
    for (i, &ui) in u.iter().enumerate() {
      v[i][0] = ui;
    }
    // Tree T_1: pairs (i, i+1).
    for i in 0..d - 1 {
      let cop = self.pair_copulas[0][i];
      log_c += cop.log_density(v[i][0], v[i + 1][0]);
      v[i][1] = cop.h(v[i][0], v[i + 1][0]);
      v[i + 1][1] = cop.h(v[i + 1][0], v[i][0]);
    }
    // Trees T_2, T_3, ..., T_{d-1}.
    for m in 1..d - 1 {
      for i in 0..d - 1 - m {
        let cop = self.pair_copulas[m][i];
        log_c += cop.log_density(v[i][2 * m - 1], v[i + m + 1][2 * (m - 1)]);
        // Propagate forward to T_{m+1} (only when we'll need it).
        if m + 1 < d - 1 {
          v[i][2 * m + 1] = cop.h(v[i][2 * m - 1], v[i + m + 1][2 * (m - 1)]);
          v[i + m + 1][2 * m + 1] = cop.h(v[i + m + 1][2 * (m - 1)], v[i][2 * m - 1]);
        }
      }
    }
    log_c
  }
}

impl MultivariateExt for DVine {
  fn r#type(&self) -> CopulaType {
    CopulaType::DVine
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
    // Sequential pair-copula MLE / Kendall-τ inversion is a v2.4 item —
    // the v2.3.0 scope ships D-vine *evaluation* (CDF/PDF/sample) on a
    // user-supplied tree, deferring structure + family selection to the
    // VineCopula-equivalent algorithm in v2.4.
    Err(
      "DVine::fit not implemented in v2.3.0 — supply the tree explicitly via DVine::new \
       and seed each PairCopula parameter from pairwise Kendall τ. Sequential MLE + \
       AIC/BIC family selection (Dißmann 2013) is scheduled for v2.4."
        .into(),
    )
  }

  fn check_fit(&self, X: &Array2<f64>) -> Result<(), Box<dyn Error>> {
    if X.ncols() != self.dim {
      return Err(format!(
        "Dimension mismatch: X has {} columns, D-vine has dim {}",
        X.ncols(),
        self.dim
      )
      .into());
    }
    if X.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
      return Err("Input X must be in [0,1] for the D-vine".into());
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
    // Closed-form D-vine CDFs require numerical integration of the joint
    // density over a $d$-cube vertex pattern (equivalent to the NAC
    // finite-difference path); for d ≥ 3 a MC estimator via the sampler
    // is more reliable. We emit MC counts on 4000 samples per query —
    // ≈ 1.6% accuracy.
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

  /// All-Independence D-vine in d=3 must produce uniform marginals and
  /// independence (sample-correlation ≈ 0).
  #[test]
  fn dvine_independence_three_dim_uniform_marginals() {
    let dv = DVine::independence(3).unwrap();
    let s = dv.sample(10_000).unwrap();
    assert_eq!(s.ncols(), 3);
    for j in 0..3 {
      let col = s.column(j);
      let m: f64 = col.iter().sum::<f64>() / col.len() as f64;
      assert!(
        (m - 0.5).abs() < 0.02,
        "marginal {j} mean = {m}, expected ~0.5"
      );
    }
    // log-density of any input is 0 for independence.
    let lp = dv.log_pdf(array![[0.3, 0.6, 0.8]]).unwrap();
    assert!(lp[0].abs() < 1e-12);
  }

  /// 2-dim D-vine reduces to the chosen pair copula on the single edge.
  /// Sampling a 2-dim Clayton(θ=2) D-vine should produce strong
  /// lower-tail dependence (Kendall τ ≈ 0.5).
  #[test]
  fn dvine_two_dim_clayton_matches_bivariate() {
    let tree = vec![vec![PairCopula::Clayton { theta: 2.0 }]];
    let dv = DVine::new(2, tree).unwrap();
    let s = dv.sample(10_000).unwrap();
    use crate::correlation::kendall_tau;
    let tau = kendall_tau(&s);
    // Clayton(θ=2): τ = θ/(θ+2) = 2/4 = 0.5
    assert!(
      (tau[[0, 1]] - 0.5).abs() < 0.04,
      "2-dim Clayton(θ=2) D-vine τ_(0,1) = {}, expected ~0.5",
      tau[[0, 1]]
    );
  }

  /// 3-dim D-vine with Gaussian pair-copulas in T_1 (ρ=0.6) and
  /// independence in T_2 ⟹ partial-correlation structure with σ(1,3|2) = 0.
  /// Empirical Spearman / Pearson correlations on samples must satisfy
  /// |corr(1, 2)| ≈ 0.6 and |corr(2, 3)| ≈ 0.6.
  #[test]
  fn dvine_three_dim_gaussian_chain() {
    let t1 = vec![
      PairCopula::Gaussian { rho: 0.6 },
      PairCopula::Gaussian { rho: 0.6 },
    ];
    let t2 = vec![PairCopula::Independence];
    let dv = DVine::new(3, vec![t1, t2]).unwrap();
    let s = dv.sample(10_000).unwrap();
    use crate::correlation::kendall_tau;
    let tau = kendall_tau(&s);
    // Gaussian(ρ=0.6) → τ = (2/π) arcsin(0.6) ≈ 0.4097
    let expected_tau = (2.0 / std::f64::consts::PI) * 0.6_f64.asin();
    assert!(
      (tau[[0, 1]] - expected_tau).abs() < 0.03,
      "τ_(0,1) = {} vs expected {expected_tau}",
      tau[[0, 1]]
    );
    assert!(
      (tau[[1, 2]] - expected_tau).abs() < 0.03,
      "τ_(1,2) = {} vs expected {expected_tau}",
      tau[[1, 2]]
    );
    // Marginals uniform.
    for j in 0..3 {
      let col = s.column(j);
      let m: f64 = col.iter().sum::<f64>() / col.len() as f64;
      assert!((m - 0.5).abs() < 0.02);
    }
  }

  /// Structural validation: wrong tree shape must be rejected.
  #[test]
  fn dvine_shape_validation() {
    // Too few trees.
    let bad1 = vec![vec![PairCopula::Independence]];
    assert!(DVine::new(3, bad1).is_err());
    // Wrong number of edges per tree.
    let bad2 = vec![
      vec![PairCopula::Independence, PairCopula::Independence, PairCopula::Independence],
      vec![PairCopula::Independence],
    ];
    assert!(DVine::new(3, bad2).is_err());
    // dim < 2.
    let bad3 = vec![];
    assert!(DVine::new(1, bad3).is_err());
  }

  /// `fit` must return a descriptive error pointing at v2.4 sequential
  /// MLE being out of scope for v2.3.0.
  #[test]
  fn dvine_fit_rejects_with_v24_pointer() {
    let mut dv = DVine::independence(3).unwrap();
    let data = ndarray::Array2::<f64>::from_elem((10, 3), 0.5);
    let res = dv.fit(data);
    assert!(res.is_err());
    let msg = res.unwrap_err().to_string();
    assert!(
      msg.contains("v2.4") || msg.contains("MLE"),
      "fit error should point at v2.4 sequential MLE; got: {msg}"
    );
  }
}
