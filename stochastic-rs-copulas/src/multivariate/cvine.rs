//! C-vine: tree $T_j$ joins its root $j$ to each $j+i$ given $1, \dots, j-1$; sampled by Aas et al. (2009) Algorithm 1 and evaluated
//! by its Algorithm 3 (density (9)) with the printed 1-based indices, taking an edge's ends in either order: exact, as every pair is exchangeable.
//! Reference: Aas, K., Czado, C., Frigessi, A. & Bakken, H. (2009), "Pair-copula constructions of multiple dependence", Insurance: Mathematics and Economics 44(2), 182–198, doi:10.1016/j.insmatheco.2007.02.001 (preprint: SFB 386 Discussion Paper 487).

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use rand::Rng;
use rand::RngExt;
use stochastic_rs_core::simd_rng::SeedExt;

use super::MultivariateFamily;
use super::dvine::PairCopula;
use super::dvine::pair_copula::check_pairs;
use super::fit::SelectionCriterion;
use super::fit::fit_cvine;
use super::rosenblatt;
use super::rosenblatt::Rosenblatt;
use super::vine_order::VineOrder;
use crate::bivariate::BivariateFamily;
use crate::error::CopulaError;
use crate::traits::MultivariateExt;

/// Canonical vine (C-vine) pair-copula construction over `dim` marginals.
///
/// `pair_copulas[m][i]` is the bivariate pair-copula on edge
/// $(m+1,\,m+i+2 \mid 1,\dots,m)$ of tree $T_{m+1}$. Outer length = $d-1$;
/// inner length of tree $m$ is $d-1-m$.
#[derive(Debug, Clone)]
pub struct CVine {
  dim: usize,
  order: VineOrder,
  pair_copulas: Vec<Vec<PairCopula>>,
}

impl CVine {
  /// Build a C-vine from a precomputed tree of pair copulas, under the shape and Student-t `nu` checks of
  /// [`super::dvine::DVine::new`].
  pub fn new(dim: usize, pair_copulas: Vec<Vec<PairCopula>>) -> Result<Self, CopulaError> {
    if dim < 2 {
      return Err(CopulaError::InvalidStructure(format!(
        "a C-vine needs dim >= 2, got {dim}"
      )));
    }
    if pair_copulas.len() != dim - 1 {
      return Err(CopulaError::InvalidStructure(format!(
        "expected {} trees for dim {dim}, got {}",
        dim - 1,
        pair_copulas.len()
      )));
    }
    for (m, tree) in pair_copulas.iter().enumerate() {
      let expected = dim - 1 - m;
      if tree.len() != expected {
        return Err(CopulaError::InvalidStructure(format!(
          "tree T_{} should have {expected} edges, got {}",
          m + 1,
          tree.len()
        )));
      }
    }
    check_pairs(&pair_copulas)?;
    Ok(Self {
      dim,
      order: VineOrder::identity(dim),
      pair_copulas,
    })
  }

  /// Which data column each vine variable reads; panics unless `order` is a permutation of `0..dim`.
  pub fn with_order(mut self, order: Vec<usize>) -> Self {
    self.order = VineOrder::new(order, self.dim);
    self
  }

  /// Data column of each vine variable.
  pub fn order(&self) -> &[usize] {
    self.order.as_slice()
  }

  /// All-Independence C-vine of the given dimension.
  pub fn independence(dim: usize) -> Result<Self, CopulaError> {
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

  fn sample_with<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Array2<f64> {
    let d = self.dim;
    let mut out = Array2::<f64>::zeros((n, d));
    let mut w = vec![0.0_f64; d + 1];
    let mut v = self.workspace();
    for mut x in out.rows_mut() {
      for wi in &mut w[1..] {
        *wi = rng.random::<f64>();
      }
      self.walk(&mut v, |i, _| Some(w[i]));
      for i in 1..=d {
        x[i - 1] = v[i][1].clamp(1e-12, 1.0 - 1e-12);
      }
    }
    out
  }

  /// Log-density of one observation by Aas et al. (2009) Algorithm 3, `v[j][i]` holding $v_{j,i}$ (row 0 is the data).
  fn log_density_one(&self, x: ArrayView1<f64>, v: &mut [Vec<f64>]) -> f64 {
    let n = self.dim;
    let theta = |j: usize, i: usize| self.pair_copulas[j - 1][i - 1];
    let mut log_c = 0.0_f64;
    for i in 1..=n {
      v[0][i] = x[i - 1];
    }
    for j in 1..n {
      for i in 1..=n - j {
        log_c += theta(j, i).log_density(v[j - 1][1], v[j - 1][i + 1]);
      }
      if j == n - 1 {
        break;
      }
      for i in 1..=n - j {
        v[j][i] = theta(j, i).h(v[j - 1][i + 1], v[j - 1][1]);
      }
    }
    log_c
  }

  fn log_density_rows(&self, x: ArrayView2<f64>) -> Array1<f64> {
    let mut v = vec![vec![0.0_f64; self.dim + 1]; self.dim];
    x.rows()
      .into_iter()
      .map(|row| self.log_density_one(row, &mut v))
      .collect()
  }
}

impl MultivariateExt for CVine {
  fn family(&self) -> MultivariateFamily {
    MultivariateFamily::CVine
  }

  fn sample<S: SeedExt>(&self, n: usize, seed: &S) -> Result<Array2<f64>, CopulaError> {
    Ok(self.order.to_data(self.sample_with(n, &mut seed.rng())))
  }

  /// Dißmann et al. (2013) over `BivariateFamily::VINE_CANDIDATES` by AIC; `fit_vine` exposes the candidates and the criterion.
  fn fit(&mut self, X: ArrayView2<f64>) -> Result<(), CopulaError> {
    let (vine, _) = fit_cvine(X, BivariateFamily::VINE_CANDIDATES, SelectionCriterion::Aic)?;
    *self = vine;
    Ok(())
  }

  fn check_fit(&self, X: ArrayView2<f64>) -> Result<(), CopulaError> {
    if X.ncols() != self.dim {
      return Err(CopulaError::DimensionMismatch {
        expected: self.dim,
        got: X.ncols(),
      });
    }
    if X.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
      return Err(CopulaError::MarginalOutOfRange);
    }
    Ok(())
  }

  fn pdf(&self, X: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit(X)?;
    let x = self.order.to_vine(X);
    Ok(self.log_density_rows(x.view()).mapv(f64::exp))
  }

  fn log_pdf(&self, X: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit(X)?;
    let x = self.order.to_vine(X);
    Ok(self.log_density_rows(x.view()))
  }

  /// Genz's separation of variables through the Rosenblatt map `sample` walks, averaged over 4096 Sobol points; a zero
  /// coordinate gives 0 and a unit one contributes `F(1 | ·) = 1` exactly.
  fn cdf(&self, X: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit(X)?;
    Ok(rosenblatt::cdf(self, self.order.to_vine(X).view()))
  }
}

impl Rosenblatt for CVine {
  fn workspace(&self) -> Vec<Vec<f64>> {
    vec![vec![0.0; self.dim + 1]; self.dim + 1]
  }

  /// Aas et al. (2009) Algorithm 1 with `target(i, v)` in place of `w_i`, leaving $x_i = v_{i,1}$ in `v[i][1]`.
  fn walk(&self, v: &mut [Vec<f64>], mut target: impl FnMut(usize, &[Vec<f64>]) -> Option<f64>) {
    let n = self.dim;
    let theta = |j: usize, i: usize| self.pair_copulas[j - 1][i - 1];
    for i in 1..=n {
      let Some(p) = target(i, v) else {
        return;
      };
      v[i][1] = p;
      for k in (1..i).rev() {
        v[i][1] = theta(k, i - k).h_inverse(v[i][1], v[k][k]);
      }
      if i == n {
        return;
      }
      for j in 1..i {
        v[i][j + 1] = theta(j, i - j).h(v[i][j], v[j][j]);
      }
    }
  }

  /// Algorithm 1's inversion loop run forward, `h` in place of `h⁻¹`, over $v_{k,k}$ for `k = 1, …, i − 1`.
  fn conditional_cdf(&self, i: usize, u: f64, v: &[Vec<f64>]) -> f64 {
    (1..i).fold(u, |t, k| self.pair_copulas[k - 1][i - k - 1].h(t, v[k][k]))
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::bivariate::clayton::Clayton;
  use crate::bivariate::frank::Frank;
  use crate::bivariate::gaussian::GaussianCopula;
  use crate::traits::BivariateExt;

  /// All-Independence C-vine on d=3: marginals uniform, log-density ≡ 0.
  #[test]
  fn cvine_independence_three_dim_uniform_marginals() {
    let cv = CVine::independence(3).unwrap();
    let s = cv.sample(10_000, &Deterministic::new(7)).unwrap();
    for j in 0..3 {
      let col = s.column(j);
      let m: f64 = col.iter().sum::<f64>() / col.len() as f64;
      assert!(
        (m - 0.5).abs() < 0.02,
        "marginal {j} mean = {m}, expected ~0.5"
      );
    }
    let lp = cv.log_pdf(array![[0.2, 0.4, 0.7]].view()).unwrap();
    assert!(lp[0].abs() < 1e-12);
  }

  /// 2-dim C-vine reduces to its single edge, matching the chosen
  /// PairCopula on the bivariate dependence.
  #[test]
  fn cvine_two_dim_clayton_matches_bivariate() {
    let cv = CVine::new(2, vec![vec![PairCopula::Clayton { theta: 2.0 }]]).unwrap();
    let s = cv.sample(10_000, &Deterministic::new(7)).unwrap();
    use crate::correlation::kendall_tau;
    let tau = kendall_tau(s.view()).unwrap();
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
    let s = cv.sample(10_000, &Deterministic::new(7)).unwrap();
    use crate::correlation::kendall_tau;
    let tau = kendall_tau(s.view()).unwrap();
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

  /// `fit` roots the first tree at the variable with the largest total |τ| and keeps the data's column order outside.
  #[test]
  fn cvine_fit_roots_the_strongest_variable() {
    let corr = ndarray::array![[1.0, 0.8, 0.6], [0.8, 1.0, 0.5], [0.6, 0.5, 1.0]];
    let truth = crate::multivariate::gaussian::GaussianMultivariate::try_new(corr).unwrap();
    let u = truth.sample(3000, &Deterministic::new(5)).unwrap();
    let mut fitted = CVine::independence(3).unwrap();
    fitted.fit(u.view()).unwrap();
    assert_eq!(fitted.order()[0], 0, "{:?}", fitted.order());
    assert!(
      fitted.pair_copulas()[0]
        .iter()
        .all(|p| matches!(p, PairCopula::Gaussian { rho } if *rho > 0.4))
    );
    let density = fitted.pdf(u.view()).unwrap();
    assert!(density.iter().all(|d| d.is_finite() && *d > 0.0));
  }

  /// Aas et al. (2009) Eq. (9) at n = 3, c₁₂ · c₁₃ · c₂₃|₁, from the bivariate copulas' own densities and h-functions.
  #[test]
  fn cvine_density_is_the_root_centred_product() {
    let tree = vec![
      vec![
        PairCopula::Gaussian { rho: 0.4 },
        PairCopula::Clayton { theta: 2.0 },
      ],
      vec![PairCopula::Frank { theta: 3.0 }],
    ];
    let (c12, c13, c23_1) = (GaussianCopula::new(0.4), Clayton::new(2.0), Frank::new(3.0));
    let points = array![
      [0.3, 0.6, 0.8],
      [0.1, 0.5, 0.9],
      [0.7, 0.2, 0.4],
      [0.95, 0.9, 0.05]
    ];
    let got = CVine::new(3, tree).unwrap().pdf(points.view()).unwrap();
    let at = |a: f64, b: f64| array![[a, b]];
    for (row, p) in points.rows().into_iter().enumerate() {
      let h2_1 = c12.partial_derivative(at(p[1], p[0]).view()).unwrap()[0];
      let h3_1 = c13.partial_derivative(at(p[2], p[0]).view()).unwrap()[0];
      let want = c12.pdf(at(p[0], p[1]).view()).unwrap()[0]
        * c13.pdf(at(p[0], p[2]).view()).unwrap()[0]
        * c23_1.pdf(at(h2_1, h3_1).view()).unwrap()[0];
      assert!(
        (got[row] / want - 1.0).abs() < 1e-10,
        "row {row}: {} vs {want}",
        got[row]
      );
    }
  }

  /// At a draw the walk places, the forward chain gives back every uniform: Algorithm 1's inversion run backwards.
  #[test]
  fn the_conditional_cdf_inverts_the_walk() {
    let trees = vec![
      vec![
        PairCopula::Gaussian { rho: 0.5 },
        PairCopula::Clayton { theta: 2.0 },
        PairCopula::Frank { theta: -3.0 },
        PairCopula::StudentT { rho: -0.2, nu: 4.0 },
      ],
      vec![
        PairCopula::Clayton { theta: 3.0 },
        PairCopula::Gaussian { rho: 0.6 },
        PairCopula::Frank { theta: 5.0 },
      ],
      vec![
        PairCopula::Frank { theta: 4.0 },
        PairCopula::Bb1 {
          theta: 0.8,
          delta: 1.6,
        },
      ],
      vec![PairCopula::Gaussian { rho: -0.4 }],
    ];
    let vine = CVine::new(5, trees).unwrap();
    let mut v = vine.workspace();
    for w in [
      [0.0, 0.3, 0.7, 0.2, 0.9, 0.55],
      [0.0, 0.05, 0.5, 0.95, 0.4, 0.15],
    ] {
      vine.walk(&mut v, |i, _| Some(w[i]));
      for i in 1..=5 {
        let back = vine.conditional_cdf(i, v[i][1], &v);
        assert!((back - w[i]).abs() < 1e-13, "x_{i}: {back} vs {}", w[i]);
      }
    }
  }
}
