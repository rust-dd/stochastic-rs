//! D-vine: tree $T_j$ joins $i$ and $i+j$ given $i+1, \dots, i+j-1$; sampled by Aas et al. (2009) Algorithm 2 and evaluated by its
//! Algorithm 4 (density (8)) with the printed 1-based indices, taking an edge's ends in either order: exact, as every pair is exchangeable.
//! Reference: Aas, K., Czado, C., Frigessi, A. & Bakken, H. (2009), "Pair-copula constructions of multiple dependence", Insurance: Mathematics and Economics 44(2), 182–198, doi:10.1016/j.insmatheco.2007.02.001 (preprint: SFB 386 Discussion Paper 487).

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use rand::Rng;
use rand::RngExt;
use stochastic_rs_core::simd_rng::SeedExt;

use super::MultivariateFamily;
use super::fit::SelectionCriterion;
use super::fit::fit_dvine;
use super::rosenblatt;
use super::rosenblatt::Rosenblatt;
use super::vine_order::VineOrder;
use crate::bivariate::BivariateFamily;
use crate::bivariate::conditional::check_unit;
use crate::error::CopulaError;
use crate::multivariate::inside;
use crate::multivariate::on_open_cube;
use crate::traits::MultivariateExt;

pub mod pair_copula;

pub use pair_copula::PairCopula;

/// D-vine over `dim` marginals: `pair_copulas[m][i]` is the [`PairCopula`] on edge `i` of tree `m`,
/// for `m = 0..d-1` and `i = 0..d-1-m`.
#[derive(Debug, Clone)]
pub struct DVine {
  dim: usize,
  order: VineOrder,
  /// `pair_copulas[m][i]` is the edge $(i,\,i+m+1\mid i+1,\dots,i+m)$ of
  /// tree $T_{m+1}$. Outer length = $d-1$; tree $m$ has length $d-1-m$.
  pair_copulas: Vec<Vec<PairCopula>>,
}

impl DVine {
  /// A D-vine on `dim` margins from its `dim − 1` trees, tree `m` holding `dim − 1 − m` pairs: `InvalidStructure` for any
  /// other shape, and a pair's bivariate `InvalidParameter` for parameters outside its family's domain.
  pub fn new(dim: usize, pair_copulas: Vec<Vec<PairCopula>>) -> Result<Self, CopulaError> {
    if dim < 2 {
      return Err(CopulaError::InvalidStructure(format!(
        "a D-vine needs dim >= 2, got {dim}"
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
    pair_copula::check_parameters(&pair_copulas)?;
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

  /// An all-independence D-vine of `dim` marginals, the product copula.
  pub fn independence(dim: usize) -> Result<Self, CopulaError> {
    // `new` names a `dim` below 2; the trees of such a `dim` are empty.
    let pair_copulas: Vec<Vec<PairCopula>> = (0..dim.saturating_sub(1))
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
        x[i - 1] = inside(v[i][1]);
      }
    }
    out
  }

  /// Log-density of one observation by Aas et al. (2009) Algorithm 4, `v[j][i]` holding $v_{j,i}$ (row 0 is the data).
  fn log_density_one(&self, x: ArrayView1<f64>, v: &mut [Vec<f64>]) -> f64 {
    let n = self.dim;
    let theta = |j: usize, i: usize| self.pair_copulas[j - 1][i - 1];
    let mut log_c = 0.0_f64;
    for i in 1..=n {
      v[0][i] = x[i - 1];
    }
    for i in 1..n {
      log_c += theta(1, i).log_density(v[0][i], v[0][i + 1]);
    }
    if n == 2 {
      return log_c;
    }
    v[1][1] = theta(1, 1).h(v[0][1], v[0][2]);
    for k in 1..=n - 3 {
      v[1][2 * k] = theta(1, k + 1).h(v[0][k + 2], v[0][k + 1]);
      v[1][2 * k + 1] = theta(1, k + 1).h(v[0][k + 1], v[0][k + 2]);
    }
    v[1][2 * n - 4] = theta(1, n - 1).h(v[0][n], v[0][n - 1]);
    for j in 2..n {
      for i in 1..=n - j {
        log_c += theta(j, i).log_density(v[j - 1][2 * i - 1], v[j - 1][2 * i]);
      }
      if j == n - 1 {
        break;
      }
      v[j][1] = theta(j, 1).h(v[j - 1][1], v[j - 1][2]);
      for i in 1..=n - j - 2 {
        v[j][2 * i] = theta(j, i + 1).h(v[j - 1][2 * i + 2], v[j - 1][2 * i + 1]);
        v[j][2 * i + 1] = theta(j, i + 1).h(v[j - 1][2 * i + 1], v[j - 1][2 * i + 2]);
      }
      v[j][2 * n - 2 * j - 2] =
        theta(j, n - j).h(v[j - 1][2 * n - 2 * j], v[j - 1][2 * n - 2 * j - 1]);
    }
    log_c
  }

  fn log_density_rows(&self, x: ArrayView2<f64>) -> Array1<f64> {
    let mut v = vec![vec![0.0_f64; 2 * self.dim]; self.dim];
    x.rows()
      .into_iter()
      .map(|row| self.log_density_one(row, &mut v))
      .collect()
  }
}

impl MultivariateExt for DVine {
  fn family(&self) -> MultivariateFamily {
    MultivariateFamily::DVine
  }

  fn sample<S: SeedExt>(&self, n: usize, seed: &S) -> Result<Array2<f64>, CopulaError> {
    Ok(self.order.to_data(self.sample_with(n, &mut seed.rng())))
  }

  /// Dißmann et al. (2013) over `BivariateFamily::VINE_CANDIDATES` by AIC; `fit_vine` exposes the candidates and the criterion.
  fn fit(&mut self, x: ArrayView2<f64>) -> Result<(), CopulaError> {
    let (vine, _) = fit_dvine(x, BivariateFamily::VINE_CANDIDATES, SelectionCriterion::Aic)?;
    *self = vine;
    Ok(())
  }

  fn check_fit(&self, x: ArrayView2<f64>) -> Result<(), CopulaError> {
    if x.ncols() != self.dim {
      return Err(CopulaError::DimensionMismatch {
        expected: self.dim,
        got: x.ncols(),
      });
    }
    check_unit(x)
  }

  fn pdf(&self, uv: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    Ok(self.log_pdf(uv)?.mapv_into(f64::exp))
  }

  fn log_pdf(&self, uv: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit(uv)?;
    let x = self.order.to_vine(uv);
    Ok(on_open_cube(uv, self.log_density_rows(x.view())))
  }

  /// Genz's separation of variables through the Rosenblatt map `sample` walks, averaged over 4096 Sobol points; a zero
  /// coordinate gives 0 and a unit one contributes `F(1 | ·) = 1` exactly.
  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit(x)?;
    Ok(rosenblatt::cdf(self, self.order.to_vine(x).view()))
  }
}

impl Rosenblatt for DVine {
  fn workspace(&self) -> Vec<Vec<f64>> {
    vec![vec![0.0; 2 * self.dim]; self.dim + 1]
  }

  /// Aas et al. (2009) Algorithm 2 with `target(i, v)` in place of `w_i`, leaving $x_i = v_{i,1}$ in `v[i][1]`.
  fn walk(&self, v: &mut [Vec<f64>], mut target: impl FnMut(usize, &[Vec<f64>]) -> Option<f64>) {
    let n = self.dim;
    let theta = |j: usize, i: usize| self.pair_copulas[j - 1][i - 1];
    let Some(p) = target(1, v) else {
      return;
    };
    v[1][1] = p;
    let Some(p) = target(2, v) else {
      return;
    };
    v[2][1] = theta(1, 1).h_inverse(p, v[1][1]);
    if n == 2 {
      return;
    }
    v[2][2] = theta(1, 1).h(v[1][1], v[2][1]);
    for i in 3..=n {
      let Some(p) = target(i, v) else {
        return;
      };
      v[i][1] = p;
      for k in (2..i).rev() {
        v[i][1] = theta(k, i - k).h_inverse(v[i][1], v[i - 1][2 * k - 2]);
      }
      v[i][1] = theta(1, i - 1).h_inverse(v[i][1], v[i - 1][1]);
      if i == n {
        return;
      }
      v[i][2] = theta(1, i - 1).h(v[i - 1][1], v[i][1]);
      v[i][3] = theta(1, i - 1).h(v[i][1], v[i - 1][1]);
      for j in 2..=i - 2 {
        v[i][2 * j] = theta(j, i - j).h(v[i - 1][2 * j - 2], v[i][2 * j - 1]);
        v[i][2 * j + 1] = theta(j, i - j).h(v[i][2 * j - 1], v[i - 1][2 * j - 2]);
      }
      v[i][2 * i - 2] = theta(i - 1, 1).h(v[i - 1][2 * i - 4], v[i][2 * i - 3]);
    }
  }

  /// Algorithm 2's inversion loop run forward, `h` in place of `h⁻¹`: tree 1 given $v_{i-1,1}$, tree `k` given
  /// $v_{i-1,2k-2}$.
  fn conditional_cdf(&self, i: usize, u: f64, v: &[Vec<f64>]) -> f64 {
    if i == 1 {
      return u;
    }
    let tree_1 = self.pair_copulas[0][i - 2].h(u, v[i - 1][1]);
    (2..i).fold(tree_1, |t, k| {
      self.pair_copulas[k - 1][i - k - 1].h(t, v[i - 1][2 * k - 2])
    })
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

  /// All-Independence D-vine in d=3 must produce uniform marginals and
  /// independence (sample-correlation ≈ 0).
  #[test]
  fn dvine_independence_three_dim_uniform_marginals() {
    let dv = DVine::independence(3).unwrap();
    let s = dv.sample(10_000, &Deterministic::new(7)).unwrap();
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
    let lp = dv.log_pdf(array![[0.3, 0.6, 0.8]].view()).unwrap();
    assert!(lp[0].abs() < 1e-12);
  }

  /// A 2-dim D-vine is its single pair copula: Clayton(θ=2) samples have Kendall τ ≈ 0.5.
  #[test]
  fn dvine_two_dim_clayton_matches_bivariate() {
    let tree = vec![vec![PairCopula::Clayton { theta: 2.0 }]];
    let dv = DVine::new(2, tree).unwrap();
    let s = dv.sample(10_000, &Deterministic::new(7)).unwrap();
    use crate::correlation::kendall_tau;
    let tau = kendall_tau(s.view()).unwrap();
    // Clayton(θ=2): τ = θ/(θ+2) = 2/4 = 0.5
    assert!(
      (tau[[0, 1]] - 0.5).abs() < 0.04,
      "2-dim Clayton(θ=2) D-vine τ_(0,1) = {}, expected ~0.5",
      tau[[0, 1]]
    );
  }

  /// Gaussian pairs (ρ=0.6) over independence in T_2 leave (1,3|2) uncorrelated: Kendall τ of (1,2)
  /// and (2,3) is (2/π) arcsin 0.6.
  #[test]
  fn dvine_three_dim_gaussian_chain() {
    let t1 = vec![
      PairCopula::Gaussian { rho: 0.6 },
      PairCopula::Gaussian { rho: 0.6 },
    ];
    let t2 = vec![PairCopula::Independence];
    let dv = DVine::new(3, vec![t1, t2]).unwrap();
    let s = dv.sample(10_000, &Deterministic::new(7)).unwrap();
    use crate::correlation::kendall_tau;
    let tau = kendall_tau(s.view()).unwrap();
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
      vec![
        PairCopula::Independence,
        PairCopula::Independence,
        PairCopula::Independence,
      ],
      vec![PairCopula::Independence],
    ];
    assert!(DVine::new(3, bad2).is_err());
    // dim < 2.
    let bad3 = vec![];
    assert!(DVine::new(1, bad3).is_err());
  }

  /// `fit` learns the Dißmann order and families; the fitted vine evaluates and samples in the data's column order.
  #[test]
  fn dvine_fit_learns_structure_and_families_in_data_order() {
    use crate::correlation::kendall_tau;
    let trees = vec![
      vec![
        PairCopula::Gaussian { rho: 0.7 },
        PairCopula::Clayton { theta: 3.0 },
        PairCopula::Frank { theta: 6.0 },
      ],
      vec![PairCopula::Independence; 2],
      vec![PairCopula::Independence],
    ];
    let truth = DVine::new(4, trees).unwrap();
    // columns shuffled: data column j holds vine variable perm[j]
    let perm = [2_usize, 0, 3, 1];
    let u = truth
      .sample(4000, &Deterministic::new(11))
      .unwrap()
      .select(ndarray::Axis(1), &perm);
    let mut fitted = DVine::independence(4).unwrap();
    fitted.fit(u.view()).unwrap();
    let position = |j: usize| fitted.order().iter().position(|&c| c == j).unwrap();
    // data columns (1, 3), (3, 0), (0, 2) are the strong edges: they must be neighbours in the fitted order
    for (a, b) in [(1, 3), (3, 0), (0, 2)] {
      assert_eq!(
        position(a).abs_diff(position(b)),
        1,
        "order {:?}",
        fitted.order()
      );
    }
    let ll: f64 = fitted.pdf(u.view()).unwrap().mapv(f64::ln).sum();
    let truth_ll: f64 = truth
      .pdf(u.select(ndarray::Axis(1), &[1, 3, 0, 2]).view())
      .unwrap()
      .mapv(f64::ln)
      .sum();
    assert!(
      ll > truth_ll - 0.02 * truth_ll.abs(),
      "fit {ll} truth {truth_ll}"
    );
    let resampled = fitted.sample(4000, &Deterministic::new(5)).unwrap();
    let (tau_data, tau_fit) = (
      kendall_tau(u.view()).unwrap(),
      kendall_tau(resampled.view()).unwrap(),
    );
    for i in 0..4 {
      for j in 0..4 {
        assert!(
          (tau_data[[i, j]] - tau_fit[[i, j]]).abs() < 0.08,
          "τ({i},{j}) {} vs {}",
          tau_data[[i, j]],
          tau_fit[[i, j]]
        );
      }
    }
  }

  #[test]
  #[should_panic(expected = "order must satisfy `order is a permutation of 0..dim`")]
  fn with_order_rejects_a_non_permutation() {
    let _ = DVine::independence(3).unwrap().with_order(vec![0, 0, 2]);
  }

  /// Aas et al. (2009) Eq. (10), c₁₂ · c₂₃ · c₁₃|₂, from the bivariate copulas' own densities and h-functions.
  #[test]
  fn dvine_density_is_the_three_pair_product() {
    let tree = vec![
      vec![
        PairCopula::Gaussian { rho: 0.4 },
        PairCopula::Clayton { theta: 2.0 },
      ],
      vec![PairCopula::Frank { theta: 3.0 }],
    ];
    let (c12, c23, c13_2) = (GaussianCopula::new(0.4), Clayton::new(2.0), Frank::new(3.0));
    let points = array![
      [0.3, 0.6, 0.8],
      [0.1, 0.5, 0.9],
      [0.7, 0.2, 0.4],
      [0.95, 0.9, 0.05]
    ];
    let got = DVine::new(3, tree).unwrap().pdf(points.view()).unwrap();
    let at = |a: f64, b: f64| array![[a, b]];
    for (row, p) in points.rows().into_iter().enumerate() {
      let h1_2 = c12.partial_derivative(at(p[0], p[1]).view()).unwrap()[0];
      let h3_2 = c23.partial_derivative(at(p[2], p[1]).view()).unwrap()[0];
      let want = c12.pdf(at(p[0], p[1]).view()).unwrap()[0]
        * c23.pdf(at(p[1], p[2]).view()).unwrap()[0]
        * c13_2.pdf(at(h1_2, h3_2).view()).unwrap()[0];
      assert!(
        (got[row] / want - 1.0).abs() < 1e-10,
        "row {row}: {} vs {want}",
        got[row]
      );
    }
  }

  /// At a draw the walk places, the forward chain gives back every uniform: Algorithm 2's inversion run backwards.
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
    let vine = DVine::new(5, trees).unwrap();
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
