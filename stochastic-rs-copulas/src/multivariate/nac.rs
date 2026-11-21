//! Nested Archimedean copula (NAC): `φ₀` the root generator, `L₀` its direct leaves, `C_i` a nested
//! copula on `S_i` with `θ_i ≥ θ₀` (sufficient nesting); Clayton needs `θ > 0`, Gumbel `θ ≥ 1`.
//!
//! $$
//! C(u_1, \dots, u_d) = \varphi_{0}^{-1}\!\bigg(
//! \sum_{j \in L_0} \varphi_0(u_j) \;+\;
//! \sum_{i=1}^{m} \varphi_0\!\big(C_i(u_{S_i})\big)
//! \bigg),
//! $$
//!
//! Reference: Joe, H. (1997), Chapman & Hall, ch. 4, DOI 10.1201/b13150.
//! Reference: Hofert (2008), 52(12), 5163-5174, DOI 10.1016/j.csda.2008.05.019.
//! Reference: Devroye, L. (2009), ACM TOMACS 19(4), Art. 18, DOI 10.1145/1596519.1596523.
//! Reference: Marsaglia, Tsang (2000), ACM TOMS 26(3), note on `α < 1`, DOI 10.1145/358407.358414.
//! Reference: Hofert (2011), Comput. Stat. Data Anal. 55(1), 57-70, DOI 10.1016/j.csda.2010.04.025.
//! Reference: Hofert, Mächler (2011), J. Stat. Softw. 39(9), 1-20, DOI 10.18637/jss.v039.i09.
//! Reference: Hofert et al. (2012), 110, 133-150, DOI 10.1016/j.jmva.2012.02.019.
//! Reference: Hofert, Pham (2013), J. Multivar. Anal. 118, 37-52, DOI 10.1016/j.jmva.2013.03.006.
//! Reference: Gnedin, Pitman (2006), J. Math. Sci. 138, 5674-5685, DOI 10.1007/s10958-006-0335-z.
//! Reference: McNeil (2008), J. Stat. Comput. Simul. 78(6), 567-581, DOI 10.1080/00949650701255834.
//! Reference: Górecki et al. (2016), 46(1), 21-59, DOI 10.1007/s10844-014-0350-3.
//! Reference: Górecki et al. (2017), 87(17), 3261-3324, DOI 10.1080/00949655.2017.1365148.
//! Reference: Okhrin et al. (2013), J. Econom. 173(2), 189-204, DOI 10.1016/j.jeconom.2012.12.001.
//! Reference: Okhrin, Ristig (2014), J. Stat. Softw. 58(4), 1-20, DOI 10.18637/jss.v058.i04.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::RealExt;
use stochastic_rs_distributions::special::expm1_over;
use stochastic_rs_distributions::special::ln_1p_over;

use super::MultivariateFamily;
use crate::bivariate::BivariateFamily;
use crate::bivariate::conditional::check_unit;
use crate::error::CopulaError;
use crate::traits::MultivariateExt;

mod density;
mod fit;
mod sample;

/// The two generators a nested Archimedean copula admits; the public API names them by
/// `BivariateFamily`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Generator {
  Clayton,
  Gumbel,
}

impl Generator {
  fn from_family(family: BivariateFamily) -> Result<Self, CopulaError> {
    match family {
      BivariateFamily::Clayton => Ok(Self::Clayton),
      BivariateFamily::Gumbel => Ok(Self::Gumbel),
      other => Err(CopulaError::Unsupported(format!(
        "a nested Archimedean copula needs a Clayton or Gumbel generator, got {other:?}"
      ))),
    }
  }

  fn family(self) -> BivariateFamily {
    match self {
      Self::Clayton => BivariateFamily::Clayton,
      Self::Gumbel => BivariateFamily::Gumbel,
    }
  }

  /// `θ`'s domain: Gumbel `θ ≥ 1`; a Clayton `θ` needs its `Γ(1/θ, 1)` frailty, `0 < 1/θ < ∞`
  /// (Hofert & Mächler 2011, Table 1), which also rules out a `θ` whose reciprocal overflows.
  fn check_domain(self, theta: f64) -> Result<(), CopulaError> {
    let (inside, constraint) = match self {
      Self::Clayton => {
        let shape = 1.0 / theta;
        (shape > 0.0 && shape.is_finite(), "0 < 1 / theta < ∞")
      }
      Self::Gumbel => (theta >= 1.0, "theta >= 1.0 for Gumbel"),
    };
    if inside {
      return Ok(());
    }
    Err(CopulaError::InvalidParameter {
      name: "theta",
      value: theta,
      constraint: constraint.into(),
    })
  }

  /// `−ln C` of a node from its parts' `y = −ln C_k`: Clayton `ln(1 + Σ expm1(θy))/θ`, Gumbel
  /// `(Σ y^θ)^{1/θ}`, about the largest part: no overflow, and a small `θ` keeps its digits.
  fn minus_ln_cdf(self, theta: f64, parts: &[f64]) -> f64 {
    let top = parts.iter().copied().fold(0.0, f64::max_or_nan);
    if top == 0.0 || top.is_infinite() {
      return top;
    }
    match self {
      Self::Clayton if theta * top <= 36.0 => {
        let t_over_theta = parts
          .iter()
          .map(|&y| y * expm1_over(theta * y))
          .sum::<f64>();
        t_over_theta * ln_1p_over(theta * t_over_theta)
      }
      Self::Clayton => {
        let s = parts
          .iter()
          .map(|&y| (theta * (y - top)).exp() * -(-theta * y).exp_m1())
          .sum::<f64>();
        top + (s.ln() + ((-theta * top).exp() / s).ln_1p()) / theta
      }
      Self::Gumbel => {
        let s = parts
          .iter()
          .map(|&y| (theta * (y / top).ln()).exp())
          .sum::<f64>();
        top * (s.ln() / theta).exp()
      }
    }
  }
}

/// A node of the NAC tree: a generator parameter, its direct leaf indices and its nested sub-trees;
/// `NestedArchimedean::new` checks that every child's `theta` is at least its parent's.
#[derive(Debug, Clone)]
pub struct NacNode {
  /// Generator parameter $\theta$ of this node: $0 < 1/\theta < \infty$ for Clayton, $\theta \ge 1$
  /// for Gumbel.
  pub theta: f64,
  /// Marginal indices whose copula is governed directly by this node's
  /// generator (i.e. no further nesting below them).
  pub leaves: Vec<usize>,
  /// Nested sub-trees, each with $\theta_{\text{child}} \ge \theta_{\text{self}}$.
  pub children: Vec<NacNode>,
}

impl NacNode {
  /// Convenience constructor for a leaf-only node (no nested sub-trees).
  pub fn leaf_group(theta: f64, leaves: Vec<usize>) -> Self {
    Self {
      theta,
      leaves,
      children: vec![],
    }
  }

  /// Walk the tree collecting all leaf indices in left-to-right order.
  fn collect_leaves(&self, out: &mut Vec<usize>) {
    out.extend_from_slice(&self.leaves);
    for c in &self.children {
      c.collect_leaves(out);
    }
  }
}

/// Nested Archimedean copula with a single family across all nodes.
#[derive(Debug, Clone)]
pub struct NestedArchimedean {
  generator: Generator,
  root: Option<NacNode>,
  dim: usize,
  index_order: Vec<usize>,
}

impl NestedArchimedean {
  /// An empty copula for `fit`; `Unsupported` unless `family` is Clayton or Gumbel.
  pub fn unfitted(family: BivariateFamily) -> Result<Self, CopulaError> {
    Ok(Self {
      generator: Generator::from_family(family)?,
      root: None,
      dim: 0,
      index_order: vec![],
    })
  }

  /// A NAC from a Clayton or Gumbel `family` (`Unsupported` otherwise) and a root node, checking
  /// sufficient nesting, the generator's bounds and that every marginal index appears once.
  pub fn new(family: BivariateFamily, root: NacNode, dim: usize) -> Result<Self, CopulaError> {
    let generator = Generator::from_family(family)?;
    Self::validate_node(generator, &root, root.theta.min(f64::INFINITY), true)?;
    let mut index_order = Vec::with_capacity(dim);
    root.collect_leaves(&mut index_order);
    if index_order.len() != dim {
      return Err(CopulaError::InvalidStructure(format!(
        "the NAC tree has {} leaves but dim = {dim}",
        index_order.len()
      )));
    }
    let mut seen = vec![false; dim];
    for &j in &index_order {
      if j >= dim {
        return Err(CopulaError::InvalidStructure(format!(
          "leaf index {j} >= dim {dim}"
        )));
      }
      if seen[j] {
        return Err(CopulaError::InvalidStructure(format!(
          "leaf index {j} appears more than once"
        )));
      }
      seen[j] = true;
    }
    if seen.iter().any(|&b| !b) {
      return Err(CopulaError::InvalidStructure(
        "not every marginal appears in the NAC tree".into(),
      ));
    }
    Ok(Self {
      generator,
      root: Some(root),
      dim,
      index_order,
    })
  }

  /// Checks `node` and its subtree: every `θ` finite and in the generator's domain, and below the
  /// root at least `parent_theta` (the sufficient nesting condition).
  fn validate_node(
    generator: Generator,
    node: &NacNode,
    parent_theta: f64,
    is_root: bool,
  ) -> Result<(), CopulaError> {
    if !node.theta.is_finite() {
      return Err(CopulaError::InvalidParameter {
        name: "theta",
        value: node.theta,
        constraint: "theta.is_finite()".into(),
      });
    }
    generator.check_domain(node.theta)?;
    if !is_root && node.theta < parent_theta {
      return Err(CopulaError::InvalidStructure(format!(
        "sufficient nesting condition violated: child theta {:?} < parent theta {:?} ({:?})",
        node.theta,
        parent_theta,
        generator.family()
      )));
    }
    for child in &node.children {
      Self::validate_node(generator, child, node.theta, false)?;
    }
    Ok(())
  }

  /// The Archimedean family every node shares; the trait's `family()` is always
  /// `NestedArchimedean`.
  pub fn generator_family(&self) -> BivariateFamily {
    self.generator.family()
  }

  /// The tree; `None` for an `unfitted` copula until `fit` estimates one.
  pub fn root(&self) -> Option<&NacNode> {
    self.root.as_ref()
  }

  fn fitted_root(&self) -> Result<&NacNode, CopulaError> {
    self.root.as_ref().ok_or(CopulaError::NotFitted)
  }

  /// Total dimension.
  pub fn dim(&self) -> usize {
    self.dim
  }

  /// The leaf indices left to right, to map tree-positional output back to marginal indices.
  pub fn index_order(&self) -> &[usize] {
    &self.index_order
  }

  /// `−ln C` of the node's subtree at `u`.
  fn cdf_node(&self, node: &NacNode, u: &[f64]) -> f64 {
    let mut parts = node.leaves.iter().map(|&j| -u[j].ln()).collect::<Vec<_>>();
    for child in &node.children {
      parts.push(self.cdf_node(child, u));
    }
    self.generator.minus_ln_cdf(node.theta, &parts)
  }
}

impl MultivariateExt for NestedArchimedean {
  fn family(&self) -> MultivariateFamily {
    MultivariateFamily::NestedArchimedean
  }

  /// Hofert & Mächler (2011), Algorithm 3.1: a Clayton child costs O(V₀) ≈ O(1/θ_root) proposals.
  fn sample<S: SeedExt>(&self, n: usize, seed: &S) -> Result<Array2<f64>, CopulaError> {
    Ok(self.sample_with(self.fitted_root()?, n, &mut seed.rng()))
  }

  /// Binary tree from Kendall's τ-b of `x`; a mean τ̂ outside the generator's range (`0 < τ < 1`
  /// Clayton, `0 ≤ τ < 1` Gumbel) is `InvalidParameter { name: "tau", .. }`, storing nothing.
  fn fit(&mut self, x: ArrayView2<f64>) -> Result<(), CopulaError> {
    let root = fit::estimate(self.generator, x)?;
    *self = Self::new(self.generator.family(), root, x.ncols())?;
    Ok(())
  }

  fn check_fit(&self, x: ArrayView2<f64>) -> Result<(), CopulaError> {
    self.fitted_root()?;
    if x.ncols() != self.dim {
      return Err(CopulaError::DimensionMismatch {
        expected: self.dim,
        got: x.ncols(),
      });
    }
    check_unit(x)
  }

  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    Ok(self.log_pdf(x)?.mapv_into(f64::exp))
  }

  /// Hofert & Pham (2013)'s recursion in logs, re-centred per node on exact gaps, valid at every
  /// admitted `θ`; NaN on a 0 or 1 coordinate.
  fn log_pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit(x)?;
    Ok(density::Density::new(self.generator, self.fitted_root()?).log_pdf(x))
  }

  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit(x)?;
    let root = self.fitted_root()?;
    let mut out = Array1::<f64>::zeros(x.nrows());
    for (i, row) in x.rows().into_iter().enumerate() {
      let u: Vec<f64> = row.iter().copied().collect();
      out[i] = (-self.cdf_node(root, &u)).exp();
    }
    Ok(out)
  }
}

#[cfg(test)]
mod tests;
