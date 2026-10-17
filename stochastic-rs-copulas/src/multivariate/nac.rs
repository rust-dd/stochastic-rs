//! # Nested Archimedean Copula (NAC)
//!
//! $$
//! C(u_1, \dots, u_d) = \varphi_{0}^{-1}\!\bigg(
//! \sum_{j \in L_0} \varphi_0(u_j) \;+\;
//! \sum_{i=1}^{m} \varphi_0\!\big(C_i(u_{S_i})\big)
//! \bigg),
//! $$
//!
//! where $\varphi_0$ is the outer (root) generator with parameter $\theta_0$,
//! $L_0$ is the set of root-direct leaves, and each $C_i$ is itself a
//! single-level Archimedean copula on the index set $S_i$ with parameter
//! $\theta_i$. The **sufficient nesting condition** (Joe 1997, McNeil 2008)
//! requires $\theta_i \ge \theta_0$ per family:
//!
//! - **Clayton:** $0 < 1/\theta < \infty$ globally (the frailty $\Gamma(1/\theta, 1)$ exists) and $\theta_{\text{child}} \ge \theta_{\text{parent}}$.
//! - **Gumbel:** $\theta \ge 1$ globally and $\theta_{\text{child}} \ge \theta_{\text{parent}}$.
//!
//! ## Sampling
//!
//! Hofert & Mächler (2011), Algorithm 3.1: root frailty `Γ(1/θ, 1)` (Clayton) or `S_+(1/θ)` (Gumbel); a child's frailty given
//! its parent's `V₀` has LST `exp(−V₀((1 + t)^α − 1))` or `exp(−V₀ t^α)`, `α = θ₀/θ₁` (their Table 3); a leaf is `ψ(E/V)`.
//!
//! ## Density
//!
//! Closed form on every tree `new` accepts: Hofert & Pham's (2013) recursion at any depth (their Remark 6.2) over the
//! generator derivatives of Hofert, Mächler & McNeil (2012), summed in logs of positive terms; NaN on a 0 or 1 coordinate.
//!
//! ## Fit
//!
//! `fit` joins bottom-up, as Okhrin, Okhrin & Schmid (2013) do, the two clusters of largest mean cross Kendall τ̂, and sets
//! θ = τ⁻¹ of that mean, trimmed to the children's θ (Górecki, Hofert & Holeňa 2016, Algorithm 3).
//!
//! References:
//! - Joe, H. (1997), *Multivariate Models and Dependence Concepts*,
//!   Chapman & Hall, ch. 4.
//! - Hofert, M. (2008), "Sampling Archimedean copulas",
//!   *Computational Statistics & Data Analysis* 52(12), 5163-5174.
//! - Hofert, M. (2011), "Efficiently sampling nested Archimedean copulas", *Computational Statistics & Data Analysis* 55(1), 57-70, doi:10.1016/j.csda.2010.04.025.
//! - Hofert, M., Mächler, M. (2011), "Nested Archimedean Copulas Meet R: The nacopula Package", *Journal of Statistical Software* 39(9), 1-20, doi:10.18637/jss.v039.i09.
//! - Hofert, M., Mächler, M., McNeil, A.J. (2012), "Likelihood inference for Archimedean copulas in high dimensions under known margins", *Journal of Multivariate Analysis* 110, 133-150, doi:10.1016/j.jmva.2012.02.019.
//! - Hofert, M., Pham, D. (2013), "Densities of nested Archimedean copulas", *Journal of Multivariate Analysis* 118, 37-52, doi:10.1016/j.jmva.2013.03.006 (arXiv:1204.2410).
//! - Gnedin, A., Pitman, J. (2006), "Exchangeable Gibbs partitions and Stirling triangles", *Journal of Mathematical Sciences* 138(3), 5674-5685, doi:10.1007/s10958-006-0335-z (arXiv:math/0412494).
//! - McNeil, A.J. (2008), "Sampling nested Archimedean copulas",
//!   *Journal of Statistical Computation and Simulation* 78(6), 567-581.
//! - Górecki, J., Hofert, M., Holeňa, M. (2016), "An approach to structure determination and estimation of hierarchical Archimedean copulas and its application to Bayesian classification", *Journal of Intelligent Information Systems* 46(1), 21-59, doi:10.1007/s10844-014-0350-3.
//! - Górecki, J., Hofert, M., Holeňa, M. (2017), "On structure, family and parameter estimation of hierarchical Archimedean copulas", *Journal of Statistical Computation and Simulation* 87(17), 3261-3324, doi:10.1080/00949655.2017.1365148.
//! - Okhrin, O., Okhrin, Y., Schmid, W. (2013), "On the structure and estimation of hierarchical Archimedean copulas", *Journal of Econometrics* 173(2), 189-204, doi:10.1016/j.jeconom.2012.12.001.
//! - Okhrin, O., Ristig, A. (2014), "Hierarchical Archimedean Copulae: The HAC Package", *Journal of Statistical Software* 58(4), 1-20, doi:10.18637/jss.v058.i04.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;
use rand::Rng;
use rand::RngExt;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::gamma::SimdGamma;

use super::MultivariateFamily;
use crate::bivariate::BivariateFamily;
use crate::error::CopulaError;
use crate::traits::MultivariateExt;

mod density;
mod fit;

/// The two generators a nested Archimedean copula admits; the public API names them by `BivariateFamily`.
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

  /// `θ`'s domain: Gumbel `θ ≥ 1`; a Clayton `θ` needs its `Γ(1/θ, 1)` frailty, `0 < 1/θ < ∞` (Hofert & Mächler 2011,
  /// Table 1), which also rules out a `θ` whose reciprocal overflows.
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

  /// LST-normalised inverse generator: Clayton $(1 + s)^{-1/\theta}$, Gumbel $\exp(-s^{1/\theta})$.
  fn inverse_generator(self, theta: f64, s: f64) -> f64 {
    match self {
      Self::Clayton => (1.0 + s).powf(-1.0 / theta),
      Self::Gumbel => (-(s.powf(1.0 / theta))).exp(),
    }
  }

  /// Generator $\varphi_\theta = \psi_\theta^{-1}$: Clayton $t^{-\theta} - 1$, Gumbel $(-\ln t)^\theta$.
  fn generator(self, theta: f64, t: f64) -> f64 {
    let t = t.clamp(1e-15, 1.0 - 1e-15);
    match self {
      Self::Clayton => t.powf(-theta) - 1.0,
      Self::Gumbel => (-t.ln()).powf(theta),
    }
  }
}

/// Recursive NAC tree node. Each non-leaf node owns a generator parameter
/// and a (possibly empty) list of direct leaf indices plus a list of nested
/// sub-trees.
///
/// **Invariant (sufficient nesting condition):** every child node's
/// `theta` must be **at least** as large as its parent's. Constructors
/// validate this; mutating individual fields directly bypasses the check —
/// prefer the builder API.
#[derive(Debug, Clone)]
pub struct NacNode {
  /// Generator parameter $\theta$ of this node: $0 < 1/\theta < \infty$ for Clayton, $\theta \ge 1$ for Gumbel.
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

  /// Builds a NAC from a Clayton or Gumbel `family` (`Unsupported` otherwise) and a root node; checks the sufficient
  /// nesting condition, the generator's parameter bounds and that every marginal index appears exactly once.
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

  /// Checks `node` and its subtree: every `θ` finite and in the generator's domain, and below the root at least
  /// `parent_theta` (the sufficient nesting condition).
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

  /// The Archimedean family every node shares; the trait's `family()` is always `NestedArchimedean`.
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

  /// Left-to-right traversal of the leaf indices, as registered at
  /// construction time. Useful for downstream consumers that need to map
  /// tree-positional output to original marginal indices.
  pub fn index_order(&self) -> &[usize] {
    &self.index_order
  }

  /// Sample a single positive $\alpha$-stable variate via the **Kanter
  /// (1975) representation** (also Devroye 1986, §IX.4.3) on $\Theta \in
  /// (0, \pi)$ — every term in the product is non-negative for $0 < \alpha
  /// < 1$, so the output is strictly positive (in contrast to the
  /// symmetric Chambers-Mallows formula on $(-\pi/2, \pi/2)$).
  ///
  /// $$
  /// X = \frac{\sin(\alpha\Theta)\cdot \sin((1-\alpha)\Theta)^{(1-\alpha)/\alpha}}
  ///          {\sin(\Theta)^{1/\alpha}\cdot W^{(1-\alpha)/\alpha}},
  /// \quad \Theta \sim U(0,\pi),\ W \sim \mathrm{Exp}(1).
  /// $$
  fn positive_stable<R: Rng + ?Sized>(rng: &mut R, alpha: f64) -> f64 {
    debug_assert!(alpha > 0.0 && alpha < 1.0);
    let u: f64 = rng.random::<f64>().clamp(1e-15, 1.0 - 1e-15);
    let theta = std::f64::consts::PI * u;
    let w_uniform: f64 = rng.random::<f64>().clamp(1e-15, 1.0 - 1e-15);
    let w = -w_uniform.ln();
    let s_a = (alpha * theta).sin();
    let s_oa = ((1.0 - alpha) * theta).sin();
    let s_t = theta.sin();
    let exponent = (1.0 - alpha) / alpha;
    let numerator = s_a * s_oa.powf(exponent);
    let denominator = s_t.powf(1.0 / alpha) * w.powf(exponent);
    numerator / denominator
  }

  /// LST `exp(−v₀((1 + t)^α − 1))` by Hofert's (2011) fast rejection: a sum of `m` positive stables of LST `exp(−(v₀/m) t^α)`,
  /// each kept with probability `e^{−S}`, `m` minimising the expected `m e^{v₀/m} ≤ e max(1, ⌈v₀⌉)` proposals.
  fn tilted_stable<R: Rng + ?Sized>(rng: &mut R, alpha: f64, v0: f64) -> f64 {
    let m = if v0 <= 1.0 {
      1.0
    } else {
      let (lo, hi) = (v0.floor(), v0.ceil());
      if lo * (v0 / lo).exp() <= hi * (v0 / hi).exp() {
        lo
      } else {
        hi
      }
    };
    let scale = (v0 / m).powf(1.0 / alpha);
    (0..m as u64)
      .map(|_| {
        loop {
          let s = scale * Self::positive_stable(rng, alpha);
          // Strict on the [0, 1) uniform, so an infinite proposal is never kept.
          if rng.random::<f64>() < (-s).exp() {
            break s;
          }
        }
      })
      .sum()
  }

  fn sample_with<R: Rng + ?Sized>(&self, root: &NacNode, n: usize, rng: &mut R) -> Array2<f64> {
    let d = self.dim;
    let mut out = Array2::<f64>::zeros((n, d));
    for r in 0..n {
      let mut row = Array1::<f64>::zeros(d);
      // Root call: `parent_state = None` signals that V_0 must be sampled
      // independently from the root family's frailty distribution.
      self.sample_node(rng, root, None, &mut row);
      for j in 0..d {
        out[[r, j]] = row[j];
      }
    }
    out
  }

  /// Recursive sample helper. `parent_state` carries (parent_theta,
  /// parent_frailty) when the node is a non-root descendant; `None`
  /// signals the root call, in which case the frailty is sampled
  /// independently. Writes leaves directly into `out` at the indices
  /// stored in the node.
  fn sample_node<R: Rng + ?Sized>(
    &self,
    rng: &mut R,
    node: &NacNode,
    parent_state: Option<(f64, f64)>,
    out: &mut Array1<f64>,
  ) {
    let v = match (self.generator, parent_state) {
      (Generator::Clayton, None) => {
        // V ~ Gamma(1/θ, 1) has LST (1 + s)^{-1/θ}; its sub-seed comes off `rng`, since an `Unseeded` source
        // here would break the replay of every Clayton tree (the root frailty is drawn on every row).
        let sub_seed = rng.random::<u64>();
        let mut g =
          SimdGamma::<f64>::new(1.0 / node.theta, 1.0).seeded(&Deterministic::new(sub_seed));
        g.sample()
      }
      (Generator::Gumbel, None) => {
        // Root Gumbel frailty: V ~ S_+(1/θ). The θ = 1 case is the
        // independence boundary (Gumbel = product copula); the positive
        // stable distribution degenerates to the constant 1 there.
        if (node.theta - 1.0).abs() < 1e-12 {
          1.0
        } else {
          Self::positive_stable(rng, 1.0 / node.theta)
        }
      }
      (Generator::Gumbel, Some((parent_theta, parent_v))) => {
        // V_c | V_p has LST exp(-V_p s^α), α = θ_p / θ_c, so V_c = V_p^{1/α} · S_+(α) (Hofert & Mächler 2011, Table 3);
        // α = 1 is the no-nesting case, V_c = V_p.
        let alpha = parent_theta / node.theta;
        if (alpha - 1.0).abs() < 1e-12 {
          parent_v
        } else {
          let s = Self::positive_stable(rng, alpha);
          parent_v.powf(1.0 / alpha) * s
        }
      }
      (Generator::Clayton, Some((parent_theta, parent_v))) => {
        // V_c | V_p is the tilted stable of LST exp(-V_p((1 + s)^α - 1)), α = θ_p / θ_c (Hofert & Mächler 2011, Table 3
        // and eq. (5)); α = 1 is the no-nesting case, V_c = V_p.
        let alpha = parent_theta / node.theta;
        if (alpha - 1.0).abs() < 1e-12 {
          parent_v
        } else {
          Self::tilted_stable(rng, alpha, parent_v)
        }
      }
    };

    // Generate direct leaves under this node's generator.
    for &j in &node.leaves {
      let e_uniform: f64 = rng.random::<f64>().clamp(1e-15, 1.0 - 1e-15);
      let e = -e_uniform.ln();
      let arg = e / v.max(1e-300);
      let u = self.generator.inverse_generator(node.theta, arg);
      out[j] = u.clamp(1e-12, 1.0 - 1e-12);
    }

    // Recurse into nested sub-trees.
    for child in &node.children {
      self.sample_node(rng, child, Some((node.theta, v)), out);
    }
  }

  fn cdf_node(&self, node: &NacNode, u: &[f64]) -> f64 {
    let mut s = 0.0;
    for &j in &node.leaves {
      s += self.generator.generator(node.theta, u[j]);
    }
    for child in &node.children {
      let c_inner = self.cdf_node(child, u);
      s += self.generator.generator(node.theta, c_inner);
    }
    self.generator.inverse_generator(node.theta, s)
  }
}

impl MultivariateExt for NestedArchimedean {
  fn family(&self) -> MultivariateFamily {
    MultivariateFamily::NestedArchimedean
  }

  fn sample<S: SeedExt>(&self, n: usize, seed: &S) -> Result<Array2<f64>, CopulaError> {
    Ok(self.sample_with(self.fitted_root()?, n, &mut seed.rng()))
  }

  /// Estimates the binary tree from Kendall's τ-b of `x` (Górecki, Hofert & Holeňa 2016, Algorithm 3); a mean τ̂ outside the
  /// generator's range (`0 < τ < 1` Clayton, `0 ≤ τ < 1` Gumbel) is `InvalidParameter { name: "tau", .. }` and stores nothing.
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
    if x.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
      return Err(CopulaError::MarginalOutOfRange);
    }
    Ok(())
  }

  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    Ok(self.log_pdf(x)?.mapv_into(f64::exp))
  }

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
      out[i] = self.cdf_node(root, &u);
    }
    Ok(out)
  }
}

#[cfg(test)]
mod tests;
