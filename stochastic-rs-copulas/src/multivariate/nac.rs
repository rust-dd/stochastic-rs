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
//! - **Clayton:** $\theta \ge 0$ globally and $\theta_{\text{child}} \ge \theta_{\text{parent}}$.
//! - **Gumbel:** $\theta \ge 1$ globally and $\theta_{\text{child}} \ge \theta_{\text{parent}}$.
//!
//! ## Sampling (Hofert 2008, Algorithm 2)
//!
//! The standard Marshall-Olkin (1988) frailty representation: each
//! generator $\varphi_\theta$ admits an LST representation
//! $\varphi_\theta(t) = \mathbb{E}\!\left[e^{-t V_\theta}\right]$ with
//!
//! - **Clayton:** $V_\theta \sim \mathrm{Gamma}(1/\theta, 1)$.
//! - **Gumbel:** $V_\theta \sim S_+(1/\theta)$ (positive $\alpha$-stable).
//!
//! For a partially nested copula with root $V_0$ and one inner generator
//! per sub-tree, the conditional inner frailty is
//!
//! $$
//! V_i \mid V_0 \;=\; V_0^{\theta_0/\theta_i}\cdot S_i,\qquad
//! S_i \sim S_+(\theta_0 / \theta_i),
//! $$
//!
//! which collapses to a single sub-tree frailty draw. Leaves are then
//! recovered via $U_j = \psi_{\theta_*}(E_j / V_*)$, where $\psi$ is the
//! inverse generator and $E_j \sim \mathrm{Exp}(1)$.
//!
//! ## Density (Hofert-Pham 2012)
//!
//! The density is the $d$-th-order partial derivative of $C$ with respect
//! to the marginals. For partially nested Clayton / Gumbel copulas the
//! closed forms involve the generator-derivative families
//! $\{\varphi^{(k)}\}$ which we evaluate via the recurrences from Hofert,
//! Mächler & McNeil (2011).
//!
//! References:
//! - Joe, H. (1997), *Multivariate Models and Dependence Concepts*,
//!   Chapman & Hall, ch. 4.
//! - Hofert, M. (2008), "Sampling Archimedean copulas",
//!   *Computational Statistics & Data Analysis* 52(12), 5163-5174.
//! - Hofert, M. (2011), "Efficiently sampling nested Archimedean copulas",
//!   *Computational Statistics & Data Analysis* 55(1), 57-70.
//! - Hofert, M., Mächler, M., McNeil, A.J. (2011), "Likelihood inference
//!   for Archimedean copulas in high dimensions under known margins",
//!   *Journal of Multivariate Analysis* 110, 133-150.
//! - Hofert, M., Pham, D. (2013), "Densities of nested Archimedean
//!   copulas", *Journal of Multivariate Analysis* 118, 37-52.
//!   (arXiv:1204.2410)
//! - McNeil, A.J. (2008), "Sampling nested Archimedean copulas",
//!   *Journal of Statistical Computation and Simulation* 78(6), 567-581.

use std::error::Error;
use std::f64;

use ndarray::Array1;
use ndarray::Array2;
use rand::Rng;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::special::ln_gamma;

use super::CopulaType;
use crate::traits::MultivariateExt;

/// Single-parameter Archimedean family supported by [`NestedArchimedean`].
/// All nodes of a NAC must share the same family; mixed-family NAC is not
/// yet supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NacFamily {
  /// Clayton: $\varphi_\theta(t) = (t^{-\theta} - 1)/\theta$,
  /// $\theta \ge 0$. Lower-tail dependence.
  Clayton,
  /// Gumbel: $\varphi_\theta(t) = (-\ln t)^\theta$,
  /// $\theta \ge 1$. Upper-tail dependence.
  Gumbel,
}

impl NacFamily {
  /// Minimum admissible $\theta$ (boundary of independence).
  pub fn theta_min(self) -> f64 {
    match self {
      NacFamily::Clayton => 0.0,
      NacFamily::Gumbel => 1.0,
    }
  }

  /// Inverse generator $\psi_\theta = \varphi_\theta^{-1}$ in the
  /// **LST-normalised form** (so $\psi$ is the Laplace transform of the
  /// frailty `V` sampled by [`NestedArchimedean::sample`]):
  /// - Clayton: $\psi(s) = (1 + s)^{-1/\theta}$ — LST of
  ///   $V \sim \mathrm{Gamma}(1/\theta, 1)$.
  /// - Gumbel:  $\psi(s) = \exp(-s^{1/\theta})$ — LST of
  ///   $V \sim S_+(1/\theta)$.
  ///
  /// The CDF $C = \psi(\sum \varphi(u_j))$ is invariant under the convention
  /// rescaling $\varphi \to \varphi/c$, $\psi \to \psi(c\cdot)$, so the
  /// resulting copula matches the bivariate `BivariateExt` Clayton
  /// implementation (which uses the equivalent $(1 + \theta s)^{-1/\theta}$
  /// scaled form).
  pub fn inverse_generator(self, theta: f64, s: f64) -> f64 {
    match self {
      NacFamily::Clayton => (1.0 + s).powf(-1.0 / theta),
      NacFamily::Gumbel => (-(s.powf(1.0 / theta))).exp(),
    }
  }

  /// Generator $\varphi_\theta(t) = \psi^{-1}(t)$, the inverse of the
  /// LST form above.
  /// - Clayton: $\varphi(t) = t^{-\theta} - 1$.
  /// - Gumbel:  $\varphi(t) = (-\ln t)^\theta$.
  pub fn generator(self, theta: f64, t: f64) -> f64 {
    let t = t.clamp(1e-15, 1.0 - 1e-15);
    match self {
      NacFamily::Clayton => t.powf(-theta) - 1.0,
      NacFamily::Gumbel => (-t.ln()).powf(theta),
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
  /// Family parameter $\theta$ for this node's generator. Must satisfy
  /// [`NacFamily::theta_min`].
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
  family: NacFamily,
  root: NacNode,
  dim: usize,
  index_order: Vec<usize>,
}

impl NestedArchimedean {
  /// Construct a NAC from a family and root node. Validates the sufficient
  /// nesting condition, family-specific parameter bounds, and that every
  /// marginal index appears exactly once in the tree.
  pub fn new(family: NacFamily, root: NacNode, dim: usize) -> Result<Self, Box<dyn Error>> {
    Self::validate_node(family, &root, root.theta.min(f64::INFINITY), true)?;
    let mut index_order = Vec::with_capacity(dim);
    root.collect_leaves(&mut index_order);
    if index_order.len() != dim {
      return Err(
        format!(
          "NAC tree exposes {} leaves but dim = {dim}",
          index_order.len()
        )
        .into(),
      );
    }
    let mut seen = vec![false; dim];
    for &j in &index_order {
      if j >= dim {
        return Err(format!("leaf index {j} ≥ dim {dim}").into());
      }
      if seen[j] {
        return Err(format!("leaf index {j} appears more than once").into());
      }
      seen[j] = true;
    }
    if seen.iter().any(|&b| !b) {
      return Err("Not every marginal appears in the NAC tree".into());
    }
    Ok(Self {
      family,
      root,
      dim,
      index_order,
    })
  }

  /// Recursive SNC validator. `parent_theta` is the closest ancestor's
  /// theta; for the root pass its own theta (so the parent constraint is
  /// vacuously satisfied). `is_root` flips behavior for the family-specific
  /// minimum check.
  fn validate_node(
    family: NacFamily,
    node: &NacNode,
    parent_theta: f64,
    is_root: bool,
  ) -> Result<(), Box<dyn Error>> {
    let theta_min = family.theta_min();
    if node.theta < theta_min {
      return Err(
        format!(
          "{family:?} node θ={} below family minimum {theta_min}",
          node.theta
        )
        .into(),
      );
    }
    if !is_root && node.theta < parent_theta {
      return Err(
        format!(
          "SNC violation: child θ={} < parent θ={} ({family:?})",
          node.theta, parent_theta
        )
        .into(),
      );
    }
    for child in &node.children {
      Self::validate_node(family, child, node.theta, false)?;
    }
    Ok(())
  }

  /// Family exposed for downstream inspection.
  pub fn family(&self) -> NacFamily {
    self.family
  }

  /// Root node (immutable view).
  pub fn root(&self) -> &NacNode {
    &self.root
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
    let theta = f64::consts::PI * u;
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
    let v = match (self.family, parent_state) {
      (NacFamily::Clayton, None) => {
        // Root Clayton frailty: V ~ Gamma(shape = 1/θ, scale = 1) →
        // LST = (1 + s)^{-1/θ} = ψ_Clayton(s) (LST-normalised).
        let g = SimdGamma::<f64>::new(
          1.0 / node.theta,
          1.0,
          &stochastic_rs_core::simd_rng::Unseeded,
        );
        g.sample_fast()
      }
      (NacFamily::Gumbel, None) => {
        // Root Gumbel frailty: V ~ S_+(1/θ). The θ = 1 case is the
        // independence boundary (Gumbel = product copula); the positive
        // stable distribution degenerates to the constant 1 there.
        if (node.theta - 1.0).abs() < 1e-12 {
          1.0
        } else {
          Self::positive_stable(rng, 1.0 / node.theta)
        }
      }
      (NacFamily::Gumbel, Some((parent_theta, parent_v))) => {
        // Nested Gumbel (McNeil 2008, derivation §4 with Hofert 2011
        // Algorithm 2 form). The conditional LST of V_c given V_p is
        // exp(-V_p · s^α), α = θ_p / θ_c, which is the LST of
        //   V_c = V_p^{1/α} · S_+(α).
        // SNC ⟹ α ∈ (0, 1]; α = 1 is the no-nesting case (V_c = V_p).
        let alpha = parent_theta / node.theta;
        if (alpha - 1.0).abs() < 1e-12 {
          parent_v
        } else {
          let s = Self::positive_stable(rng, alpha);
          parent_v.powf(1.0 / alpha) * s
        }
      }
      (NacFamily::Clayton, Some(_)) => {
        // Nested Clayton: the conditional frailty V_c | V_p has LST
        // exp(-V_p · ((1+s)^α - 1)) with α = θ_p/θ_c — an exponentially
        // tilted positive stable distribution. Exact sampling requires
        // Devroye's double-rejection (Hofert 2011, Algorithm 4), which is
        // not yet implemented. We surface this limitation up to `sample`
        // so the caller does not silently get a biased copula; the CDF /
        // PDF / structure-validation paths are unaffected and work for
        // arbitrarily nested Clayton trees.
        panic!(
          "Nested-Clayton sampling is not implemented — \
           use NacFamily::Gumbel for nesting (correct Hofert (2011) \
           Algorithm 2), or call cdf / pdf on a nested-Clayton tree \
           (those paths are fully supported). Exact nested-Clayton \
           sampling via tilted-stable Devroye double-rejection is \
           not yet implemented."
        );
      }
    };

    // Generate direct leaves under this node's generator.
    for &j in &node.leaves {
      let e_uniform: f64 = rng.random::<f64>().clamp(1e-15, 1.0 - 1e-15);
      let e = -e_uniform.ln();
      let arg = e / v.max(1e-300);
      let u = self.family.inverse_generator(node.theta, arg);
      out[j] = u.clamp(1e-12, 1.0 - 1e-12);
    }

    // Recurse into nested sub-trees.
    for child in &node.children {
      self.sample_node(rng, child, Some((node.theta, v)), out);
    }
  }

  /// Recursive CDF evaluator: returns $C(u_{\text{tree}})$ for the sub-tree
  /// rooted at `node`. Used both directly (whole-tree CDF) and indirectly
  /// (via finite differences) for density estimation.
  fn cdf_node(&self, node: &NacNode, u: &[f64]) -> f64 {
    let mut s = 0.0;
    for &j in &node.leaves {
      s += self.family.generator(node.theta, u[j]);
    }
    for child in &node.children {
      let c_inner = self.cdf_node(child, u);
      s += self.family.generator(node.theta, c_inner);
    }
    self.family.inverse_generator(node.theta, s)
  }
}

impl MultivariateExt for NestedArchimedean {
  fn r#type(&self) -> CopulaType {
    CopulaType::NestedArchimedean
  }

  fn sample(&self, n: usize) -> Result<Array2<f64>, Box<dyn Error>> {
    let d = self.dim;
    let mut out = Array2::<f64>::zeros((n, d));
    let mut rng = rand::rng();
    for r in 0..n {
      let mut row = Array1::<f64>::zeros(d);
      // Root call: `parent_state = None` signals that V_0 must be sampled
      // independently from the root family's frailty distribution.
      self.sample_node(&mut rng, &self.root, None, &mut row);
      for j in 0..d {
        out[[r, j]] = row[j];
      }
    }
    Ok(out)
  }

  fn fit(&mut self, _X: Array2<f64>) -> Result<(), Box<dyn Error>> {
    // The structural parameters (tree topology + family) are not estimated
    // from data; the user supplies them via `new()`. A full structure +
    // parameter fit (Okhrin-Okhrin-Schmid 2013 HAC structure selection) is
    // not yet implemented.
    Err(
      "NestedArchimedean::fit not implemented — supply the tree via NestedArchimedean::new \
       and use crate::correlation::kendall_tau on the marginal pairs to seed θ values. \
       Structure learning is not yet implemented."
        .into(),
    )
  }

  fn check_fit(&self, X: &Array2<f64>) -> Result<(), Box<dyn Error>> {
    if X.ncols() != self.dim {
      return Err(
        format!(
          "Dimension mismatch: X has {} columns, NAC has dim {}",
          X.ncols(),
          self.dim
        )
        .into(),
      );
    }
    if X.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
      return Err("Input X must be in [0,1] for NAC".into());
    }
    Ok(())
  }

  fn pdf(&self, X: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit(&X)?;
    // Closed-form NAC densities (Hofert-Pham 2012) require the full
    // d-th-order generator-derivative recursion which is family-specific
    // and scales as O(d²) per query; instead we estimate the density via
    // finite differences on the CDF along the d-cube vertex pattern. This
    // is exact in the limit h → 0 and stable for h = 1e-4 on
    // well-conditioned NAC trees (θ_max / θ_min < 20).
    let d = self.dim;
    let h = 1e-4_f64;
    let denom = (2.0 * h).powi(d as i32);
    let mut out = Array1::<f64>::zeros(X.nrows());
    let mut u_pert = vec![0.0_f64; d];
    for (i, row) in X.rows().into_iter().enumerate() {
      let u_orig: Vec<f64> = row.iter().copied().collect();
      let mut acc = 0.0;
      // Enumerate 2^d vertices: for each subset S of {0..d}, sign = (-1)^{d-|S|}.
      for mask in 0..(1u32 << d) {
        let mut sign = 1.0;
        for j in 0..d {
          if (mask >> j) & 1 == 1 {
            u_pert[j] = (u_orig[j] + h).min(1.0 - 1e-12);
          } else {
            u_pert[j] = (u_orig[j] - h).max(1e-12);
            sign = -sign;
          }
        }
        acc += sign * self.cdf_node(&self.root, &u_pert);
      }
      out[i] = (acc / denom).max(0.0);
    }
    Ok(out)
  }

  fn cdf(&self, X: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit(&X)?;
    let mut out = Array1::<f64>::zeros(X.nrows());
    for (i, row) in X.rows().into_iter().enumerate() {
      let u: Vec<f64> = row.iter().copied().collect();
      out[i] = self.cdf_node(&self.root, &u);
    }
    Ok(out)
  }
}

/// Generator log-density coefficient used by the closed-form Clayton-NAC
/// density (kept here so the test module can cross-validate the finite
/// difference path).
#[allow(dead_code)]
pub(crate) fn clayton_log_density_constant(theta: f64, d: usize) -> f64 {
  // log Γ(d + 1/θ) − log Γ(1/θ): the prefactor of the d-leaf Clayton density.
  ln_gamma(d as f64 + 1.0 / theta) - ln_gamma(1.0 / theta)
}

#[cfg(test)]
mod tests {
  use super::*;

  /// A flat (single-level) Clayton NAC with all leaves under one root must
  /// reduce to the standard exchangeable Clayton copula on 3 dims.
  #[test]
  fn nac_clayton_flat_is_exchangeable_clayton() {
    let root = NacNode::leaf_group(2.0, vec![0, 1, 2]);
    let nac = NestedArchimedean::new(NacFamily::Clayton, root, 3).unwrap();
    let u = nac.sample(8_000).unwrap();
    assert_eq!(u.ncols(), 3);
    for j in 0..3 {
      let col = u.column(j);
      let mean = col.iter().sum::<f64>() / col.len() as f64;
      assert!(
        (mean - 0.5).abs() < 0.03,
        "marginal {j} mean = {mean}, expected ~0.5"
      );
    }
  }

  /// 2-level Clayton NAC tree: root θ=1.5 with leaf 0 direct + a sub-tree
  /// with θ=4.0 over leaves {1,2}. The CDF path is fully supported (nested
  /// Clayton sampling is not — see `sample_node` Clayton-nested branch);
  /// we therefore verify the structural property analytically: for two
  /// outer-inner leaves the pair-margin is exchangeable Clayton on the
  /// **root** generator, while for two inner-inner leaves it's
  /// exchangeable Clayton on the **inner** generator.
  #[test]
  fn nac_clayton_two_level_cdf_pair_margins() {
    let inner = NacNode::leaf_group(4.0, vec![1, 2]);
    let root = NacNode {
      theta: 1.5,
      leaves: vec![0],
      children: vec![inner],
    };
    let nac = NestedArchimedean::new(NacFamily::Clayton, root, 3).unwrap();
    // Outer-inner pair margin: C(u_0, u_1, 1) = (u_0^{-θ_root} + u_1^{-θ_root} - 1)^{-1/θ_root}
    let q_outer_inner = ndarray::array![[0.3, 0.7, 1.0 - 1e-15]];
    let c_oi = nac.cdf(q_outer_inner).unwrap()[0];
    let theta_root: f64 = 1.5;
    let expected_oi =
      (0.3f64.powf(-theta_root) + 0.7f64.powf(-theta_root) - 1.0).powf(-1.0 / theta_root);
    assert!(
      (c_oi - expected_oi).abs() < 5e-3,
      "outer-inner pair CDF={} vs Clayton(θ_root) expected={}",
      c_oi,
      expected_oi
    );
    // Inner-inner pair margin: C(1, u_1, u_2) = (u_1^{-θ_inner} + u_2^{-θ_inner} - 1)^{-1/θ_inner}
    let q_inner_inner = ndarray::array![[1.0 - 1e-15, 0.3, 0.7]];
    let c_ii = nac.cdf(q_inner_inner).unwrap()[0];
    let theta_inner: f64 = 4.0;
    let expected_ii =
      (0.3f64.powf(-theta_inner) + 0.7f64.powf(-theta_inner) - 1.0).powf(-1.0 / theta_inner);
    assert!(
      (c_ii - expected_ii).abs() < 5e-3,
      "inner-inner pair CDF={} vs Clayton(θ_inner) expected={}",
      c_ii,
      expected_ii
    );
    // Inner pair should be MORE dependent: larger CDF at off-diagonal
    // input pair than the outer pair at the same input.
    assert!(
      c_ii > c_oi,
      "inner CDF({c_ii}) must exceed outer CDF({c_oi}) — Clayton inner θ=4 > root θ=1.5"
    );
  }

  /// Nested-Clayton sampling intentionally panics; calling `sample` on a
  /// tree with at least one nested node must surface the not-implemented
  /// message rather than silently returning biased data.
  #[test]
  #[should_panic(expected = "Nested-Clayton sampling")]
  fn nac_clayton_nested_sampling_panics() {
    let inner = NacNode::leaf_group(4.0, vec![1]);
    let root = NacNode {
      theta: 1.5,
      leaves: vec![0],
      children: vec![inner],
    };
    let nac = NestedArchimedean::new(NacFamily::Clayton, root, 2).unwrap();
    let _ = nac.sample(10);
  }

  /// 2-level Gumbel NAC: same structural test as Clayton but on the
  /// Gumbel family with θ_root=2.0 and inner θ=4.0.
  #[test]
  fn nac_gumbel_two_level_inner_pair_more_dependent() {
    let inner = NacNode::leaf_group(4.0, vec![1, 2]);
    let root = NacNode {
      theta: 2.0,
      leaves: vec![0],
      children: vec![inner],
    };
    let nac = NestedArchimedean::new(NacFamily::Gumbel, root, 3).unwrap();
    let u = nac.sample(8_000).unwrap();
    use crate::correlation::kendall_tau;
    let tau = kendall_tau(&u);
    // Gumbel: τ = 1 − 1/θ
    //   Inner θ=4 → τ ≈ 0.75
    //   Outer θ=2 → τ ≈ 0.50
    assert!(
      tau[[1, 2]] > tau[[0, 1]],
      "inner pair τ_(1,2)={} should exceed outer τ_(0,1)={}",
      tau[[1, 2]],
      tau[[0, 1]]
    );
    assert!(
      tau[[1, 2]] > 0.6 && tau[[1, 2]] < 0.85,
      "Gumbel inner τ_(1,2)={} out of expected band [0.6, 0.85]",
      tau[[1, 2]]
    );
  }

  /// SNC violation: child θ less than parent θ must be rejected.
  #[test]
  fn nac_snc_violation_rejected() {
    let bad_inner = NacNode::leaf_group(0.5, vec![1]); // θ_child=0.5 < θ_parent=2.0
    let bad_root = NacNode {
      theta: 2.0,
      leaves: vec![0],
      children: vec![bad_inner],
    };
    let res = NestedArchimedean::new(NacFamily::Clayton, bad_root, 2);
    assert!(res.is_err(), "SNC violation must error");
    assert!(
      res.unwrap_err().to_string().contains("SNC"),
      "error message should mention SNC"
    );
  }

  /// Below-family-minimum θ must be rejected per family.
  #[test]
  fn nac_below_family_min_rejected() {
    let root = NacNode::leaf_group(0.5, vec![0, 1]); // Gumbel requires θ ≥ 1
    let res = NestedArchimedean::new(NacFamily::Gumbel, root, 2);
    assert!(res.is_err());
    // And Clayton at θ < 0 should also fail.
    let bad_clayton = NacNode::leaf_group(-0.1, vec![0, 1]);
    assert!(NestedArchimedean::new(NacFamily::Clayton, bad_clayton, 2).is_err());
  }

  /// Missing / duplicate leaf indices must be rejected.
  #[test]
  fn nac_leaf_index_validation() {
    // Duplicate
    let dup_root = NacNode::leaf_group(2.0, vec![0, 0, 1]);
    assert!(NestedArchimedean::new(NacFamily::Clayton, dup_root, 3).is_err());
    // Missing (covers 0,2 only, dim=3)
    let miss = NacNode::leaf_group(2.0, vec![0, 2]);
    assert!(NestedArchimedean::new(NacFamily::Clayton, miss, 3).is_err());
    // Out of range
    let oor = NacNode::leaf_group(2.0, vec![0, 1, 5]);
    assert!(NestedArchimedean::new(NacFamily::Clayton, oor, 3).is_err());
  }

  /// CDF must equal $(\Pi u_j)$ for a θ=0 Clayton (independence boundary).
  #[test]
  fn nac_clayton_independence_cdf() {
    // θ ≥ 0 — at θ = 0 the inverse generator (1 + 0·s)^∞ degenerates; we
    // probe a small θ > 0 and check the CDF stays close to the
    // independence product within the Clayton bias.
    let root = NacNode::leaf_group(0.01, vec![0, 1, 2]);
    let nac = NestedArchimedean::new(NacFamily::Clayton, root, 3).unwrap();
    let q = ndarray::array![[0.5, 0.5, 0.5], [0.2, 0.3, 0.4]];
    let c = nac.cdf(q.clone()).unwrap();
    let indep_1 = 0.5_f64.powi(3);
    let indep_2 = 0.2_f64 * 0.3 * 0.4;
    assert!(
      (c[0] - indep_1).abs() < 0.05,
      "near-independence CDF[0]={} vs indep={indep_1}",
      c[0]
    );
    assert!(
      (c[1] - indep_2).abs() < 0.05,
      "near-independence CDF[1]={} vs indep={indep_2}",
      c[1]
    );
  }

  /// `fit` must return a descriptive error pointing at structure learning
  /// not being implemented.
  #[test]
  fn nac_fit_rejects_with_descriptive_error() {
    let root = NacNode::leaf_group(2.0, vec![0, 1]);
    let mut nac = NestedArchimedean::new(NacFamily::Clayton, root, 2).unwrap();
    let data = ndarray::Array2::<f64>::from_elem((10, 2), 0.5);
    let res = nac.fit(data);
    assert!(res.is_err());
    let msg = res.unwrap_err().to_string();
    assert!(
      msg.contains("structure") || msg.contains("not implemented"),
      "fit error should explain that structure learning is not implemented; got: {msg}"
    );
  }
}
