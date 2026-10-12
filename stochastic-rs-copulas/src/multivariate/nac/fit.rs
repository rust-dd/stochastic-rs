use ndarray::ArrayView2;

use super::Generator;
use super::NacNode;
use crate::bivariate::clayton::Clayton;
use crate::bivariate::gumbel::Gumbel;
use crate::correlation::check_defined;
use crate::correlation::kendall_tau;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::bivariate::tau_out_of_range;

impl Generator {
  /// The family's Kendall inversion on the nested range: a Clayton node needs `θ > 0` (a `Γ(1/θ)` frailty), so `τ > 0`.
  pub(super) fn theta_from_tau(self, tau: f64) -> Result<f64, CopulaError> {
    match self {
      Self::Clayton if !(tau > 0.0 && tau < 1.0) => {
        Err(tau_out_of_range(tau, "0.0 < tau < 1.0".into()))
      }
      Self::Clayton => Clayton::default().theta_from_tau(tau),
      Self::Gumbel => Gumbel::default().theta_from_tau(tau),
    }
  }
}

/// A cluster of the agglomeration: a margin, or a subtree already joined.
enum Item {
  Leaf(usize),
  Node(NacNode),
}

impl Item {
  fn theta(&self) -> Option<f64> {
    match self {
      Item::Leaf(_) => None,
      Item::Node(node) => Some(node.theta),
    }
  }

  fn merge(theta: f64, a: Item, b: Item) -> NacNode {
    let mut node = NacNode {
      theta,
      leaves: vec![],
      children: vec![],
    };
    for item in [a, b] {
      match item {
        Item::Leaf(j) => node.leaves.push(j),
        Item::Node(child) => node.children.push(child),
      }
    }
    node
  }
}

/// The checks run in the order every multivariate fit uses: columns, rows, range, then a defined τ-b for every column.
pub(super) fn estimate(generator: Generator, X: ArrayView2<f64>) -> Result<NacNode, CopulaError> {
  let (n, d) = (X.nrows(), X.ncols());
  if d < 2 {
    return Err(CopulaError::InvalidStructure(format!(
      "a nested Archimedean copula needs dim >= 2, got {d}"
    )));
  }
  if n < 2 {
    return Err(CopulaError::InsufficientData { needed: 2, got: n });
  }
  if X.iter().any(|v| !(0.0..=1.0).contains(v)) {
    return Err(CopulaError::MarginalOutOfRange);
  }
  let tau = kendall_tau(X)?;
  check_defined(tau.view())?;
  agglomerate(generator, tau.view())
}

/// Górecki, Hofert & Holeňa (2016), Algorithm 3 with the average over `d ≥ 2` columns: join the clusters `I`, `J` of
/// largest mean τ̂ over `I × J` (the first in label order on a tie), θ = τ⁻¹ of that mean trimmed to the children's θ.
pub(super) fn agglomerate(
  generator: Generator,
  tau: ArrayView2<f64>,
) -> Result<NacNode, CopulaError> {
  let mut clusters = (0..tau.nrows())
    .map(|j| (Item::Leaf(j), vec![j]))
    .collect::<Vec<_>>();
  loop {
    let (mut best, mut best_tau) = ((0, 1), f64::NEG_INFINITY);
    for a in 0..clusters.len() {
      for b in (a + 1)..clusters.len() {
        let mean = mean_cross_tau(tau, &clusters[a].1, &clusters[b].1);
        if mean > best_tau {
          best = (a, b);
          best_tau = mean;
        }
      }
    }
    let estimate = generator.theta_from_tau(best_tau)?;
    let (item_b, members_b) = clusters.remove(best.1);
    let (item_a, mut members) = clusters.remove(best.0);
    // Average linkage keeps a parent's mean at or below its children's (Górecki et al. 2017, Corollary 6.2), so the trim
    // only absorbs the rounding of the means.
    let theta = [item_a.theta(), item_b.theta()]
      .into_iter()
      .flatten()
      .fold(estimate, f64::min);
    let node = Item::merge(theta, item_a, item_b);
    if clusters.is_empty() {
      return Ok(node);
    }
    members.extend(members_b);
    clusters.push((Item::Node(node), members));
  }
}

fn mean_cross_tau(tau: ArrayView2<f64>, a: &[usize], b: &[usize]) -> f64 {
  let sum = a
    .iter()
    .flat_map(|&i| b.iter().map(move |&j| tau[[i, j]]))
    .sum::<f64>();
  sum / (a.len() * b.len()) as f64
}
