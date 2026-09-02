//! # Vine fitting
//!
//! Sequential structure and family selection for D-vines and C-vines
//! after Dißmann, Brechmann, Czado & Kurowicka (2013): the first tree is
//! chosen on the empirical Kendall's τ (a greedy maximal path for the
//! D-vine order, the strongest-dependence root for the C-vine), each edge's
//! pair copula is picked among the candidate families by AIC or BIC with
//! its parameters estimated on the current pseudo-observations, and the
//! h-functions of the fitted pairs produce the pseudo-observations of the
//! next tree (Aas, Czado, Frigessi & Bakken 2009, Algorithms 3–4).
//!
//! Parameter estimation per family: Gaussian and Student-t correlation by
//! Kendall inversion `ρ = sin(πτ/2)` (the degrees of freedom by profile
//! likelihood on a grid), Clayton and Frank by Kendall inversion, BB1 and
//! BB7 by maximum likelihood.
//!
//! References: Dißmann, J., Brechmann, E. C., Czado, C. & Kurowicka, D.
//! (2013), *Selecting and estimating regular vine copulae and application
//! to financial returns*, Computational Statistics & Data Analysis 59,
//! 52–69; Aas, K., Czado, C., Frigessi, A. & Bakken, H. (2009), *Pair-copula
//! constructions of multiple dependence*, Insurance: Mathematics and
//! Economics 44(2), 182–198.

use std::error::Error;

use ndarray::Array1;
use ndarray::Array2;

use super::cvine::CVine;
use super::dvine::DVine;
use super::dvine::PairCopula;
use super::rvine::RVine;
use crate::bivariate::bb1::Bb1;
use crate::bivariate::bb7::Bb7;
use crate::bivariate::frank::Frank;
use crate::correlation::kendall_tau;
use crate::traits::BivariateExt;

/// Candidate pair-copula families for the selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairFamily {
  Independence,
  Gaussian,
  StudentT,
  Clayton,
  Frank,
  Bb1,
  Bb7,
}

impl PairFamily {
  /// The families the crate can estimate, in the order they are tried.
  pub const ALL: [PairFamily; 7] = [
    PairFamily::Independence,
    PairFamily::Gaussian,
    PairFamily::StudentT,
    PairFamily::Clayton,
    PairFamily::Frank,
    PairFamily::Bb1,
    PairFamily::Bb7,
  ];

  /// Number of free parameters.
  pub fn parameter_count(self) -> usize {
    match self {
      PairFamily::Independence => 0,
      PairFamily::Gaussian | PairFamily::Clayton | PairFamily::Frank => 1,
      PairFamily::StudentT | PairFamily::Bb1 | PairFamily::Bb7 => 2,
    }
  }
}

/// Information criterion used to pick the family of each edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SelectionCriterion {
  #[default]
  Aic,
  Bic,
}

/// Vine structure to fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VineStructure {
  #[default]
  DVine,
  CVine,
}

/// A fitted vine with the variable order the trees are built on.
#[derive(Clone, Debug)]
pub struct VineFit {
  /// The fitted vine on the variables in `order`.
  pub vine: RVine,
  /// Column indices of the data in the order the vine's variables refer to.
  pub order: Vec<usize>,
  /// Families chosen per tree and edge.
  pub families: Vec<Vec<PairFamily>>,
  pub log_likelihood: f64,
  pub parameter_count: usize,
  pub aic: f64,
  pub bic: f64,
}

/// Kendall's τ of two pseudo-observation columns.
fn tau_of(u: &[f64], v: &[f64]) -> f64 {
  kendalls::tau_b_with_comparator(u, v, |a: &f64, b: &f64| {
    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater)
  })
  .map(|(t, ..)| t)
  .unwrap_or(0.0)
}

fn log_likelihood(pair: &PairCopula, u: &[f64], v: &[f64]) -> f64 {
  u.iter().zip(v).map(|(&a, &b)| pair.log_density(a, b)).sum()
}

/// Estimates `family` on the pair `(u, v)`; `None` when the family cannot
/// represent the observed dependence (e.g. Clayton at negative τ).
fn estimate(family: PairFamily, u: &[f64], v: &[f64]) -> Option<PairCopula> {
  let tau = tau_of(u, v).clamp(-0.99, 0.99);
  let rho = (std::f64::consts::FRAC_PI_2 * tau).sin();
  match family {
    PairFamily::Independence => Some(PairCopula::Independence),
    PairFamily::Gaussian => Some(PairCopula::Gaussian { rho }),
    PairFamily::StudentT => {
      let best = [3.0_f64, 4.0, 6.0, 8.0, 12.0, 20.0, 40.0]
        .into_iter()
        .map(|nu| (nu, log_likelihood(&PairCopula::StudentT { rho, nu }, u, v)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;
      Some(PairCopula::StudentT { rho, nu: best.0 })
    }
    PairFamily::Clayton => (tau > 0.02).then(|| PairCopula::Clayton {
      theta: 2.0 * tau / (1.0 - tau),
    }),
    PairFamily::Frank => {
      if tau.abs() < 1e-3 {
        return None;
      }
      let mut frank = Frank::new(None, Some(tau));
      let theta = frank.compute_theta();
      frank.set_theta(theta);
      theta.is_finite().then_some(PairCopula::Frank { theta })
    }
    PairFamily::Bb1 | PairFamily::Bb7 => {
      if tau < 0.05 {
        return None;
      }
      let x = ndarray::stack![
        ndarray::Axis(1),
        Array1::from_vec(u.to_vec()),
        Array1::from_vec(v.to_vec())
      ];
      if family == PairFamily::Bb1 {
        let mut c = Bb1::default();
        c.fit_parameters(&x).ok()?;
        Some(PairCopula::Bb1 {
          theta: c.theta?,
          delta: c.delta,
        })
      } else {
        let mut c = Bb7::default();
        c.fit_parameters(&x).ok()?;
        Some(PairCopula::Bb7 {
          theta: c.theta?,
          delta: c.delta,
        })
      }
    }
  }
}

/// Best family on the pair by the criterion; returns the copula, family,
/// log-likelihood and parameter count.
fn select_pair(
  u: &[f64],
  v: &[f64],
  families: &[PairFamily],
  criterion: SelectionCriterion,
) -> (PairCopula, PairFamily, f64, usize) {
  let n = u.len() as f64;
  let mut best: Option<(PairCopula, PairFamily, f64, usize, f64)> = None;
  for &family in families {
    let Some(pair) = estimate(family, u, v) else {
      continue;
    };
    let ll = log_likelihood(&pair, u, v);
    if !ll.is_finite() {
      continue;
    }
    let k = family.parameter_count();
    let score = match criterion {
      SelectionCriterion::Aic => -2.0 * ll + 2.0 * k as f64,
      SelectionCriterion::Bic => -2.0 * ll + (k as f64) * n.ln(),
    };
    if best.as_ref().is_none_or(|b| score < b.4) {
      best = Some((pair, family, ll, k, score));
    }
  }
  let (pair, family, ll, k, _) = best.expect("Independence always estimates");
  (pair, family, ll, k)
}

/// Greedy maximal path on `|τ|` for the first D-vine tree: start from the
/// strongest pair and extend the path at whichever end has the strongest
/// remaining neighbour (Dißmann et al. 2013, §4.1 heuristic).
fn dvine_order(tau: &Array2<f64>) -> Vec<usize> {
  let d = tau.nrows();
  if d == 1 {
    return vec![0];
  }
  let mut best = (0, 1, 0.0_f64);
  for i in 0..d {
    for j in (i + 1)..d {
      if tau[(i, j)].abs() >= best.2 {
        best = (i, j, tau[(i, j)].abs());
      }
    }
  }
  let mut path = vec![best.0, best.1];
  let mut used = vec![false; d];
  used[best.0] = true;
  used[best.1] = true;
  while path.len() < d {
    let (head, tail) = (path[0], *path.last().expect("non-empty"));
    let mut candidate = (0usize, false, -1.0_f64);
    for k in 0..d {
      if used[k] {
        continue;
      }
      if tau[(head, k)].abs() > candidate.2 {
        candidate = (k, true, tau[(head, k)].abs());
      }
      if tau[(tail, k)].abs() > candidate.2 {
        candidate = (k, false, tau[(tail, k)].abs());
      }
    }
    used[candidate.0] = true;
    if candidate.1 {
      path.insert(0, candidate.0);
    } else {
      path.push(candidate.0);
    }
  }
  path
}

/// Fits a vine of the requested structure to the pseudo-observations `u`
/// (rows = observations, columns = variables in `[0, 1]`).
pub fn fit_vine(
  u: &Array2<f64>,
  structure: VineStructure,
  families: &[PairFamily],
  criterion: SelectionCriterion,
) -> Result<VineFit, Box<dyn Error>> {
  let d = u.ncols();
  if d < 2 {
    return Err("a vine needs at least two variables".into());
  }
  if u.nrows() < 10 {
    return Err("a vine fit needs at least ten observations".into());
  }
  let tau = kendall_tau(u);
  match structure {
    VineStructure::DVine => fit_dvine(u, &tau, families, criterion),
    VineStructure::CVine => fit_cvine(u, &tau, families, criterion),
  }
}

fn fit_dvine(
  u: &Array2<f64>,
  tau: &Array2<f64>,
  families: &[PairFamily],
  criterion: SelectionCriterion,
) -> Result<VineFit, Box<dyn Error>> {
  let d = u.ncols();
  let n = u.nrows() as f64;
  let order = dvine_order(tau);
  // Edge inputs for the current tree: left[i] = F(x_i | cond), right[i] = F(x_{i+m+1} | cond).
  let mut left: Vec<Vec<f64>> = (0..d - 1).map(|i| u.column(order[i]).to_vec()).collect();
  let mut right: Vec<Vec<f64>> = (0..d - 1)
    .map(|i| u.column(order[i + 1]).to_vec())
    .collect();
  let mut trees: Vec<Vec<PairCopula>> = Vec::with_capacity(d - 1);
  let mut chosen: Vec<Vec<PairFamily>> = Vec::with_capacity(d - 1);
  let (mut ll_total, mut k_total) = (0.0, 0usize);
  for m in 0..d - 1 {
    let edges = d - 1 - m;
    let mut tree = Vec::with_capacity(edges);
    let mut tree_families = Vec::with_capacity(edges);
    for i in 0..edges {
      let (pair, family, ll, k) = select_pair(&left[i], &right[i], families, criterion);
      ll_total += ll;
      k_total += k;
      tree.push(pair);
      tree_families.push(family);
    }
    if m + 1 < d - 1 {
      let mut next_left = Vec::with_capacity(edges - 1);
      let mut next_right = Vec::with_capacity(edges - 1);
      for i in 0..edges - 1 {
        // Left input: the left variable of edge i conditioned on its right end.
        next_left.push(
          left[i]
            .iter()
            .zip(&right[i])
            .map(|(&a, &b)| tree[i].h(a, b))
            .collect(),
        );
        // Right input: the right variable of edge i+1 conditioned on its left end.
        next_right.push(
          right[i + 1]
            .iter()
            .zip(&left[i + 1])
            .map(|(&a, &b)| tree[i + 1].h(a, b))
            .collect(),
        );
      }
      left = next_left;
      right = next_right;
    }
    trees.push(tree);
    chosen.push(tree_families);
  }
  let vine = RVine::from_dvine(DVine::new(d, trees)?);
  Ok(VineFit {
    vine,
    order,
    families: chosen,
    log_likelihood: ll_total,
    parameter_count: k_total,
    aic: -2.0 * ll_total + 2.0 * k_total as f64,
    bic: -2.0 * ll_total + k_total as f64 * n.ln(),
  })
}

fn fit_cvine(
  u: &Array2<f64>,
  tau: &Array2<f64>,
  families: &[PairFamily],
  criterion: SelectionCriterion,
) -> Result<VineFit, Box<dyn Error>> {
  let d = u.ncols();
  let n = u.nrows() as f64;
  // Root of the first tree: the variable with the largest total |τ|.
  let strength = |i: usize, t: &Array2<f64>| {
    (0..d)
      .filter(|&j| j != i)
      .map(|j| t[(i, j)].abs())
      .sum::<f64>()
  };
  let mut remaining: Vec<usize> = (0..d).collect();
  let root = remaining
    .iter()
    .copied()
    .max_by(|&a, &b| {
      strength(a, tau)
        .partial_cmp(&strength(b, tau))
        .unwrap_or(std::cmp::Ordering::Equal)
    })
    .expect("non-empty");
  remaining.retain(|&j| j != root);
  let mut order = vec![root];
  let mut root_values = u.column(root).to_vec();
  let mut values: Vec<(usize, Vec<f64>)> = remaining
    .iter()
    .map(|&j| (j, u.column(j).to_vec()))
    .collect();
  // Per tree: (variable, fitted pair, family) of every edge to the root.
  let mut tree_entries: Vec<Vec<(usize, PairCopula, PairFamily)>> = Vec::with_capacity(d - 1);
  let (mut ll_total, mut k_total) = (0.0, 0usize);
  for _m in 0..d - 1 {
    let mut entries = Vec::with_capacity(values.len());
    let mut next_values: Vec<(usize, Vec<f64>)> = Vec::with_capacity(values.len());
    for (j, column) in &values {
      let (pair, family, ll, k) = select_pair(&root_values, column, families, criterion);
      ll_total += ll;
      k_total += k;
      // Pseudo-observation of variable j given the root: h(x_j | root).
      let conditioned: Vec<f64> = column
        .iter()
        .zip(&root_values)
        .map(|(&a, &b)| pair.h(a, b))
        .collect();
      next_values.push((*j, conditioned));
      entries.push((*j, pair, family));
    }
    tree_entries.push(entries);
    if next_values.len() <= 1 {
      if let Some((j, _)) = next_values.first() {
        order.push(*j);
      }
      break;
    }
    // Next root: the strongest node among the conditioned pseudo-observations.
    let k = next_values.len();
    let mut pseudo = Array2::<f64>::zeros((u.nrows(), k));
    for (c, (_, col)) in next_values.iter().enumerate() {
      pseudo.column_mut(c).assign(&Array1::from_vec(col.clone()));
    }
    let tau_next = kendall_tau(&pseudo);
    let best = (0..k)
      .max_by(|&a, &b| {
        let sa: f64 = (0..k)
          .filter(|&x| x != a)
          .map(|x| tau_next[(a, x)].abs())
          .sum();
        let sb: f64 = (0..k)
          .filter(|&x| x != b)
          .map(|x| tau_next[(b, x)].abs())
          .sum();
        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
      })
      .expect("non-empty");
    let (new_root, new_root_values) = next_values.remove(best);
    order.push(new_root);
    root_values = new_root_values;
    values = next_values;
  }
  // `CVine` stores edge i of tree m as (m, m + i + 1) in the vine's variable
  // order, i.e. `order` above, so each tree's edges are laid out by the
  // position of their variable in that order.
  let mut position = vec![0usize; d];
  for (k, &j) in order.iter().enumerate() {
    position[j] = k;
  }
  let mut trees: Vec<Vec<PairCopula>> = Vec::with_capacity(d - 1);
  let mut chosen: Vec<Vec<PairFamily>> = Vec::with_capacity(d - 1);
  for mut entries in tree_entries {
    entries.sort_by_key(|(j, ..)| position[*j]);
    trees.push(entries.iter().map(|(_, pair, _)| *pair).collect());
    chosen.push(entries.iter().map(|(.., family)| *family).collect());
  }
  let vine = RVine::from_cvine(CVine::new(d, trees)?);
  Ok(VineFit {
    vine,
    order,
    families: chosen,
    log_likelihood: ll_total,
    parameter_count: k_total,
    aic: -2.0 * ll_total + 2.0 * k_total as f64,
    bic: -2.0 * ll_total + k_total as f64 * n.ln(),
  })
}

#[cfg(test)]
mod tests;
