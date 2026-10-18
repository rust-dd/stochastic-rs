//! Dißmann, Brechmann, Czado & Kurowicka (2013), Algorithm 3.1 with a path (D-vine) or a star (C-vine) for each tree of
//! largest Σ|τ̂|, each edge's family by AIC or BIC over maximum-likelihood fits, its h-functions giving the next tree's data.
//! Reference: Dißmann, J., Brechmann, E. C., Czado, C. & Kurowicka, D. (2013), "Selecting and estimating regular vine copulae and application to financial returns", Computational Statistics & Data Analysis 59, 52–69, doi:10.1016/j.csda.2012.08.010 (preprint arXiv:1202.2002).
//! Reference: Aas, K., Czado, C., Frigessi, A. & Bakken, H. (2009), "Pair-copula constructions of multiple dependence", Insurance: Mathematics and Economics 44(2), 182–198, doi:10.1016/j.insmatheco.2007.02.001.
//! Reference: Held, M. & Karp, R. M. (1962), "A Dynamic Programming Approach to Sequencing Problems", Journal of the Society for Industrial and Applied Mathematics 10(1), 196–210, doi:10.1137/0110015.

use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;

use super::cvine::CVine;
use super::dvine::DVine;
use super::dvine::PairCopula;
use super::dvine::pair_copula;
use super::rvine::RVine;
use crate::bivariate::BivariateFamily;
use crate::bivariate::bb1::Bb1;
use crate::bivariate::bb7::Bb7;
use crate::bivariate::clayton::Clayton;
use crate::bivariate::conditional::check_open;
use crate::bivariate::frank::Frank;
use crate::bivariate::gaussian::GaussianCopula;
use crate::bivariate::t_copula::TCopula;
use crate::correlation::check_defined;
use crate::correlation::kendall_tau;
use crate::correlation::kendall_tau_b;
use crate::correlation::undefined_columns;
use crate::correlation::undefined_tau_b;
use crate::error::CopulaError;
use crate::optim::SQRT_MACHEPS;
use crate::optim::localmin;
use crate::student_t;
use crate::student_t::TDensity;
use crate::traits::BivariateExt;

/// Information criterion used to pick the family of each edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SelectionCriterion {
  #[default]
  Aic,
  Bic,
}

/// Vine structure to fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum VineStructure {
  #[default]
  DVine,
  CVine,
}

/// The selection's score sheet.
#[derive(Clone, Debug)]
pub(crate) struct FitSummary {
  pub(crate) families: Vec<Vec<BivariateFamily>>,
  pub(crate) log_likelihood: f64,
  pub(crate) parameter_count: usize,
  pub(crate) aic: f64,
  pub(crate) bic: f64,
}

impl FitSummary {
  fn new(
    families: Vec<Vec<BivariateFamily>>,
    log_likelihood: f64,
    parameter_count: usize,
    n: f64,
  ) -> Self {
    Self {
      families,
      log_likelihood,
      parameter_count,
      aic: -2.0 * log_likelihood + 2.0 * parameter_count as f64,
      bic: -2.0 * log_likelihood + parameter_count as f64 * n.ln(),
    }
  }
}

/// A fitted vine with its score sheet; the variable order lives on the vine (`vine.order()`).
#[derive(Clone, Debug)]
pub struct VineFit {
  pub vine: RVine,
  /// Families chosen per tree and edge.
  pub families: Vec<Vec<BivariateFamily>>,
  pub log_likelihood: f64,
  pub parameter_count: usize,
  pub aic: f64,
  pub bic: f64,
}

/// Kendall's τ-b of two pseudo-observation columns.
fn tau_of(u: &[f64], v: &[f64]) -> f64 {
  kendall_tau_b(ArrayView1::from(u), ArrayView1::from(v)).expect("check_vine_data checked n >= 10")
}

fn log_likelihood(pair: &PairCopula, u: &[f64], v: &[f64]) -> f64 {
  u.iter().zip(v).map(|(&a, &b)| pair.log_density(a, b)).sum()
}

/// Estimates `family` on the pair `(u, v)` by maximum likelihood; `None` when its Kendall start cannot represent the
/// dependence (e.g. Clayton at negative τ) or has no finite likelihood, and for a Student t whose ν̂ exceeds 30.
fn estimate(family: BivariateFamily, u: &[f64], v: &[f64]) -> Option<PairCopula> {
  let tau = tau_of(u, v).clamp(-0.99, 0.99);
  match family {
    BivariateFamily::Independence => Some(PairCopula::Independence),
    BivariateFamily::Gaussian => {
      maximise::<GaussianCopula>(tau, u, v, |rho| PairCopula::Gaussian { rho })
    }
    BivariateFamily::TCopula => student_t(u, v),
    BivariateFamily::Clayton => {
      if tau <= 0.02 {
        return None;
      }
      maximise::<Clayton>(tau, u, v, |theta| PairCopula::Clayton { theta })
    }
    BivariateFamily::Frank => {
      if tau.abs() < 1e-3 {
        return None;
      }
      maximise::<Frank>(tau, u, v, |theta| PairCopula::Frank { theta })
    }
    BivariateFamily::Bb1 | BivariateFamily::Bb7 => {
      if tau < 0.05 {
        return None;
      }
      let (u, v) = (ArrayView1::from(u), ArrayView1::from(v));
      if family == BivariateFamily::Bb1 {
        let mut c = Bb1::default();
        c.fit_parameters(u, v).ok()?;
        Some(PairCopula::Bb1 {
          theta: c.theta()?,
          delta: c.delta(),
        })
      } else {
        let mut c = Bb7::default();
        c.fit_parameters(u, v).ok()?;
        Some(PairCopula::Bb7 {
          theta: c.theta()?,
          delta: c.delta(),
        })
      }
    }
    BivariateFamily::Amh
    | BivariateFamily::Fgm
    | BivariateFamily::Galambos
    | BivariateFamily::Gumbel
    | BivariateFamily::HuslerReiss
    | BivariateFamily::Joe
    | BivariateFamily::MarshallOlkin
    | BivariateFamily::Plackett => None,
  }
}

/// The τ the family reaches inside its θ bounds: `localmin` never evaluates the ends, and a τ past what `theta_from_tau`
/// resolves scores `+∞`.
fn tau_range(copula: &impl BivariateExt) -> (f64, f64) {
  let (lo, hi) = copula.theta_bounds();
  let reach = |theta: f64| {
    if theta.is_finite() {
      copula.tau_from_theta(theta)
    } else {
      theta.signum()
    }
  };
  (reach(lo), reach(hi))
}

/// The minimised `−ll`, `+∞` where the likelihood is not finite, which `localmin` then steps away from.
fn objective(ll: f64) -> f64 {
  if ll.is_finite() { -ll } else { f64::INFINITY }
}

/// The likelihood maximised by `localmin` over the family's τ range at θ = `theta_from_tau(τ)`, kept where it beats the
/// Kendall start `tau`; `None` when the start has no finite likelihood.
fn maximise<C: BivariateExt + Default>(
  tau: f64,
  u: &[f64],
  v: &[f64],
  pair: impl Fn(f64) -> PairCopula,
) -> Option<PairCopula> {
  let copula = C::default();
  let at = |t: f64| copula.theta_from_tau(t).ok().map(&pair);
  let nll = |t: f64| at(t).map_or(f64::INFINITY, |p| objective(log_likelihood(&p, u, v)));
  let start = nll(tau);
  if !start.is_finite() {
    return None;
  }
  let (lo, hi) = tau_range(&copula);
  let (best, value) = localmin(lo, hi, SQRT_MACHEPS, SQRT_MACHEPS, &nll);
  at(if value <= start { best } else { tau })
}

/// Student t by maximum likelihood, ν by `localmin` on `(2, 50)` profiled over ρ; `None` past ν̂ = 30, where Dißmann et
/// al. (2013, §3.2) take the Gaussian instead.
fn student_t(u: &[f64], v: &[f64]) -> Option<PairCopula> {
  let copula = TCopula::default();
  let (lo, hi) = tau_range(&copula);
  let profile = |nu: f64| {
    let x: Vec<f64> = u
      .iter()
      .map(|&a| student_t::quantile(pair_copula::clamp(a), nu))
      .collect();
    let y: Vec<f64> = v
      .iter()
      .map(|&b| student_t::quantile(pair_copula::clamp(b), nu))
      .collect();
    let nll = |t: f64| match copula.theta_from_tau(t) {
      Ok(rho) => {
        let density = TDensity::new(rho, nu);
        objective(x.iter().zip(&y).map(|(&a, &b)| density.ln_c(a, b)).sum())
      }
      Err(_) => f64::INFINITY,
    };
    localmin(lo, hi, SQRT_MACHEPS, SQRT_MACHEPS, nll)
  };
  // `localmin` returns the latest of its least values, which `inner` keeps the same way.
  let mut inner = (f64::NAN, f64::INFINITY);
  let (nu, value) = localmin(2.0, 50.0, 1e-3, 1e-3, |nu| {
    let (tau, value) = profile(nu);
    if value <= inner.1 {
      inner = (tau, value);
    }
    value
  });
  if nu > 30.0 || !value.is_finite() {
    return None;
  }
  let tau = inner.0;
  Some(PairCopula::StudentT {
    rho: copula.theta_from_tau(tau).ok()?,
    nu,
  })
}

/// The criterion's best family on the pair (copula, family, log-likelihood, parameter count), `None` if no family fits;
/// independence competes by its score, without the optional Genest–Favre pre-test of Dißmann et al. (2013, §3.2).
fn select_pair(
  u: &[f64],
  v: &[f64],
  families: &[BivariateFamily],
  criterion: SelectionCriterion,
) -> Option<(PairCopula, BivariateFamily, f64, usize)> {
  let n = u.len() as f64;
  let mut best: Option<(PairCopula, BivariateFamily, f64, usize, f64)> = None;
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
  best.map(|(pair, family, ll, k, _)| (pair, family, ll, k))
}

/// `Unsupported` naming the data columns of the edge `a`–`b` given `given` that no family in `families` represents.
fn unrepresented(a: usize, b: usize, given: &[usize], families: &[BivariateFamily]) -> CopulaError {
  let mut given = given.to_vec();
  given.sort_unstable();
  let given = if given.is_empty() {
    String::new()
  } else {
    format!(" given columns {given:?}")
  };
  CopulaError::Unsupported(format!(
    "no candidate family represents the edge between columns {} and {}{given}: {families:?}",
    a.min(b),
    a.max(b)
  ))
}

/// The first D-vine tree, the Hamiltonian path of largest Σ|τ̂| (Dißmann et al. 2013, §3.1): exact by Held–Karp up to
/// `d = 8`, the greedy extension from the strongest pair above.
fn dvine_order(tau: ArrayView2<f64>) -> Vec<usize> {
  if tau.nrows() <= 8 {
    heaviest_path(tau)
  } else {
    greedy_path(tau)
  }
}

/// Held–Karp: `weight[mask][j]` is the heaviest path through the columns in `mask` ending at `j`; returned from its smaller end.
fn heaviest_path(tau: ArrayView2<f64>) -> Vec<usize> {
  let d = tau.nrows();
  let full = (1_usize << d) - 1;
  let mut weight = vec![f64::NEG_INFINITY; (full + 1) * d];
  let mut previous = vec![usize::MAX; (full + 1) * d];
  for j in 0..d {
    weight[(1 << j) * d + j] = 0.0;
  }
  for mask in 1..=full {
    for j in (0..d).filter(|&j| mask & (1 << j) != 0) {
      let w = weight[mask * d + j];
      for k in (0..d).filter(|&k| mask & (1 << k) == 0) {
        let (next, through) = ((mask | (1 << k)) * d + k, w + tau[(j, k)].abs());
        if through > weight[next] {
          weight[next] = through;
          previous[next] = j;
        }
      }
    }
  }
  let mut end = (0..d)
    .max_by(|&a, &b| weight[full * d + a].total_cmp(&weight[full * d + b]))
    .expect("d >= 1");
  let (mut path, mut mask) = (Vec::with_capacity(d), full);
  loop {
    path.push(end);
    let before = previous[mask * d + end];
    if before == usize::MAX {
      break;
    }
    mask &= !(1 << end);
    end = before;
  }
  if path[0] > path[d - 1] {
    path.reverse();
  }
  path
}

/// Start from the strongest pair and extend the path at whichever end has the strongest remaining neighbour.
fn greedy_path(tau: ArrayView2<f64>) -> Vec<usize> {
  let d = tau.nrows();
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

/// Columns, rows, range and candidates every structure shares, then the τ-b matrix, defined for every column.
fn check_vine_data(
  u: ArrayView2<f64>,
  families: &[BivariateFamily],
) -> Result<Array2<f64>, CopulaError> {
  let d = u.ncols();
  if d < 2 {
    return Err(CopulaError::InvalidStructure(format!(
      "a vine needs dim >= 2, got {d}"
    )));
  }
  if u.nrows() < 10 {
    return Err(CopulaError::InsufficientData {
      needed: 10,
      got: u.nrows(),
    });
  }
  if u.iter().any(|v| !(0.0..=1.0).contains(v)) {
    return Err(CopulaError::MarginalOutOfRange);
  }
  check_open(u)?;
  if let Some(family) = families
    .iter()
    .find(|f| !BivariateFamily::VINE_CANDIDATES.contains(f))
  {
    return Err(CopulaError::Unsupported(format!(
      "the vine fitter cannot estimate {family:?}; the candidates are BivariateFamily::VINE_CANDIDATES"
    )));
  }
  let tau = kendall_tau(u)?;
  check_defined(tau.view())?;
  Ok(tau)
}

/// Fits a vine of `structure` to the pseudo-observations `u` (rows = observations); `MarginalOutOfRange` outside `[0, 1]`, `MarginalOnBoundary` at
/// 0 or 1, `Unsupported` for a family outside `VINE_CANDIDATES` or an edge no family represents, `InvalidStructure` for a constant or NaN column.
pub fn fit_vine(
  u: ArrayView2<f64>,
  structure: VineStructure,
  families: &[BivariateFamily],
  criterion: SelectionCriterion,
) -> Result<VineFit, CopulaError> {
  let (vine, summary) = match structure {
    VineStructure::DVine => fit_dvine(u, families, criterion).map(|(v, s)| (RVine::D(v), s))?,
    VineStructure::CVine => fit_cvine(u, families, criterion).map(|(v, s)| (RVine::C(v), s))?,
  };
  Ok(VineFit {
    vine,
    families: summary.families,
    log_likelihood: summary.log_likelihood,
    parameter_count: summary.parameter_count,
    aic: summary.aic,
    bic: summary.bic,
  })
}

pub(crate) fn fit_dvine(
  u: ArrayView2<f64>,
  families: &[BivariateFamily],
  criterion: SelectionCriterion,
) -> Result<(DVine, FitSummary), CopulaError> {
  let tau = check_vine_data(u, families)?;
  let d = u.ncols();
  let n = u.nrows() as f64;
  let order = dvine_order(tau.view());
  // Edge inputs for the current tree: left[i] = F(x_i | cond), right[i] = F(x_{i+m+1} | cond).
  let mut left: Vec<Vec<f64>> = (0..d - 1).map(|i| u.column(order[i]).to_vec()).collect();
  let mut right: Vec<Vec<f64>> = (0..d - 1)
    .map(|i| u.column(order[i + 1]).to_vec())
    .collect();
  let mut trees: Vec<Vec<PairCopula>> = Vec::with_capacity(d - 1);
  let mut chosen: Vec<Vec<BivariateFamily>> = Vec::with_capacity(d - 1);
  let (mut ll_total, mut k_total) = (0.0, 0usize);
  for m in 0..d - 1 {
    let edges = d - 1 - m;
    let mut tree = Vec::with_capacity(edges);
    let mut tree_families = Vec::with_capacity(edges);
    for i in 0..edges {
      let (pair, family, ll, k) = select_pair(&left[i], &right[i], families, criterion)
        .ok_or_else(|| {
          unrepresented(
            order[i],
            order[i + m + 1],
            &order[i + 1..i + m + 1],
            families,
          )
        })?;
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
  let vine = DVine::new(d, trees)?.with_order(order);
  Ok((vine, FitSummary::new(chosen, ll_total, k_total, n)))
}

/// The variable with the largest total |τ| to the others.
fn strongest(tau: ArrayView2<f64>) -> usize {
  let k = tau.nrows();
  let strength = |i: usize| {
    (0..k)
      .filter(|&j| j != i)
      .map(|j| tau[(i, j)].abs())
      .sum::<f64>()
  };
  (0..k)
    .max_by(|&a, &b| strength(a).total_cmp(&strength(b)))
    .expect("non-empty")
}

/// The next C-vine root among `values`, the variables conditioned on the roots `given`: the one of largest Σ|τ̂|;
/// `InvalidStructure` when one of them has no τ-b.
fn next_root(values: &[(usize, Vec<f64>)], given: &[usize]) -> Result<usize, CopulaError> {
  let mut pseudo = Array2::<f64>::zeros((values[0].1.len(), values.len()));
  for (c, (_, col)) in values.iter().enumerate() {
    pseudo
      .column_mut(c)
      .assign(&ArrayView1::from(col.as_slice()));
  }
  let tau = kendall_tau(pseudo.view())?;
  if let Some(culprits) = undefined_columns(tau.view()) {
    let columns: Vec<usize> = culprits.iter().map(|&c| values[c].0).collect();
    return Err(undefined_tau_b(
      &columns,
      &format!(" given columns {given:?}"),
    ));
  }
  Ok(strongest(tau.view()))
}

pub(crate) fn fit_cvine(
  u: ArrayView2<f64>,
  families: &[BivariateFamily],
  criterion: SelectionCriterion,
) -> Result<(CVine, FitSummary), CopulaError> {
  let tau = check_vine_data(u, families)?;
  let d = u.ncols();
  let n = u.nrows() as f64;
  let root = strongest(tau.view());
  let mut order = vec![root];
  let mut root_values = u.column(root).to_vec();
  let mut values: Vec<(usize, Vec<f64>)> = (0..d)
    .filter(|&j| j != root)
    .map(|j| (j, u.column(j).to_vec()))
    .collect();
  // Per tree: (variable, fitted pair, family) of every edge to the root.
  let mut tree_entries: Vec<Vec<(usize, PairCopula, BivariateFamily)>> = Vec::with_capacity(d - 1);
  let (mut ll_total, mut k_total) = (0.0, 0usize);
  for _m in 0..d - 1 {
    let mut entries = Vec::with_capacity(values.len());
    let mut next_values: Vec<(usize, Vec<f64>)> = Vec::with_capacity(values.len());
    let (current, given) = order.split_last().expect("the root is in the order");
    for (j, column) in &values {
      let (pair, family, ll, k) = select_pair(&root_values, column, families, criterion)
        .ok_or_else(|| unrepresented(*current, *j, given, families))?;
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
    let (new_root, new_root_values) = next_values.remove(next_root(&next_values, &order)?);
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
  let mut chosen: Vec<Vec<BivariateFamily>> = Vec::with_capacity(d - 1);
  for mut entries in tree_entries {
    entries.sort_by_key(|(j, ..)| position[*j]);
    trees.push(entries.iter().map(|(_, pair, _)| *pair).collect());
    chosen.push(entries.iter().map(|(.., family)| *family).collect());
  }
  let vine = CVine::new(d, trees)?.with_order(order);
  Ok((vine, FitSummary::new(chosen, ll_total, k_total, n)))
}

#[cfg(test)]
mod tests;
