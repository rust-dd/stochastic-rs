use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use stochastic_rs_distributions::RealExt;

use super::Generator;
use super::NacNode;

impl Generator {
  /// `(ln ψ⁻¹(u), ln |ψ⁻¹'(u)|)` for `0 < u < 1`.
  fn ln_leaf(self, theta: f64, u: f64) -> (f64, f64) {
    let ln_u = u.ln();
    match self {
      Self::Clayton => (ln_expm1(-theta * ln_u), theta.ln() - (1.0 + theta) * ln_u),
      Self::Gumbel => {
        let ln_x = (-ln_u).ln();
        (theta * ln_x, theta.ln() + (theta - 1.0) * ln_x - ln_u)
      }
    }
  }

  /// `ln ψ̊(t)` for the node map `ψ̊ = ψ_parent⁻¹ ∘ ψ_child`: `(1 + t)^α − 1` Clayton, `t^α` Gumbel.
  fn ln_node_map(self, alpha: f64, ln_t: f64) -> f64 {
    match self {
      Self::Clayton => ln_expm1(alpha * ln_1p_exp(ln_t)),
      Self::Gumbel => alpha * ln_t,
    }
  }

  /// `ln b` for the base `b` of the inner coefficients `b^{αl − n} |s_nl(α)|`: `1 + t` Clayton, `t` Gumbel.
  fn ln_base(self, ln_t: f64) -> f64 {
    match self {
      Self::Clayton => ln_1p_exp(ln_t),
      Self::Gumbel => ln_t,
    }
  }
}

/// A subtree node with `ln |s_nl(α)|`, `α = θ_parent / θ`, for `l ≤ n ≤ size`.
struct Node {
  theta: f64,
  alpha: f64,
  leaves: Vec<usize>,
  children: Vec<usize>,
  size: usize,
  ln_s: Vec<f64>,
}

/// The tree children-first, and the root's derivative table: Clayton `ln (1/θ)_n` (rising), Gumbel `ln |s_nk(1/θ)|`.
pub(super) struct Density {
  generator: Generator,
  nodes: Vec<Node>,
  root_table: Vec<f64>,
}

struct Scratch {
  ln_t: Vec<f64>,
  polys: Vec<Vec<f64>>,
  b: Vec<f64>,
  tmp: Vec<f64>,
  t_terms: Vec<f64>,
}

impl Density {
  pub(super) fn new(generator: Generator, root: &NacNode) -> Self {
    let mut nodes = vec![];
    flatten(root, None, &mut nodes);
    let d = nodes.last().map_or(0, |n| n.size);
    let root_table = match generator {
      Generator::Clayton => ln_rising(1.0 / root.theta, d),
      Generator::Gumbel => ln_s_triangle(1.0 / root.theta, d),
    };
    Self {
      generator,
      nodes,
      root_table,
    }
  }

  pub(super) fn log_pdf(&self, x: ArrayView2<f64>) -> Array1<f64> {
    let mut scratch = Scratch {
      ln_t: vec![0.0; self.nodes.len()],
      polys: vec![vec![]; self.nodes.len()],
      b: vec![],
      tmp: vec![],
      t_terms: vec![],
    };
    x.rows()
      .into_iter()
      .map(|u| self.row(u, &mut scratch))
      .collect()
  }

  /// Every coefficient is the absolute value of Hofert & Pham's signed one (their §5 fixes the signs), so each sum runs
  /// in logs over positive terms; a node's `Ã` follows their eq. (31), carried to any depth as their Remark 6.2 says.
  fn row(&self, u: ArrayView1<f64>, s: &mut Scratch) -> f64 {
    if u.iter().any(|&x| x <= 0.0 || x >= 1.0) {
      return f64::NAN;
    }
    let mut ln_jacobian = 0.0;
    let last = self.nodes.len() - 1;
    for i in 0..last {
      let ln_t = self.gather(i, u, s, &mut ln_jacobian);
      let node = &self.nodes[i];
      let ln_base = self.generator.ln_base(ln_t);
      let top = s.b.len() - 1;
      let low = s
        .b
        .iter()
        .position(|&x| x > f64::NEG_INFINITY)
        .unwrap_or(top);
      let out = &mut s.polys[i];
      out.clear();
      for l in 0..=top {
        let b = &s.b;
        let term = move |n: usize| {
          node.ln_s[tri(n, l)] + (node.alpha * l as f64 - n as f64) * ln_base + b[n]
        };
        out.push(lse((l.max(low)..=top).map(term)));
      }
    }
    let ln_t = self.gather(last, u, s, &mut ln_jacobian);
    ln_jacobian + self.root_sum(&s.b, ln_t)
  }

  /// Sets `s.b` to the product of the node's part polynomials (a leaf is `v`) and returns `ln t` at the node.
  fn gather(&self, i: usize, u: ArrayView1<f64>, s: &mut Scratch, ln_jacobian: &mut f64) -> f64 {
    let node = &self.nodes[i];
    s.t_terms.clear();
    s.b.clear();
    s.b.resize(node.leaves.len() + 1, f64::NEG_INFINITY);
    s.b[node.leaves.len()] = 0.0;
    for &j in &node.leaves {
      let (ln_g, ln_dg) = self.generator.ln_leaf(node.theta, u[j]);
      s.t_terms.push(ln_g);
      *ln_jacobian += ln_dg;
    }
    for &c in &node.children {
      let ln_t = s.ln_t[c];
      s.t_terms
        .push(self.generator.ln_node_map(self.nodes[c].alpha, ln_t));
      cauchy(&s.b, &s.polys[c], &mut s.tmp);
      std::mem::swap(&mut s.b, &mut s.tmp);
    }
    let ln_t = lse(s.t_terms.iter().copied());
    s.ln_t[i] = ln_t;
    ln_t
  }

  /// `ln Σ_n B̃_n (−1)^n ψ^{(n)}(t)`: Clayton `(1/θ)_n (1 + t)^{−n−1/θ}`, Gumbel `ψ(t) t^{−n} Σ_k |s_nk(1/θ)| t^{k/θ}`
  /// (Hofert, Mächler & McNeil 2012, Thm 3.2 (2) and (4)).
  fn root_sum(&self, b: &[f64], ln_t: f64) -> f64 {
    let theta = self.nodes[self.nodes.len() - 1].theta;
    let table = &self.root_table;
    let finite = b.iter().position(|&x| x > f64::NEG_INFINITY).unwrap_or(0);
    match self.generator {
      Generator::Clayton => {
        let ln_1p_t = ln_1p_exp(ln_t);
        lse((finite..b.len()).map(|n| b[n] + table[n] - (n as f64 + 1.0 / theta) * ln_1p_t))
      }
      Generator::Gumbel => {
        let ln_x = ln_t / theta;
        let x = ln_x.exp();
        lse((finite..b.len()).map(|n| {
          let ln_p = if n == 0 {
            0.0
          } else {
            lse((1..=n).map(|k| table[tri(n, k)] + k as f64 * ln_x)) - n as f64 * ln_t
          };
          b[n] - x + ln_p
        }))
      }
    }
  }
}

/// Pushes the subtree children-first and returns its index; a subtree without leaves has `C = 1` and is dropped.
fn flatten(node: &NacNode, parent_theta: Option<f64>, out: &mut Vec<Node>) -> Option<usize> {
  let children = node
    .children
    .iter()
    .filter_map(|c| flatten(c, Some(node.theta), out))
    .collect::<Vec<_>>();
  let size = node.leaves.len() + children.iter().map(|&c| out[c].size).sum::<usize>();
  if size == 0 && parent_theta.is_some() {
    return None;
  }
  let alpha = parent_theta.map_or(1.0, |p| p / node.theta);
  out.push(Node {
    theta: node.theta,
    alpha,
    leaves: node.leaves.clone(),
    children,
    size,
    ln_s: parent_theta.map_or_else(Vec::new, |_| ln_s_triangle(alpha, size)),
  });
  Some(out.len() - 1)
}

fn tri(n: usize, k: usize) -> usize {
  n * (n + 1) / 2 + k
}

/// `ln |s_nk(α)|`, `k ≤ n ≤ top`, by `|s_{n+1,k}| = (n − αk)|s_nk| + α|s_{n,k−1}|`: positive terms for `0 < α ≤ 1`, where
/// the Stirling sum of Hofert, Mächler & McNeil (2012), Thm 3.2 (4), alternates and cancels.
fn ln_s_triangle(alpha: f64, top: usize) -> Vec<f64> {
  let mut out = vec![f64::NEG_INFINITY; tri(top + 1, 0)];
  out[0] = 0.0;
  let ln_alpha = alpha.ln();
  for n in 0..top {
    for k in 0..=n + 1 {
      let stay = if k <= n {
        (n as f64 - alpha * k as f64).ln() + out[tri(n, k)]
      } else {
        f64::NEG_INFINITY
      };
      let step = if k >= 1 {
        ln_alpha + out[tri(n, k - 1)]
      } else {
        f64::NEG_INFINITY
      };
      out[tri(n + 1, k)] = lse([stay, step]);
    }
  }
  out
}

fn ln_rising(x: f64, top: usize) -> Vec<f64> {
  let mut out = vec![0.0; top + 1];
  for k in 0..top {
    out[k + 1] = out[k] + (x + k as f64).ln();
  }
  out
}

fn cauchy(a: &[f64], b: &[f64], out: &mut Vec<f64>) {
  out.clear();
  for n in 0..a.len() + b.len() - 1 {
    let first = n.saturating_sub(b.len() - 1);
    out.push(lse((first..=n.min(a.len() - 1)).map(|i| a[i] + b[n - i])));
  }
}

/// `ln Σ e^{x_i}`, pulling out the largest term; `−∞` for no terms, NaN if one is NaN.
fn lse<I>(terms: I) -> f64
where
  I: IntoIterator<Item = f64> + Clone,
{
  let m = terms
    .clone()
    .into_iter()
    .fold(f64::NEG_INFINITY, f64::max_or_nan);
  if m == f64::NEG_INFINITY {
    return m;
  }
  m + terms
    .into_iter()
    .map(|x| if x == m { 1.0 } else { (x - m).exp() })
    .sum::<f64>()
    .ln()
}

fn ln_expm1(x: f64) -> f64 {
  if x > std::f64::consts::LN_2 {
    x + (-(-x).exp()).ln_1p()
  } else {
    x.exp_m1().ln()
  }
}

fn ln_1p_exp(x: f64) -> f64 {
  if x > 0.0 {
    x + (-x).exp().ln_1p()
  } else {
    x.exp().ln_1p()
  }
}

#[cfg(test)]
mod tests;
