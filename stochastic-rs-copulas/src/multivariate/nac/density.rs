use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use stochastic_rs_distributions::RealExt;

use super::Generator;
use super::NacNode;
use crate::bivariate::conditional::ln_log_ratio;
use crate::bivariate::conditional::ln_ratio;

/// Up to this `θa` Clayton's offset is `ln(1 + t) − θa`, `t = Σ(e^{θ·part} − 1)`, cancelling by at most `θa` roundings;
/// past it the sum is taken about the anchor, `ln(Σ e^{θ·gap}(1 − e^{−θ·part}) + e^{−θa})`.
const CLAYTON_SUM_BELOW: f64 = 36.0;

/// Up to this `θ` the Gumbel gap is the difference of the cached `ln y`, whose rounding (`|ln y| < 37` on the open
/// interval) `θ` keeps under `1e-13`; past it, near a ratio of 1, the exact `ln(y_j/y_a)`.
const GUMBEL_EXACT_ABOVE: f64 = 4.0;

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

/// Per node, the leaf of its subtree's smallest `u`, whose `a = −ln u` is the largest `y`, and the offset `o` of
/// `L = −ln C` from `a`: Clayton `θ(L − a)`, Gumbel `θ ln(L/a)`, between 0 and `ln` of the subtree's leaf count.
struct Scratch {
  anchor: Vec<usize>,
  offset: Vec<f64>,
  y: Vec<f64>,
  ln_y: Vec<f64>,
  polys: Vec<Vec<f64>>,
  b: Vec<f64>,
  tmp: Vec<f64>,
  shifted: Vec<f64>,
  terms: Vec<(f64, f64)>,
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
    let n = self.nodes.len();
    let mut scratch = Scratch {
      anchor: vec![0; n],
      offset: vec![0.0; n],
      y: vec![0.0; x.ncols()],
      ln_y: vec![0.0; x.ncols()],
      polys: vec![vec![]; n],
      b: vec![],
      tmp: vec![],
      shifted: vec![],
      terms: vec![],
    };
    x.rows()
      .into_iter()
      .map(|u| self.row(u, &mut scratch))
      .collect()
  }

  /// Hofert & Pham's sums, coefficients made positive (§5), a node's `Ã` by eq. (31) to any depth, in logs, each node
  /// re-centred on its anchor so `θ·L` and `θ·y` enter only as `θ` times an exact gap.
  fn row(&self, u: ArrayView1<f64>, s: &mut Scratch) -> f64 {
    if u.iter().any(|&x| x <= 0.0 || x >= 1.0) {
      return f64::NAN;
    }
    for (j, &x) in u.iter().enumerate() {
      s.y[j] = -x.ln();
      if self.generator == Generator::Gumbel {
        s.ln_y[j] = s.y[j].ln();
      }
    }
    let mut ln_jacobian = 0.0;
    let last = self.nodes.len() - 1;
    for i in 0..=last {
      self.gather(i, u, s, &mut ln_jacobian);
      if i == last {
        break;
      }
      let node = &self.nodes[i];
      let offset = s.offset[i];
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
        let term =
          move |n: usize| node.ln_s[tri(n, l)] + (node.alpha * l as f64 - n as f64) * offset + b[n];
        out.push(lse((l.max(low)..=top).map(term)));
      }
    }
    ln_jacobian + self.root_sum(&s.b, s.y[s.anchor[last]], s.offset[last])
  }

  /// `θ` times the gap from anchor leaf `a` to leaf `j`, `≤ 0`: Clayton `ln(u_a/u_j)`, Gumbel `ln(y_j/y_a)`, each exact
  /// near a ratio of 1, where `θ` would scale the rounding of two separate logarithms.
  fn gap(&self, theta: f64, u: ArrayView1<f64>, s: &Scratch, a: usize, j: usize) -> f64 {
    if a == j {
      return 0.0;
    }
    match self.generator {
      Generator::Clayton => theta * ln_ratio(u[a], u[j]),
      Generator::Gumbel
        if theta > GUMBEL_EXACT_ABOVE && (0.5..=2.0).contains(&(s.y[j] / s.y[a])) =>
      {
        theta * ln_log_ratio(u[j], u[a], s.y[j], s.y[a])
      }
      Generator::Gumbel => theta * (s.ln_y[j] - s.ln_y[a]),
    }
  }

  /// Sets node `i`'s anchor and offset, `s.b` to the product of its part polynomials (a leaf is `v`, a child's comes
  /// shifted to this anchor), and adds its leaves' re-centred Jacobian terms.
  fn gather(&self, i: usize, u: ArrayView1<f64>, s: &mut Scratch, ln_jacobian: &mut f64) {
    let node = &self.nodes[i];
    let anchor = node
      .leaves
      .iter()
      .copied()
      .chain(node.children.iter().map(|&c| s.anchor[c]))
      .min_by(|&p, &q| u[p].total_cmp(&u[q]))
      .expect("a node holds a leaf");
    s.anchor[i] = anchor;
    let theta = node.theta;
    // Each part's `θ·gap` to the anchor, and Clayton's `θ·part`; a child's part is its `L`.
    s.terms.clear();
    for &j in &node.leaves {
      let gap = self.gap(theta, u, s, anchor, j);
      s.terms.push((gap, theta * s.y[j]));
      *ln_jacobian += theta.ln()
        + s.y[j]
        + gap
        + match self.generator {
          Generator::Clayton => 0.0,
          Generator::Gumbel => -s.ln_y[j],
        };
    }
    s.b.clear();
    s.b.resize(node.leaves.len() + 1, f64::NEG_INFINITY);
    s.b[node.leaves.len()] = 0.0;
    for &c in &node.children {
      let child = &self.nodes[c];
      let shift = self.gap(theta, u, s, anchor, s.anchor[c]);
      let lift = child.alpha * s.offset[c];
      s.terms
        .push((shift + lift, theta * s.y[s.anchor[c]] + lift));
      s.shifted.clear();
      s.shifted.extend(
        s.polys[c]
          .iter()
          .enumerate()
          .map(|(l, &p)| p + l as f64 * shift),
      );
      cauchy(&s.b, &s.shifted, &mut s.tmp);
      std::mem::swap(&mut s.b, &mut s.tmp);
    }
    let theta_a = theta * s.y[anchor];
    s.offset[i] = match self.generator {
      Generator::Clayton if theta_a <= CLAYTON_SUM_BELOW => {
        s.terms
          .iter()
          .map(|&(_, full)| full.exp_m1())
          .sum::<f64>()
          .ln_1p()
          - theta_a
      }
      Generator::Clayton => {
        let sum = s
          .terms
          .iter()
          .map(|&(gap, full)| gap.exp() * -(-full).exp_m1())
          .sum::<f64>();
        sum.ln() + ((-theta_a).exp() / sum).ln_1p()
      }
      Generator::Gumbel => lse(s.terms.iter().map(|&(gap, _)| gap)),
    };
    // The `1/θ` power of Clayton's root factor `1 + t` leaves `−a` beside its offset term.
    if i == self.nodes.len() - 1 && self.generator == Generator::Clayton {
      *ln_jacobian -= s.y[anchor];
    }
  }

  /// `ln Σ_n B̃_n (−1)^n ψ^{(n)}(t)` re-centred: Clayton `(1/θ)_n e^{−(n + 1/θ)o}`, Gumbel
  /// `e^{−no − L} Σ_k |s_nk(1/θ)| L^k`, `L = a e^{o/θ}` (Hofert, Mächler & McNeil 2012, Thm 3.2 (2) and (4)).
  fn root_sum(&self, b: &[f64], a: f64, offset: f64) -> f64 {
    let theta = self.nodes[self.nodes.len() - 1].theta;
    let table = &self.root_table;
    let finite = b.iter().position(|&x| x > f64::NEG_INFINITY).unwrap_or(0);
    match self.generator {
      Generator::Clayton => {
        lse((finite..b.len()).map(|n| b[n] + table[n] - (n as f64 + 1.0 / theta) * offset))
      }
      Generator::Gumbel => {
        let ln_l = a.ln() + offset / theta;
        let l = ln_l.exp();
        lse((finite..b.len()).map(|n| {
          let ln_p = if n == 0 {
            0.0
          } else {
            lse((1..=n).map(|k| table[tri(n, k)] + k as f64 * ln_l))
          };
          b[n] - n as f64 * offset - l + ln_p
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

#[cfg(test)]
mod tests;
