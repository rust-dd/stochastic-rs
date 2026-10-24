//! Vertex-centred mesh and quadrature for the forward Kolmogorov equation.
//! Reference: Wyns & Du Toit (2016), §2 and §4, arXiv:1611.02961.

use crate::traits::RealExt;

/// The `(x, v)` mesh with its cell widths: `wx[i] = (Δx_i + Δx_{i+1}) / 2`,
/// the width of the cell around node `i`, `Δx_1 = Δx_{m1+1} = 0`.
pub(super) struct Mesh {
  pub x: Vec<f64>,
  pub v: Vec<f64>,
  pub wx: Vec<f64>,
  pub wv: Vec<f64>,
  /// The node the density starts on.
  pub i0: usize,
  pub j0: usize,
}

impl Mesh {
  /// A log-spot mesh clustered at `x0` by `x = x0 + c sinh(ξ)` on a uniform
  /// `ξ`, `x0` a node (`m1` is made odd). The variance mesh is uniform on
  /// `[0, v0]` and stretched by `v = v0 + d sinh(ξ)` above it, resolving
  /// both the attainable boundary and the Dirac start as required by
  /// Wyns & Du Toit (2016), §4. Both `0` and `v0` are exact nodes.
  pub fn new(
    x0: f64,
    x_half_width: f64,
    x_stretch: f64,
    m1: usize,
    v0: f64,
    v_max: f64,
    v_stretch: f64,
    m2: usize,
  ) -> Self {
    let m1 = m1.max(3) | 1;
    let half = (m1 - 1) / 2;
    let xi_max = (x_half_width / x_stretch).asinh();
    let d_xi = xi_max / half as f64;
    let mut x = (0..m1)
      .map(|i| x0 + x_stretch * ((i as f64 - half as f64) * d_xi).sinh())
      .collect::<Vec<_>>();
    x[half] = x0;

    let m2 = m2.max(3);
    let intervals = m2 - 1;
    let mut v = Vec::with_capacity(m2);
    let j0 = if v0 <= 0.0 {
      let xi_hi = (v_max / v_stretch).asinh();
      v.extend((0..m2).map(|j| v_stretch * (xi_hi * j as f64 / intervals as f64).sinh()));
      0
    } else {
      let xi_hi = ((v_max - v0) / v_stretch).asinh();
      let share = v0 / (v0 + v_stretch * xi_hi);
      let n_lo = ((intervals as f64 * share).round() as usize).clamp(1, intervals - 1);
      let n_hi = intervals - n_lo;
      v.extend((0..=n_lo).map(|k| v0 * k as f64 / n_lo as f64));
      v.extend((1..=n_hi).map(|k| v0 + v_stretch * (xi_hi * k as f64 / n_hi as f64).sinh()));
      v[n_lo] = v0;
      n_lo
    };
    v[0] = 0.0;
    v[m2 - 1] = v_max;

    let wx = cell_widths(&x);
    let wv = cell_widths(&v);
    Self {
      x,
      v,
      wx,
      wv,
      i0: half,
      j0,
    }
  }

  pub fn m1(&self) -> usize {
    self.x.len()
  }

  pub fn m2(&self) -> usize {
    self.v.len()
  }

  pub fn len(&self) -> usize {
    self.x.len() * self.v.len()
  }

  /// The cell-average Dirac at `(x0, v0)`: `1 / |Ω_{i0, j0}|` on that node.
  pub fn dirac(&self) -> Vec<f64> {
    let mut p = vec![0.0; self.len()];
    p[self.j0 * self.m1() + self.i0] = 1.0 / (self.wx[self.i0] * self.wv[self.j0]);
    p
  }

  /// `Σ_{i,j} ω_i ω'_j P_{i,j}`.
  pub fn mass(&self, p: &[f64]) -> f64 {
    let m1 = self.m1();
    self
      .wv
      .iter()
      .enumerate()
      .map(|(j, wv)| {
        wv * self
          .wx
          .iter()
          .enumerate()
          .map(|(i, wx)| wx * p[j * m1 + i])
          .sum::<f64>()
      })
      .sum()
  }

  /// The marginal density of the log-spot, `Σ_j ω'_j P_{i,j}` per node.
  pub fn marginal(&self, p: &[f64]) -> Vec<f64> {
    let m1 = self.m1();
    (0..m1)
      .map(|i| {
        self
          .wv
          .iter()
          .enumerate()
          .map(|(j, wv)| wv * p[j * m1 + i])
          .sum()
      })
      .collect()
  }

  /// `Σ_{i,j} ω_i ω'_j v_j P_{i,j}`, the mean of the variance under the
  /// (unnormalised) density.
  pub fn variance_mean(&self, p: &[f64]) -> f64 {
    let m1 = self.m1();
    self
      .wv
      .iter()
      .zip(&self.v)
      .enumerate()
      .map(|(j, (wv, vj))| {
        wv * vj
          * self
            .wx
            .iter()
            .enumerate()
            .map(|(i, wx)| wx * p[j * m1 + i])
            .sum::<f64>()
      })
      .sum()
  }

  /// Wyns & Du Toit's (4.5): the trapezoid `Σ_j v_j |P_{i,j}| ω'_j /
  /// Σ_j |P_{i,j}| ω'_j` per log-spot node, `None` where the column carries
  /// no mass beyond round-off — below `threshold` of the heaviest column.
  pub fn conditional_variance(&self, p: &[f64], threshold: f64) -> Vec<Option<f64>> {
    let m1 = self.m1();
    let sums = (0..m1)
      .map(|i| {
        let (mut num, mut den) = (0.0, 0.0);
        for (j, (wv, vj)) in self.wv.iter().zip(&self.v).enumerate() {
          let weight = p[j * m1 + i].abs() * wv;
          num += vj * weight;
          den += weight;
        }
        (num, den)
      })
      .collect::<Vec<_>>();
    let heaviest = sums.iter().map(|(_, den)| *den).fold(0.0, f64::max_or_nan);
    sums
      .into_iter()
      .map(|(num, den)| (den > threshold * heaviest && den > 0.0).then(|| num / den))
      .collect()
  }
}

/// `ω_i = (Δx_i + Δx_{i+1}) / 2` with the two outer widths zero.
fn cell_widths(nodes: &[f64]) -> Vec<f64> {
  let n = nodes.len();
  (0..n)
    .map(|i| {
      let left = if i == 0 { 0.0 } else { nodes[i] - nodes[i - 1] };
      let right = if i + 1 == n {
        0.0
      } else {
        nodes[i + 1] - nodes[i]
      };
      0.5 * (left + right)
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use super::Mesh;

  #[test]
  fn variance_mesh_resolves_zero_and_the_initial_variance() {
    for (v0, theta) in [(0.0625_f64, 0.16_f64), (0.0348, 0.0348), (0.0, 0.04)] {
      let scale = v0.max(theta);
      let mesh = Mesh::new(0.0, 3.4, 0.2, 201, v0, 15.0 * scale, 0.2 * scale, 100);
      assert_eq!(mesh.v[0], 0.0);
      assert_eq!(mesh.v[mesh.j0], v0);
      assert!(mesh.v.windows(2).all(|pair| pair[1] > pair[0]));
      let at_zero = mesh.v[1];
      let at_start = mesh.v[mesh.j0 + 1] - v0;
      assert!((at_zero / at_start - 1.0).abs() < 0.05);
      assert!(mesh.v[99] - mesh.v[98] > 10.0 * at_zero);
      assert!((mesh.mass(&mesh.dirac()) - 1.0).abs() < 1e-15);
    }
  }
}
