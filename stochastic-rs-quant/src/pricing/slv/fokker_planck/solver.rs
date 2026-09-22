//! The finite-volume semi-discretisation of Wyns & Du Toit (2016), §2.4, of
//! the forward Kolmogorov equation of the Heston SLV model on a vertex-centred
//! `(x, v)` mesh, split into its `x`-direction, `v`-direction and mixed parts,
//! and the Hundsdorfer–Verwer time step of their §3 on that split.
//!
//! Unknowns are cell averages `P[j * m1 + i] ≈ p(x_i, v_j)`, row-major in
//! `j`. Every flux at the boundary of the truncated domain is zero, so the
//! total mass `Σ ω_i ω'_j P_{i,j}` is constant in time up to the mixed flux
//! at the two corners on `v = v_max`, where the density is negligible.

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
  /// `ξ`, `x0` a node (`m1` is made odd), and a variance mesh clustered at
  /// `v0` by `v = v0 + d sinh(ξ)` with `v = 0` and `v0` exact nodes — the
  /// meshes of in 't Hout & Foulon (2010) that Wyns & Du Toit take over,
  /// laid so the Dirac start sits on a node.
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
      let xi_lo = (-v0 / v_stretch).asinh();
      let xi_hi = ((v_max - v0) / v_stretch).asinh();
      let share = -xi_lo / (xi_hi - xi_lo);
      let n_lo = ((intervals as f64 * share).round() as usize).clamp(1, intervals - 1);
      let n_hi = intervals - n_lo;
      v.extend((0..=n_lo).map(|k| v0 + v_stretch * (xi_lo * (1.0 - k as f64 / n_lo as f64)).sinh()));
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
      .map(|(j, wv)| wv * self.wx.iter().enumerate().map(|(i, wx)| wx * p[j * m1 + i]).sum::<f64>())
      .sum()
  }

  /// The marginal density of the log-spot, `Σ_j ω'_j P_{i,j}` per node.
  pub fn marginal(&self, p: &[f64]) -> Vec<f64> {
    let m1 = self.m1();
    (0..m1)
      .map(|i| self.wv.iter().enumerate().map(|(j, wv)| wv * p[j * m1 + i]).sum())
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
      .map(|(j, (wv, vj))| wv * vj * self.wx.iter().enumerate().map(|(i, wx)| wx * p[j * m1 + i]).sum::<f64>())
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
    let heaviest = sums.iter().map(|(_, den)| *den).fold(0.0, f64::max);
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
      let right = if i + 1 == n { 0.0 } else { nodes[i + 1] - nodes[i] };
      0.5 * (left + right)
    })
    .collect()
}

/// A direction operator: the three coefficients on the neighbours along one
/// axis, per node, from the central advection flux
/// `μ_{i−½} (P_{i−1} + P_i) / 2` and the conservative diffusion flux
/// `−(D_i P_i − D_{i−1} P_{i−1}) / Δx_i` of Wyns & Du Toit (2.22a–b), the
/// boundary fluxes zero.
pub(super) struct Direction {
  lower: Vec<f64>,
  diag: Vec<f64>,
  upper: Vec<f64>,
}

/// The coefficients along a line of nodes: `diffusion[k] = D` at node `k`,
/// `advection[k] = μ` at the midpoint between nodes `k − 1` and `k`
/// (`advection[0]` unused).
fn line_stencil(
  nodes: &[f64],
  widths: &[f64],
  diffusion: &[f64],
  advection: &[f64],
  lower: &mut [f64],
  diag: &mut [f64],
  upper: &mut [f64],
) {
  let n = nodes.len();
  for k in 0..n {
    let w = 1.0 / widths[k];
    let (mut lo, mut di, mut up) = (0.0, 0.0, 0.0);
    if k > 0 {
      let h = nodes[k] - nodes[k - 1];
      let a = advection[k];
      lo += 0.5 * a + diffusion[k - 1] / h;
      di += 0.5 * a - diffusion[k] / h;
    }
    if k + 1 < n {
      let h = nodes[k + 1] - nodes[k];
      let a = advection[k + 1];
      di -= 0.5 * a + diffusion[k] / h;
      up += diffusion[k + 1] / h - 0.5 * a;
    }
    lower[k] = lo * w;
    diag[k] = di * w;
    upper[k] = up * w;
  }
}

impl Direction {
  /// The `x` part at one time level: `D = ½ L_i² v_j`, `μ = r − q − ½ L² v_j`
  /// with `L` at a midpoint the mean of its two nodes.
  pub fn along_x(mesh: &Mesh, carry: f64, leverage: &[f64]) -> Self {
    let (m1, m2) = (mesh.m1(), mesh.m2());
    let mut out = Self::empty(m1 * m2);
    let mut diffusion = vec![0.0; m1];
    let mut advection = vec![0.0; m1];
    for j in 0..m2 {
      let vj = mesh.v[j];
      for i in 0..m1 {
        diffusion[i] = 0.5 * leverage[i] * leverage[i] * vj;
        if i > 0 {
          let l_mid = 0.5 * (leverage[i - 1] + leverage[i]);
          advection[i] = carry - 0.5 * l_mid * l_mid * vj;
        }
      }
      let row = j * m1..(j + 1) * m1;
      line_stencil(
        &mesh.x,
        &mesh.wx,
        &diffusion,
        &advection,
        &mut out.lower[row.clone()],
        &mut out.diag[row.clone()],
        &mut out.upper[row],
      );
    }
    out
  }

  /// The `v` part: `D = ½ ξ² v_j`, `μ = κ(θ − v)` at the midpoints; constant
  /// in time and the same on every `x` line.
  pub fn along_v(mesh: &Mesh, kappa: f64, theta: f64, xi: f64) -> Self {
    let (m1, m2) = (mesh.m1(), mesh.m2());
    let mut out = Self::empty(m1 * m2);
    let diffusion = mesh.v.iter().map(|vj| 0.5 * xi * xi * vj).collect::<Vec<_>>();
    let advection = (0..m2)
      .map(|j| {
        if j == 0 {
          0.0
        } else {
          kappa * (theta - 0.5 * (mesh.v[j - 1] + mesh.v[j]))
        }
      })
      .collect::<Vec<_>>();
    let (mut lo, mut di, mut up) = (vec![0.0; m2], vec![0.0; m2], vec![0.0; m2]);
    line_stencil(&mesh.v, &mesh.wv, &diffusion, &advection, &mut lo, &mut di, &mut up);
    for j in 0..m2 {
      for i in 0..m1 {
        let k = j * m1 + i;
        out.lower[k] = lo[j];
        out.diag[k] = di[j];
        out.upper[k] = up[j];
      }
    }
    out
  }

  fn empty(n: usize) -> Self {
    Self {
      lower: vec![0.0; n],
      diag: vec![0.0; n],
      upper: vec![0.0; n],
    }
  }

  /// `out = A p` along `x` (stride one).
  pub fn apply_x(&self, mesh: &Mesh, p: &[f64], out: &mut [f64]) {
    let (m1, m2) = (mesh.m1(), mesh.m2());
    for j in 0..m2 {
      for i in 0..m1 {
        let k = j * m1 + i;
        let mut acc = self.diag[k] * p[k];
        if i > 0 {
          acc += self.lower[k] * p[k - 1];
        }
        if i + 1 < m1 {
          acc += self.upper[k] * p[k + 1];
        }
        out[k] = acc;
      }
    }
  }

  /// `out = A p` along `v` (stride `m1`).
  pub fn apply_v(&self, mesh: &Mesh, p: &[f64], out: &mut [f64]) {
    let (m1, m2) = (mesh.m1(), mesh.m2());
    for j in 0..m2 {
      for i in 0..m1 {
        let k = j * m1 + i;
        let mut acc = self.diag[k] * p[k];
        if j > 0 {
          acc += self.lower[k] * p[k - m1];
        }
        if j + 1 < m2 {
          acc += self.upper[k] * p[k + m1];
        }
        out[k] = acc;
      }
    }
  }

  /// Solves `(I − c A) out = rhs` line by line along `x` by the Thomas
  /// algorithm.
  pub fn solve_x(&self, mesh: &Mesh, c: f64, rhs: &[f64], out: &mut [f64], scratch: &mut Scratch) {
    let (m1, m2) = (mesh.m1(), mesh.m2());
    scratch.resize(m1);
    for j in 0..m2 {
      let row = j * m1;
      for i in 0..m1 {
        scratch.load(i, -c * self.lower[row + i], 1.0 - c * self.diag[row + i], -c * self.upper[row + i], rhs[row + i]);
      }
      scratch.solve(m1, &mut out[row..row + m1], 1);
    }
  }

  /// Solves `(I − c A) out = rhs` line by line along `v`.
  pub fn solve_v(&self, mesh: &Mesh, c: f64, rhs: &[f64], out: &mut [f64], scratch: &mut Scratch) {
    let (m1, m2) = (mesh.m1(), mesh.m2());
    scratch.resize(m2);
    for i in 0..m1 {
      for j in 0..m2 {
        let k = j * m1 + i;
        scratch.load(j, -c * self.lower[k], 1.0 - c * self.diag[k], -c * self.upper[k], rhs[k]);
      }
      scratch.solve(m2, &mut out[i..], m1);
    }
  }
}

/// Tridiagonal work space for the Thomas algorithm.
#[derive(Default)]
pub(super) struct Scratch {
  sub: Vec<f64>,
  main: Vec<f64>,
  sup: Vec<f64>,
  rhs: Vec<f64>,
}

impl Scratch {
  fn resize(&mut self, n: usize) {
    for buffer in [&mut self.sub, &mut self.main, &mut self.sup, &mut self.rhs] {
      buffer.resize(n, 0.0);
    }
  }

  fn load(&mut self, k: usize, sub: f64, main: f64, sup: f64, rhs: f64) {
    self.sub[k] = sub;
    self.main[k] = main;
    self.sup[k] = sup;
    self.rhs[k] = rhs;
  }

  /// Forward elimination and back substitution in place; the solution lands
  /// in `out[0], out[stride], out[2 stride], …`.
  fn solve(&mut self, n: usize, out: &mut [f64], stride: usize) {
    for k in 1..n {
      let factor = self.sub[k] / self.main[k - 1];
      self.main[k] -= factor * self.sup[k - 1];
      self.rhs[k] -= factor * self.rhs[k - 1];
    }
    out[(n - 1) * stride] = self.rhs[n - 1] / self.main[n - 1];
    for k in (0..n - 1).rev() {
      out[k * stride] = (self.rhs[k] - self.sup[k] * out[(k + 1) * stride]) / self.main[k];
    }
  }
}

/// The mixed part (2.22c) at one time level: the corner flux
/// `ρ ξ L(x_{i−½}) v_{j−½} · (P_{i−1,j−1} + P_{i−1,j} + P_{i,j−1} + P_{i,j}) / 4`
/// with the ghost values (2.24), replaced on the corner row above `v = 0`
/// by the first-order forward mean of the row-`1` values, since the Heston
/// boundary `v = 0` is attainable.
pub(super) struct Mixed<'a> {
  mesh: &'a Mesh,
  rho_xi: f64,
  leverage: &'a [f64],
  corners: Vec<f64>,
}

impl<'a> Mixed<'a> {
  pub fn new(mesh: &'a Mesh, rho: f64, xi: f64, leverage: &'a [f64]) -> Self {
    Self {
      mesh,
      rho_xi: rho * xi,
      leverage,
      corners: vec![0.0; (mesh.m1() + 1) * (mesh.m2() + 1)],
    }
  }

  /// `out = A0 p`.
  pub fn apply(&mut self, p: &[f64], out: &mut [f64]) {
    let mesh = self.mesh;
    let (m1, m2) = (mesh.m1(), mesh.m2());
    let width = m1 + 1;
    for cj in 0..=m2 {
      let v_corner = match cj {
        0 => 0.0,
        _ if cj == m2 => mesh.v[m2 - 1],
        _ => 0.5 * (mesh.v[cj - 1] + mesh.v[cj]),
      };
      let (j_lo, j_hi) = (cj.saturating_sub(1), cj.min(m2 - 1));
      for ci in 0..=m1 {
        let l_corner = match ci {
          0 => self.leverage[0],
          _ if ci == m1 => self.leverage[m1 - 1],
          _ => 0.5 * (self.leverage[ci - 1] + self.leverage[ci]),
        };
        let (i_lo, i_hi) = (ci.saturating_sub(1), ci.min(m1 - 1));
        let average = if cj == 1 {
          0.5 * (p[m1 + i_lo] + p[m1 + i_hi])
        } else {
          0.25 * (p[j_lo * m1 + i_lo] + p[j_lo * m1 + i_hi] + p[j_hi * m1 + i_lo] + p[j_hi * m1 + i_hi])
        };
        self.corners[cj * width + ci] = self.rho_xi * l_corner * v_corner * average;
      }
    }
    for j in 0..m2 {
      let scale_j = 1.0 / mesh.wv[j];
      for i in 0..m1 {
        let f = |ci: usize, cj: usize| self.corners[cj * width + ci];
        out[j * m1 + i] =
          (f(i + 1, j + 1) - f(i + 1, j) - f(i, j + 1) + f(i, j)) * scale_j / mesh.wx[i];
      }
    }
  }
}

/// The split operator at one time level: `F = A0 + A1 + A2`.
pub(super) struct Level<'a> {
  pub x: Direction,
  pub mixed: Mixed<'a>,
}

impl<'a> Level<'a> {
  pub fn new(mesh: &'a Mesh, carry: f64, rho: f64, xi: f64, leverage: &'a [f64]) -> Self {
    Self {
      x: Direction::along_x(mesh, carry, leverage),
      mixed: Mixed::new(mesh, rho, xi, leverage),
    }
  }
}

/// Buffers one time step reuses.
pub(super) struct StepBuffers {
  f0: Vec<f64>,
  f1: Vec<f64>,
  f2: Vec<f64>,
  y0: Vec<f64>,
  y1: Vec<f64>,
  y2: Vec<f64>,
  g0: Vec<f64>,
  g1: Vec<f64>,
  g2: Vec<f64>,
  rhs: Vec<f64>,
  scratch: Scratch,
}

impl StepBuffers {
  pub fn new(n: usize) -> Self {
    Self {
      f0: vec![0.0; n],
      f1: vec![0.0; n],
      f2: vec![0.0; n],
      y0: vec![0.0; n],
      y1: vec![0.0; n],
      y2: vec![0.0; n],
      g0: vec![0.0; n],
      g1: vec![0.0; n],
      g2: vec![0.0; n],
      rhs: vec![0.0; n],
      scratch: Scratch::default(),
    }
  }
}

/// One Hundsdorfer–Verwer step (Wyns & Du Toit (3.1)) from `w` at the level
/// `prev` to the level `next`, or, with `douglas`, one Douglas step at
/// `θ = 1` — the implicit-Euler form of the Rannacher start-up.
#[allow(clippy::too_many_arguments)]
pub(super) fn step(
  mesh: &Mesh,
  along_v: &Direction,
  prev: &mut Level<'_>,
  next: &mut Level<'_>,
  dt: f64,
  theta: f64,
  douglas: bool,
  w: &[f64],
  out: &mut [f64],
  b: &mut StepBuffers,
) {
  let n = w.len();
  let theta_dt = if douglas { dt } else { theta * dt };
  prev.mixed.apply(w, &mut b.f0);
  prev.x.apply_x(mesh, w, &mut b.f1);
  along_v.apply_v(mesh, w, &mut b.f2);
  for k in 0..n {
    b.y0[k] = w[k] + dt * (b.f0[k] + b.f1[k] + b.f2[k]);
  }
  for k in 0..n {
    b.rhs[k] = b.y0[k] - theta_dt * b.f1[k];
  }
  next.x.solve_x(mesh, theta_dt, &b.rhs, &mut b.y1, &mut b.scratch);
  for k in 0..n {
    b.rhs[k] = b.y1[k] - theta_dt * b.f2[k];
  }
  along_v.solve_v(mesh, theta_dt, &b.rhs, &mut b.y2, &mut b.scratch);
  if douglas {
    out.copy_from_slice(&b.y2);
    return;
  }
  next.mixed.apply(&b.y2, &mut b.g0);
  next.x.apply_x(mesh, &b.y2, &mut b.g1);
  along_v.apply_v(mesh, &b.y2, &mut b.g2);
  for k in 0..n {
    let f_w = b.f0[k] + b.f1[k] + b.f2[k];
    let f_y2 = b.g0[k] + b.g1[k] + b.g2[k];
    b.y0[k] += 0.5 * dt * (f_y2 - f_w);
  }
  for k in 0..n {
    b.rhs[k] = b.y0[k] - theta_dt * b.g1[k];
  }
  next.x.solve_x(mesh, theta_dt, &b.rhs, &mut b.y1, &mut b.scratch);
  for k in 0..n {
    b.rhs[k] = b.y1[k] - theta_dt * b.g2[k];
  }
  along_v.solve_v(mesh, theta_dt, &b.rhs, out, &mut b.scratch);
}
