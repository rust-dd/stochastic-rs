//! Semi-discrete Heston operator `U' = A U + b(t)` of in 't Hout & Foulon
//! (2008), §2.2, split as `A = A0 + A1 + A2` (§2.4): `A0` the mixed
//! derivative, `A1` the `s`-direction terms, `A2` the `v`-direction terms,
//! with the reaction term `−r_d u` shared evenly between `A1` and `A2`.
//!
//! Unknowns live on `G = {(s_i, v_j): 1 ≤ i ≤ m1, 0 ≤ j ≤ m2 − 1}`, stored
//! row-major in `j` (`index = j · m1 + (i − 1)`). The Dirichlet data at
//! `s = s_0` (zero), the Neumann data `u_s(S) = e^{−r_f t}` and the Dirichlet
//! data `u(s, V) = (s − s_0) e^{−r_f t}` all scale with `e^{−r_f t}`, so every
//! boundary vector is stored once and multiplied by that factor per step.

use super::grid::backward_first;
use super::grid::central_first;
use super::grid::central_second;
use super::grid::forward_first;

/// Model and market inputs the operator is assembled from.
#[derive(Clone, Copy, Debug)]
pub(super) struct HestonCoefficients {
  pub kappa: f64,
  pub eta: f64,
  pub sigma: f64,
  pub rho: f64,
  pub r_d: f64,
  pub r_f: f64,
}

/// One row of a direction operator: three coefficients on the neighbours
/// at the given offsets along that direction.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Stencil {
  pub offsets: [isize; 3],
  pub coefficients: [f64; 3],
  /// Optional fourth coupling `(offset, coefficient)` used by the upwind rows.
  pub extra: Option<(isize, f64)>,
}

/// Assembled operator on the mesh.
#[derive(Clone, Debug)]
pub(super) struct Operators {
  pub m1: usize,
  pub m2: usize,
  pub s: Vec<f64>,
  pub v: Vec<f64>,
  pub r_f: f64,
  /// `A1` rows: stencil along `i` (offsets in `i`).
  a1: Vec<Stencil>,
  /// `A2` rows: stencil along `j` (offsets in `j`).
  a2: Vec<Stencil>,
  /// `A0` rows: the nine mixed-derivative weights on `(i+k, j+l)`, `k, l ∈ {−1, 0, 1}`, row-major in `k`.
  a0: Vec<[f64; 9]>,
  /// Boundary vectors per split part, to be scaled by `e^{−r_f t}`.
  b0: Vec<f64>,
  b1: Vec<f64>,
  b2: Vec<f64>,
}

impl Operators {
  /// Assembles the split operator on the meshes `s` (`s_0` = lower
  /// boundary, `s_{m1}` = `S`) and `v` (`v_0 = 0`, `v_{m2} = V`).
  pub fn new(s: Vec<f64>, v: Vec<f64>, c: HestonCoefficients) -> Self {
    let m1 = s.len() - 1;
    let m2 = v.len() - 1;
    let n = m1 * m2;
    let mut a1 = vec![Stencil::default(); n];
    let mut a2 = vec![Stencil::default(); n];
    let mut a0 = vec![[0.0; 9]; n];
    let (mut b0, mut b1, mut b2) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let lower = s[0];
    let top = |i: usize| s[i] - lower;
    let ds = |i: usize| s[i] - s[i - 1];
    let dv = |j: usize| v[j] - v[j - 1];
    let half_r = 0.5 * c.r_d;
    for j in 0..m2 {
      let vj = v[j];
      for i in 1..=m1 {
        let row = j * m1 + (i - 1);
        let si = s[i];
        // A1: ½ s² v u_ss + (r_d − r_f) s u_s − ½ r_d u along i.
        let convection = (c.r_d - c.r_f) * si;
        let diffusion = 0.5 * si * si * vj;
        if i < m1 {
          let (d_m, d_0, d_p) = central_second(ds(i), ds(i + 1));
          let (b_m, b_0, b_p) = central_first(ds(i), ds(i + 1));
          a1[row] = Stencil {
            offsets: [-1, 0, 1],
            coefficients: [
              diffusion * d_m + convection * b_m,
              diffusion * d_0 + convection * b_0 - half_r,
              diffusion * d_p + convection * b_p,
            ],
            extra: None,
          };
        } else {
          // s = S: u_s given by (2.4), u_ss by (2.10) with the virtual point
          // S + Δs_{m1} extrapolated from (2.4); the mixed term vanishes.
          let h = ds(m1);
          let (d_m, d_0, d_p) = central_second(h, h);
          a1[row] = Stencil {
            offsets: [-1, 0, 0],
            coefficients: [diffusion * d_m, diffusion * (d_0 + d_p) - half_r, 0.0],
            extra: None,
          };
          b1[row] += diffusion * d_p * h + convection;
        }
        // A2: ½ σ² v u_vv + κ(η − v) u_v − ½ r_d u along j.
        let drift = c.kappa * (c.eta - vj);
        let vol_diffusion = 0.5 * c.sigma * c.sigma * vj;
        if j == 0 {
          // v = 0: outflow boundary, forward stencil for u_v, no diffusion.
          let (g0, g1, g2) = forward_first(dv(1), dv(2));
          a2[row] = Stencil {
            offsets: [0, 1, 2],
            coefficients: [drift * g0 - half_r, drift * g1, drift * g2],
            extra: None,
          };
        } else {
          let (d_m, d_0, d_p) = central_second(dv(j), dv(j + 1));
          let use_upwind = vj > 1.0 && drift < 0.0 && j >= 2;
          let (f_m2, f_m, f_0, f_p) = if use_upwind {
            let (a_m2, a_m1, a_0) = backward_first(dv(j - 1), dv(j));
            (a_m2, a_m1, a_0, 0.0)
          } else {
            let (b_m, b_0, b_p) = central_first(dv(j), dv(j + 1));
            (0.0, b_m, b_0, b_p)
          };
          if use_upwind {
            a2[row] = Stencil {
              offsets: [-2, -1, 0],
              coefficients: [
                drift * f_m2,
                vol_diffusion * d_m + drift * f_m,
                vol_diffusion * d_0 + drift * f_0 - half_r,
              ],
              extra: None,
            };
            // The second-derivative coupling to j + 1 is kept explicit in the
            // boundary vector only when j + 1 is the Dirichlet boundary; the
            // upwind rows sit strictly inside (v_j > 1 < V), where j + 1 < m2
            // holds for every practical mesh — that coupling is folded into
            // the `a0`-free diagonal part below through `extra_upper`.
            extra_upper(&mut a2[row], vol_diffusion * d_p);
          } else {
            a2[row] = Stencil {
              offsets: [-1, 0, 1],
              coefficients: [
                vol_diffusion * d_m + drift * f_m,
                vol_diffusion * d_0 + drift * f_0 - half_r,
                vol_diffusion * d_p + drift * f_p,
              ],
              extra: None,
            };
          }
          if j + 1 == m2 {
            // Neighbour at v = V is the Dirichlet value (s − s_0) e^{−r_f t},
            // whether it sits in the three-slot stencil or in the extra slot.
            let upper = a2[row].coefficient_at(1);
            a2[row].zero_at(1);
            b2[row] += upper * top(i);
            if let Some((1, coef)) = a2[row].extra {
              b2[row] += coef * top(i);
              a2[row].extra = None;
            }
          }
        }
        // A0: ρ σ s v u_sv on interior rows (vanishes at s = S and at v = 0).
        if i < m1 && j >= 1 {
          let scale = c.rho * c.sigma * si * vj;
          let (bs_m, bs_0, bs_p) = central_first(ds(i), ds(i + 1));
          let (bv_m, bv_0, bv_p) = central_first(dv(j), dv(j + 1));
          let bs = [bs_m, bs_0, bs_p];
          let bv = [bv_m, bv_0, bv_p];
          for (k, &ws) in bs.iter().enumerate() {
            for (l, &wv) in bv.iter().enumerate() {
              let weight = scale * ws * wv;
              let ii = i as isize + k as isize - 1;
              let jj = j as isize + l as isize - 1;
              if ii == 0 {
                continue; // s = s_0 boundary value is zero
              }
              if jj == m2 as isize {
                b0[row] += weight * top(ii as usize);
                continue;
              }
              a0[row][k * 3 + l] = weight;
            }
          }
        }
      }
    }
    Self {
      m1,
      m2,
      s,
      v,
      r_f: c.r_f,
      a1,
      a2,
      a0,
      b0,
      b1,
      b2,
    }
  }

  pub fn len(&self) -> usize {
    self.m1 * self.m2
  }

  fn index(&self, i: usize, j: usize) -> usize {
    j * self.m1 + (i - 1)
  }

  /// `F_0(t, u) = A0 u + b0(t)`.
  pub fn f0(&self, t: f64, u: &[f64]) -> Vec<f64> {
    let scale = (-self.r_f * t).exp();
    let mut out = vec![0.0; u.len()];
    for j in 1..self.m2 {
      for i in 1..self.m1 {
        let row = self.index(i, j);
        let w = &self.a0[row];
        let mut acc = self.b0[row] * scale;
        for k in 0..3 {
          let ii = i + k - 1;
          if ii == 0 {
            continue;
          }
          for l in 0..3 {
            let jj = j + l - 1;
            if jj >= self.m2 {
              continue;
            }
            acc += w[k * 3 + l] * u[self.index(ii, jj)];
          }
        }
        out[row] = acc;
      }
    }
    out
  }

  /// `F_1(t, u) = A1 u + b1(t)`.
  pub fn f1(&self, t: f64, u: &[f64]) -> Vec<f64> {
    let scale = (-self.r_f * t).exp();
    let mut out = vec![0.0; u.len()];
    for j in 0..self.m2 {
      for i in 1..=self.m1 {
        let row = self.index(i, j);
        let st = &self.a1[row];
        let mut acc = self.b1[row] * scale;
        for (off, coef) in st.offsets.iter().zip(&st.coefficients) {
          if *coef == 0.0 {
            continue;
          }
          let ii = (i as isize + off) as usize;
          if ii >= 1 && ii <= self.m1 {
            acc += coef * u[self.index(ii, j)];
          }
        }
        out[row] = acc;
      }
    }
    out
  }

  /// `F_2(t, u) = A2 u + b2(t)`.
  pub fn f2(&self, t: f64, u: &[f64]) -> Vec<f64> {
    let scale = (-self.r_f * t).exp();
    let mut out = vec![0.0; u.len()];
    for j in 0..self.m2 {
      for i in 1..=self.m1 {
        let row = self.index(i, j);
        let st = &self.a2[row];
        let mut acc = self.b2[row] * scale;
        for (off, coef) in st.offsets.iter().zip(&st.coefficients) {
          if *coef == 0.0 {
            continue;
          }
          let jj = j as isize + off;
          if jj >= 0 && (jj as usize) < self.m2 {
            acc += coef * u[self.index(i, jj as usize)];
          }
        }
        if let Some((off, coef)) = st.extra {
          let jj = j as isize + off;
          if jj >= 0 && (jj as usize) < self.m2 {
            acc += coef * u[self.index(i, jj as usize)];
          }
        }
        out[row] = acc;
      }
    }
    out
  }

  /// Boundary vectors of the split parts at time `t`.
  pub fn b1_at(&self, t: f64) -> Vec<f64> {
    let scale = (-self.r_f * t).exp();
    self.b1.iter().map(|b| b * scale).collect()
  }

  pub fn b2_at(&self, t: f64) -> Vec<f64> {
    let scale = (-self.r_f * t).exp();
    self.b2.iter().map(|b| b * scale).collect()
  }

  /// `F(t, u) = F_0 + F_1 + F_2`.
  pub fn f(&self, t: f64, u: &[f64]) -> Vec<f64> {
    let (f0, f1, f2) = (self.f0(t, u), self.f1(t, u), self.f2(t, u));
    f0.iter()
      .zip(&f1)
      .zip(&f2)
      .map(|((a, b), c)| a + b + c)
      .collect()
  }

  /// Solves `(I − c A1) x = rhs` line by line along `i` (tridiagonal).
  pub fn solve_a1(&self, c: f64, rhs: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0; rhs.len()];
    let n = self.m1;
    let (mut lo, mut di, mut up, mut d) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for j in 0..self.m2 {
      for i in 1..=n {
        let row = self.index(i, j);
        let st = &self.a1[row];
        let (mut l, mut m, mut u_) = (0.0, 1.0, 0.0);
        for (off, coef) in st.offsets.iter().zip(&st.coefficients) {
          match off {
            -1 => l -= c * coef,
            0 => m -= c * coef,
            1 => u_ -= c * coef,
            _ => unreachable!("A1 is tridiagonal"),
          }
        }
        lo[i - 1] = l;
        di[i - 1] = m;
        up[i - 1] = u_;
        d[i - 1] = rhs[row];
      }
      let sol = solve_banded(&lo, &di, &up, None, None, &d);
      for i in 1..=n {
        x[self.index(i, j)] = sol[i - 1];
      }
    }
    x
  }

  /// Solves `(I − c A2) x = rhs` line by line along `j` (bandwidth two).
  pub fn solve_a2(&self, c: f64, rhs: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0; rhs.len()];
    let n = self.m2;
    let (mut lo2, mut lo, mut di, mut up, mut up2, mut d) = (
      vec![0.0; n],
      vec![0.0; n],
      vec![0.0; n],
      vec![0.0; n],
      vec![0.0; n],
      vec![0.0; n],
    );
    for i in 1..=self.m1 {
      for j in 0..n {
        let row = self.index(i, j);
        let st = &self.a2[row];
        let mut band = [0.0_f64; 5];
        band[2] = 1.0;
        let mut put = |off: isize, coef: f64| {
          let k = (off + 2) as usize;
          band[k] -= c * coef;
        };
        for (off, coef) in st.offsets.iter().zip(&st.coefficients) {
          put(*off, *coef);
        }
        if let Some((off, coef)) = st.extra {
          put(off, coef);
        }
        lo2[j] = band[0];
        lo[j] = band[1];
        di[j] = band[2];
        up[j] = band[3];
        up2[j] = band[4];
        d[j] = rhs[row];
      }
      let sol = solve_banded(&lo, &di, &up, Some(&lo2), Some(&up2), &d);
      for j in 0..n {
        x[self.index(i, j)] = sol[j];
      }
    }
    x
  }

  /// Bilinear interpolation of a grid vector at `(s, v)`; `top_scale` is
  /// `e^{−r_f t}` for the Dirichlet row at `v = V`.
  pub fn interpolate(&self, u: &[f64], s: f64, v: f64, top_scale: f64) -> f64 {
    let value = |i: usize, j: usize| -> f64 {
      if i == 0 {
        0.0
      } else if j >= self.m2 {
        (self.s[i] - self.s[0]) * top_scale
      } else {
        u[self.index(i, j)]
      }
    };
    let i = self.s.partition_point(|&x| x <= s).clamp(1, self.m1);
    let j = self.v.partition_point(|&x| x <= v).clamp(1, self.m2);
    let (s0, s1) = (self.s[i - 1], self.s[i]);
    let (v0, v1) = (self.v[j - 1], self.v[j]);
    let ws = ((s - s0) / (s1 - s0)).clamp(0.0, 1.0);
    let wv = ((v - v0) / (v1 - v0)).clamp(0.0, 1.0);
    let f00 = value(i - 1, j - 1);
    let f10 = value(i, j - 1);
    let f01 = value(i - 1, j);
    let f11 = value(i, j);
    (1.0 - ws) * (1.0 - wv) * f00 + ws * (1.0 - wv) * f10 + (1.0 - ws) * wv * f01 + ws * wv * f11
  }
}

impl Stencil {
  fn coefficient_at(&self, offset: isize) -> f64 {
    self
      .offsets
      .iter()
      .zip(&self.coefficients)
      .find(|(o, _)| **o == offset)
      .map_or(0.0, |(_, c)| *c)
  }

  fn zero_at(&mut self, offset: isize) {
    for (o, c) in self.offsets.iter().zip(self.coefficients.iter_mut()) {
      if *o == offset {
        *c = 0.0;
      }
    }
  }
}

/// Fourth coupling of an upwind `A2` row: the central second derivative's
/// `j + 1` weight, which the three-slot stencil has no room for.
fn extra_upper(stencil: &mut Stencil, coefficient: f64) {
  stencil.extra = Some((1, coefficient));
}

/// Gaussian elimination without pivoting on a banded system with lower and
/// upper bandwidth two (the outer bands optional); the `A1`/`A2` systems are
/// diagonally dominant for the step sizes of interest.
pub(super) fn solve_banded(
  lo: &[f64],
  di: &[f64],
  up: &[f64],
  lo2: Option<&[f64]>,
  up2: Option<&[f64]>,
  rhs: &[f64],
) -> Vec<f64> {
  let n = rhs.len();
  // Dense band storage: columns j−2..j+2 per row.
  let mut band: Vec<[f64; 5]> = (0..n)
    .map(|j| {
      [
        lo2.map_or(0.0, |b| b[j]),
        lo[j],
        di[j],
        up[j],
        up2.map_or(0.0, |b| b[j]),
      ]
    })
    .collect();
  let mut b = rhs.to_vec();
  for k in 0..n {
    let pivot = band[k][2];
    assert!(pivot != 0.0, "banded solve hit a zero pivot");
    for r in 1..=2 {
      let row = k + r;
      if row >= n {
        break;
      }
      let factor = band[row][2 - r] / pivot;
      if factor == 0.0 {
        continue;
      }
      for c in 0..=2 {
        let col_k = 2 + c; // column k + c in row k
        let col_row = 2 - r + c; // same column in row `row`
        if col_k < 5 && col_row < 5 {
          band[row][col_row] -= factor * band[k][col_k];
        }
      }
      band[row][2 - r] = 0.0;
      b[row] -= factor * b[k];
    }
  }
  let mut x = vec![0.0; n];
  for k in (0..n).rev() {
    let mut acc = b[k];
    for c in 1..=2 {
      if k + c < n {
        acc -= band[k][2 + c] * x[k + c];
      }
    }
    x[k] = acc / band[k][2];
  }
  x
}
