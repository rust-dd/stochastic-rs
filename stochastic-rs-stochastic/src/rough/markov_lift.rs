//! # Markov-lift Volterra SDE stepper
//!
//! $$
//! \begin{aligned} X_{n+1} &= X_0 + \frac{\delta t^{H+1/2} f(X_n)}{\Gamma(H+3/2)} + \frac{\sum_{l=1}^{N'} w_l\, e^{-x_l \delta t}\,(H_l^{(n)} + J_l^{(n)})}{\Gamma(H+1/2)} + \frac{\delta t^{H-1/2} g(X_n)\,\delta W_n}{\Gamma(H+1/2)} \\ H_l^{(n+1)} &= \tfrac{f(X_n)}{x_l}\bigl(1 - e^{-x_l \delta t}\bigr) + e^{-x_l \delta t}\, H_l^{(n)} \\ J_l^{(n+1)} &= e^{-x_l \delta t}\bigl(g(X_n)\,\delta W_n + J_l^{(n)}\bigr) \end{aligned}
//! $$
//!
//! A Volterra-SDE simulator collapsing the full history into a bounded
//! state of $N' \approx \log n$ exponential factors. Provides two entry
//! points:
//!
//! - [`simulate`](MarkovLift::simulate) — single path, SIMD across the
//!   $N'$ quadrature factors via `wide::f64x4` (f64) / `wide::f32x8` (f32).
//! - [`simulate_batch`](MarkovLift::simulate_batch) — $m$ paths in one
//!   pass, SIMD across the *path* axis at each factor $l$ (BLAS-style
//!   batch parallelism, matches the layout of the Python reference
//!   `RoughHestonFast`).
//!
//! Load/store goes through unsafe pointer casts to avoid the element-wise
//! scatter-gather overhead of `f64x4::from([a, b, c, d])`. Reference:
//! Bilokon & Wong (2026), p. 16 of J. Appl. Probab. 2026.
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray::Axis;
use ndarray::parallel::prelude::*;
use wide::f32x8;
use wide::f64x4;

use super::kernel::RlKernel;
use crate::traits::FloatExt;

/// Per-scalar SIMD kernel for the inner factor / path loops. Implemented for
/// `f64` with `f64x4` (4-wide) and `f32` with `f32x8` (8-wide).
pub trait RoughSimd: FloatExt {
  /// Single-path factor reduction: $\sum_l (w_l e_l)\,(H_l + J_l)$ with
  /// `we[l] = w_l * e_l` pre-merged.
  fn history_sum_fused(we: &[Self], h_state: &[Self], j_state: &[Self]) -> Self;

  /// Single-path fused state update using pre-computed
  /// $\mathrm{omx}_l = (1 - e_l)/x_l$.
  fn update_state_fused(
    h_state: &mut [Self],
    j_state: &mut [Self],
    exp_neg: &[Self],
    omx: &[Self],
    f_prev: Self,
    g_dw: Self,
  );

  /// Batch path reduction for a single factor $l$:
  /// `history[p] += we_l * (h_row[p] + j_row[p])` for all paths $p$.
  fn batch_history_accumulate(we_l: Self, h_row: &[Self], j_row: &[Self], history: &mut [Self]);

  /// Batch state update for a single factor $l$:
  /// - `h_row[p] = e_l * h_row[p] + f_prev[p] * omx_l`
  /// - `j_row[p] = e_l * (g_dw[p] + j_row[p])`
  fn batch_update_state(
    e_l: Self,
    omx_l: Self,
    h_row: &mut [Self],
    j_row: &mut [Self],
    f_prev: &[Self],
    g_dw: &[Self],
  );
}

impl RoughSimd for f64 {
  #[inline]
  fn history_sum_fused(we: &[f64], h_state: &[f64], j_state: &[f64]) -> f64 {
    let n = we.len();
    let chunks = n / 4;
    let mut acc = f64x4::splat(0.0);
    unsafe {
      for i in 0..chunks {
        let base = 4 * i;
        let we_v = load_f64x4(we, base);
        let h_v = load_f64x4(h_state, base);
        let j_v = load_f64x4(j_state, base);
        acc = we_v.mul_add(h_v + j_v, acc);
      }
    }
    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..n {
      sum += we[i] * (h_state[i] + j_state[i]);
    }
    sum
  }

  #[inline]
  fn update_state_fused(
    h_state: &mut [f64],
    j_state: &mut [f64],
    exp_neg: &[f64],
    omx: &[f64],
    f_prev: f64,
    g_dw: f64,
  ) {
    let n = h_state.len();
    let chunks = n / 4;
    let f_v = f64x4::splat(f_prev);
    let gdw_v = f64x4::splat(g_dw);
    unsafe {
      for i in 0..chunks {
        let base = 4 * i;
        let e = load_f64x4(exp_neg, base);
        let o = load_f64x4(omx, base);
        let h = load_f64x4(h_state, base);
        let j = load_f64x4(j_state, base);
        let h_new = e.mul_add(h, f_v * o);
        let j_new = e * (gdw_v + j);
        store_f64x4(h_state, base, h_new);
        store_f64x4(j_state, base, j_new);
      }
    }
    for i in (chunks * 4)..n {
      let e = exp_neg[i];
      let o = omx[i];
      h_state[i] = f_prev * o + e * h_state[i];
      j_state[i] = e * (g_dw + j_state[i]);
    }
  }

  #[inline]
  fn batch_history_accumulate(we_l: f64, h_row: &[f64], j_row: &[f64], history: &mut [f64]) {
    let m = history.len();
    let chunks = m / 4;
    let we_v = f64x4::splat(we_l);
    unsafe {
      for i in 0..chunks {
        let base = 4 * i;
        let h = load_f64x4(h_row, base);
        let j = load_f64x4(j_row, base);
        let hist = load_f64x4(history, base);
        let new_hist = we_v.mul_add(h + j, hist);
        store_f64x4(history, base, new_hist);
      }
    }
    for i in (chunks * 4)..m {
      history[i] += we_l * (h_row[i] + j_row[i]);
    }
  }

  #[inline]
  fn batch_update_state(
    e_l: f64,
    omx_l: f64,
    h_row: &mut [f64],
    j_row: &mut [f64],
    f_prev: &[f64],
    g_dw: &[f64],
  ) {
    let m = h_row.len();
    let chunks = m / 4;
    let e_v = f64x4::splat(e_l);
    let omx_v = f64x4::splat(omx_l);
    unsafe {
      for i in 0..chunks {
        let base = 4 * i;
        let h = load_f64x4(h_row, base);
        let j = load_f64x4(j_row, base);
        let fp = load_f64x4(f_prev, base);
        let gdw = load_f64x4(g_dw, base);
        let h_new = e_v.mul_add(h, fp * omx_v);
        let j_new = e_v * (gdw + j);
        store_f64x4(h_row, base, h_new);
        store_f64x4(j_row, base, j_new);
      }
    }
    for i in (chunks * 4)..m {
      h_row[i] = e_l * h_row[i] + f_prev[i] * omx_l;
      j_row[i] = e_l * (g_dw[i] + j_row[i]);
    }
  }
}

impl RoughSimd for f32 {
  #[inline]
  fn history_sum_fused(we: &[f32], h_state: &[f32], j_state: &[f32]) -> f32 {
    let n = we.len();
    let chunks = n / 8;
    let mut acc = f32x8::splat(0.0);
    unsafe {
      for i in 0..chunks {
        let base = 8 * i;
        let we_v = load_f32x8(we, base);
        let h_v = load_f32x8(h_state, base);
        let j_v = load_f32x8(j_state, base);
        acc = we_v.mul_add(h_v + j_v, acc);
      }
    }
    let mut sum = acc.reduce_add();
    for i in (chunks * 8)..n {
      sum += we[i] * (h_state[i] + j_state[i]);
    }
    sum
  }

  #[inline]
  fn update_state_fused(
    h_state: &mut [f32],
    j_state: &mut [f32],
    exp_neg: &[f32],
    omx: &[f32],
    f_prev: f32,
    g_dw: f32,
  ) {
    let n = h_state.len();
    let chunks = n / 8;
    let f_v = f32x8::splat(f_prev);
    let gdw_v = f32x8::splat(g_dw);
    unsafe {
      for i in 0..chunks {
        let base = 8 * i;
        let e = load_f32x8(exp_neg, base);
        let o = load_f32x8(omx, base);
        let h = load_f32x8(h_state, base);
        let j = load_f32x8(j_state, base);
        let h_new = e.mul_add(h, f_v * o);
        let j_new = e * (gdw_v + j);
        store_f32x8(h_state, base, h_new);
        store_f32x8(j_state, base, j_new);
      }
    }
    for i in (chunks * 8)..n {
      let e = exp_neg[i];
      let o = omx[i];
      h_state[i] = f_prev * o + e * h_state[i];
      j_state[i] = e * (g_dw + j_state[i]);
    }
  }

  #[inline]
  fn batch_history_accumulate(we_l: f32, h_row: &[f32], j_row: &[f32], history: &mut [f32]) {
    let m = history.len();
    let chunks = m / 8;
    let we_v = f32x8::splat(we_l);
    unsafe {
      for i in 0..chunks {
        let base = 8 * i;
        let h = load_f32x8(h_row, base);
        let j = load_f32x8(j_row, base);
        let hist = load_f32x8(history, base);
        let new_hist = we_v.mul_add(h + j, hist);
        store_f32x8(history, base, new_hist);
      }
    }
    for i in (chunks * 8)..m {
      history[i] += we_l * (h_row[i] + j_row[i]);
    }
  }

  #[inline]
  fn batch_update_state(
    e_l: f32,
    omx_l: f32,
    h_row: &mut [f32],
    j_row: &mut [f32],
    f_prev: &[f32],
    g_dw: &[f32],
  ) {
    let m = h_row.len();
    let chunks = m / 8;
    let e_v = f32x8::splat(e_l);
    let omx_v = f32x8::splat(omx_l);
    unsafe {
      for i in 0..chunks {
        let base = 8 * i;
        let h = load_f32x8(h_row, base);
        let j = load_f32x8(j_row, base);
        let fp = load_f32x8(f_prev, base);
        let gdw = load_f32x8(g_dw, base);
        let h_new = e_v.mul_add(h, fp * omx_v);
        let j_new = e_v * (gdw + j);
        store_f32x8(h_row, base, h_new);
        store_f32x8(j_row, base, j_new);
      }
    }
    for i in (chunks * 8)..m {
      h_row[i] = e_l * h_row[i] + f_prev[i] * omx_l;
      j_row[i] = e_l * (g_dw[i] + j_row[i]);
    }
  }
}

#[inline(always)]
unsafe fn load_f64x4(src: &[f64], base: usize) -> f64x4 {
  debug_assert!(base + 4 <= src.len());
  let ptr = unsafe { src.as_ptr().add(base) as *const [f64; 4] };
  f64x4::from(unsafe { *ptr })
}

#[inline(always)]
unsafe fn store_f64x4(dst: &mut [f64], base: usize, v: f64x4) {
  debug_assert!(base + 4 <= dst.len());
  let ptr = unsafe { dst.as_mut_ptr().add(base) as *mut [f64; 4] };
  unsafe { *ptr = v.to_array() };
}

#[inline(always)]
unsafe fn load_f32x8(src: &[f32], base: usize) -> f32x8 {
  debug_assert!(base + 8 <= src.len());
  let ptr = unsafe { src.as_ptr().add(base) as *const [f32; 8] };
  f32x8::from(unsafe { *ptr })
}

#[inline(always)]
unsafe fn store_f32x8(dst: &mut [f32], base: usize, v: f32x8) {
  debug_assert!(base + 8 <= dst.len());
  let ptr = unsafe { dst.as_mut_ptr().add(base) as *mut [f32; 8] };
  unsafe { *ptr = v.to_array() };
}

/// Single-path and batch Markov-lift stepper for $f,g$-driven RL-Volterra SDEs.
#[derive(Debug, Clone)]
pub struct MarkovLift<T: FloatExt> {
  /// Kernel approximation (nodes + scaled weights).
  pub kernel: RlKernel<T>,
  /// Time-step size $\delta t$.
  pub dt: T,
  /// $e^{-x_l \delta t}$ for each node.
  pub(crate) exp_neg_x_dt: Array1<T>,
  /// Pre-merged $w_l\,e^{-x_l \delta t}$ (constant across steps).
  pub(crate) we: Array1<T>,
  /// Pre-computed $(1 - e^{-x_l \delta t})/x_l$ for the $H_l$ update.
  pub(crate) one_minus_e_over_x: Array1<T>,
  /// $\delta t^{H+1/2}$ (boundary drift coefficient numerator).
  pub(crate) dt_pow_h_plus_half: T,
  /// $\delta t^{H-1/2}$ (boundary diffusion coefficient numerator).
  pub(crate) dt_pow_h_minus_half: T,
  /// $\Gamma(H + 3/2) = (H+1/2)\,\Gamma(H+1/2)$.
  pub(crate) gamma_h_plus_three_half: T,
}

impl<T: FloatExt> MarkovLift<T> {
  /// Build a stepper for the given kernel and step size $\delta t > 0$.
  #[must_use]
  pub fn new(kernel: RlKernel<T>, dt: T) -> Self {
    assert!(dt > T::zero(), "dt must be positive");

    let h = kernel.hurst;
    let half = T::from_f64_fast(0.5);
    let h_plus_half = h + half;

    let n_prime = kernel.degree();
    let mut exp_neg_x_dt = Array1::<T>::zeros(n_prime);
    let mut we = Array1::<T>::zeros(n_prime);
    let mut one_minus_e_over_x = Array1::<T>::zeros(n_prime);
    for i in 0..n_prime {
      let e = (-kernel.nodes[i] * dt).exp();
      exp_neg_x_dt[i] = e;
      we[i] = kernel.weights[i] * e;
      one_minus_e_over_x[i] = (T::one() - e) / kernel.nodes[i];
    }

    Self {
      dt_pow_h_plus_half: dt.powf(h_plus_half),
      dt_pow_h_minus_half: dt.powf(h - half),
      gamma_h_plus_three_half: kernel.gamma_h_half * h_plus_half,
      exp_neg_x_dt,
      we,
      one_minus_e_over_x,
      kernel,
      dt,
    }
  }
}

impl<T: FloatExt + RoughSimd> MarkovLift<T> {
  /// Integrate a single path. `dw` carries Brownian increments on the same
  /// grid as the output (length $n{-}1$).
  pub fn simulate<F, G>(&self, x0: T, f: F, g: G, dw: &[T]) -> Array1<T>
  where
    F: Fn(T) -> T,
    G: Fn(T) -> T,
  {
    let n = dw.len() + 1;
    let n_prime = self.kernel.degree();

    let mut path = Array1::<T>::zeros(n);
    path[0] = x0;

    let mut h_state = vec![T::zero(); n_prime];
    let mut j_state = vec![T::zero(); n_prime];

    let inv_gamma_h_half = T::one() / self.kernel.gamma_h_half;
    let inv_gamma_h_three_half = T::one() / self.gamma_h_plus_three_half;

    let we = self.we.as_slice().expect("we must be contiguous");
    let exp_neg = self
      .exp_neg_x_dt
      .as_slice()
      .expect("exp_neg must be contiguous");
    let omx = self
      .one_minus_e_over_x
      .as_slice()
      .expect("omx must be contiguous");

    for step in 0..n - 1 {
      let x_prev = path[step];
      let f_prev = f(x_prev);
      let g_prev = g(x_prev);
      let dw_n = dw[step];
      let g_dw = g_prev * dw_n;

      let history = T::history_sum_fused(we, &h_state, &j_state);

      path[step + 1] = x0
        + self.dt_pow_h_plus_half * f_prev * inv_gamma_h_three_half
        + history * inv_gamma_h_half
        + self.dt_pow_h_minus_half * g_prev * dw_n * inv_gamma_h_half;

      T::update_state_fused(&mut h_state, &mut j_state, exp_neg, omx, f_prev, g_dw);
    }

    path
  }

  /// Integrate $m$ independent paths driven by the given Brownian increment
  /// matrix `dw` of shape $(m, n{-}1)$. Returns an $(m, n)$ path matrix.
  ///
  /// Uses **cache-tiled path-SIMD**: the $m$ paths are processed in blocks
  /// of [`BATCH_TILE`] columns so that the state arrays
  /// $(N', \mathrm{tile})$ fit in L1 ($\lesssim 32$ KB). Each tile advances
  /// independently through all $n$ time steps; the outer tile loop can be
  /// parallelised across cores.
  ///
  /// Layout matches the Python reference `RoughHestonFast` (numpy
  /// `(p, N')` matmul) but with explicit cache blocking to avoid the memory
  /// bandwidth ceiling that a naive single large batch would hit.
  pub fn simulate_batch<F, G>(&self, x0: T, f: F, g: G, dw: ArrayView2<T>) -> Array2<T>
  where
    F: Fn(T) -> T,
    G: Fn(T) -> T,
  {
    let (m, n_minus_1) = dw.dim();
    let n = n_minus_1 + 1;

    let mut paths = Array2::<T>::zeros((m, n));
    for p in 0..m {
      paths[[p, 0]] = x0;
    }

    let mut tile_start = 0_usize;
    while tile_start < m {
      let tile_end = (tile_start + BATCH_TILE).min(m);
      self.simulate_tile(x0, &f, &g, dw, tile_start, tile_end, &mut paths);
      tile_start = tile_end;
    }
    paths
  }

  /// Same as [`simulate_batch`](Self::simulate_batch) but parallelises the
  /// outer tile loop with rayon — combines per-core SIMD path-batching with
  /// multi-core scheduling. Requires `f` and `g` to be `Send + Sync`.
  pub fn simulate_batch_par<F, G>(&self, x0: T, f: F, g: G, dw: ArrayView2<T>) -> Array2<T>
  where
    F: Fn(T) -> T + Send + Sync,
    G: Fn(T) -> T + Send + Sync,
  {
    let (m, _) = dw.dim();
    let n = dw.ncols() + 1;

    let mut paths = Array2::<T>::zeros((m, n));
    for p in 0..m {
      paths[[p, 0]] = x0;
    }

    paths
      .axis_chunks_iter_mut(Axis(0), BATCH_TILE)
      .into_par_iter()
      .enumerate()
      .for_each(|(tile_idx, mut chunk)| {
        let tile_start = tile_idx * BATCH_TILE;
        let tile_rows = chunk.nrows();
        let tile_end = tile_start + tile_rows;
        self.simulate_tile_into(x0, &f, &g, dw, tile_start, tile_end, chunk.view_mut());
      });

    paths
  }

  fn simulate_tile<F, G>(
    &self,
    x0: T,
    f: &F,
    g: &G,
    dw: ArrayView2<T>,
    tile_start: usize,
    tile_end: usize,
    paths: &mut Array2<T>,
  ) where
    F: Fn(T) -> T,
    G: Fn(T) -> T,
  {
    let view = paths.view_mut();
    self.simulate_tile_into(x0, f, g, dw, tile_start, tile_end, view);
  }

  fn simulate_tile_into<F, G>(
    &self,
    x0: T,
    f: &F,
    g: &G,
    dw: ArrayView2<T>,
    tile_start: usize,
    tile_end: usize,
    mut paths: ndarray::ArrayViewMut2<T>,
  ) where
    F: Fn(T) -> T,
    G: Fn(T) -> T,
  {
    let tile_size = tile_end - tile_start;
    let n = paths.ncols();
    let n_minus_1 = n - 1;
    let n_prime = self.kernel.degree();

    let mut dw_t = vec![T::zero(); n_minus_1 * tile_size];
    for (local_p, p) in (tile_start..tile_end).enumerate() {
      for s in 0..n_minus_1 {
        dw_t[s * tile_size + local_p] = dw[[p, s]];
      }
    }

    let mut h_state = vec![T::zero(); n_prime * tile_size];
    let mut j_state = vec![T::zero(); n_prime * tile_size];
    let mut history = vec![T::zero(); tile_size];
    let mut f_prev = vec![T::zero(); tile_size];
    let mut g_dw = vec![T::zero(); tile_size];
    let mut current_x = vec![x0; tile_size];
    let mut next_x = vec![T::zero(); tile_size];

    let inv_gamma_h_half = T::one() / self.kernel.gamma_h_half;
    let inv_gamma_h_three_half = T::one() / self.gamma_h_plus_three_half;
    let we = self.we.as_slice().expect("we contiguous");
    let exp_neg = self.exp_neg_x_dt.as_slice().expect("exp_neg contiguous");
    let omx = self.one_minus_e_over_x.as_slice().expect("omx contiguous");

    let k_drift = self.dt_pow_h_plus_half * inv_gamma_h_three_half;
    let k_hist = inv_gamma_h_half;
    let k_diff = self.dt_pow_h_minus_half * inv_gamma_h_half;

    for step in 0..n_minus_1 {
      let dw_row = &dw_t[step * tile_size..(step + 1) * tile_size];

      for local_p in 0..tile_size {
        let xp = current_x[local_p];
        f_prev[local_p] = f(xp);
        g_dw[local_p] = g(xp) * dw_row[local_p];
      }

      for h in history.iter_mut() {
        *h = T::zero();
      }
      for l in 0..n_prime {
        let h_row = &h_state[l * tile_size..(l + 1) * tile_size];
        let j_row = &j_state[l * tile_size..(l + 1) * tile_size];
        T::batch_history_accumulate(we[l], h_row, j_row, &mut history);
      }

      for local_p in 0..tile_size {
        next_x[local_p] =
          x0 + k_drift * f_prev[local_p] + k_hist * history[local_p] + k_diff * g_dw[local_p];
      }

      let write_rows = tile_end - tile_start;
      for local_p in 0..write_rows {
        let global_p = if paths.nrows() == write_rows {
          local_p
        } else {
          tile_start + local_p
        };
        paths[[global_p, step + 1]] = next_x[local_p];
      }

      current_x.copy_from_slice(&next_x);

      for l in 0..n_prime {
        let h_row = &mut h_state[l * tile_size..(l + 1) * tile_size];
        let j_row = &mut j_state[l * tile_size..(l + 1) * tile_size];
        T::batch_update_state(exp_neg[l], omx[l], h_row, j_row, &f_prev, &g_dw);
      }
    }
  }
}

/// Path block size for [`MarkovLift::simulate_batch`]. Chosen so the
/// $(N', \mathrm{tile})$ state arrays fit in a 32 KB L1 cache for
/// $N' \lesssim 30$ (tile × 30 × 2 × 8 ≈ 31 KB at tile = 64).
pub const BATCH_TILE: usize = 64;

#[cfg(test)]
mod tests {
  use ndarray::Array2;

  use super::super::kernel::RlKernel;
  use super::MarkovLift;

  #[test]
  fn trivial_drift_zero_diffusion_stays_at_x0() {
    let kernel = RlKernel::<f64>::new(0.15, 30);
    let dt = 0.01_f64;
    let step = MarkovLift::new(kernel, dt);
    let dw = vec![0.0_f64; 50];
    let path = step.simulate(0.42, |_| 0.0, |_| 0.0, &dw);
    for v in path.iter() {
      assert!((*v - 0.42).abs() < 1e-12);
    }
  }

  #[test]
  fn constant_drift_matches_mittag_leffler_linear_case() {
    let hurst = 0.3_f64;
    let c = 1.5_f64;
    let n = 201;
    let total_t = 1.0_f64;
    let dt = total_t / (n as f64 - 1.0);
    let kernel = RlKernel::<f64>::new(hurst, 40);
    let step = MarkovLift::new(kernel, dt);
    let dw = vec![0.0_f64; n - 1];

    let path = step.simulate(0.0, |_| c, |_| 0.0, &dw);

    let exponent = hurst + 0.5;
    let gamma_h_three_half = statrs::function::gamma::gamma(hurst + 1.5);
    for i in 1..n {
      let t = dt * i as f64;
      let truth = c * t.powf(exponent) / gamma_h_three_half;
      let rel = (path[i] - truth).abs() / truth.abs().max(1e-9);
      assert!(
        rel < 2e-2,
        "i={i} t={t} got={} truth={truth} rel={rel}",
        path[i]
      );
    }
  }

  #[test]
  fn f32_path_is_finite() {
    let kernel = RlKernel::<f32>::new(0.25_f32, 32);
    let dt = 0.005_f32;
    let step = MarkovLift::new(kernel, dt);
    let dw: Vec<f32> = (0..100).map(|i| ((i as f32 * 0.1).sin()) * 0.05).collect();
    let path = step.simulate(0.3_f32, |x| 0.5 * (1.0 - x), |_| 0.2, &dw);
    assert!(path.iter().all(|v| v.is_finite()));
  }

  /// The batch simulator must produce the same path as repeated single-path
  /// runs with the matching per-row increments.
  #[test]
  fn batch_matches_single_path_row_by_row() {
    let hurst = 0.22_f64;
    let n = 65;
    let m = 7;
    let dt = 1.0_f64 / (n as f64 - 1.0);
    let kernel = RlKernel::<f64>::new(hurst, 30);
    let step = MarkovLift::new(kernel, dt);

    let mut dw = Array2::<f64>::zeros((m, n - 1));
    for p in 0..m {
      for i in 0..n - 1 {
        dw[[p, i]] = ((p as f64 + 1.0) * 0.13 + (i as f64) * 0.027).sin() * 0.02;
      }
    }

    let batch = step.simulate_batch(0.4, |x| 0.6 * (1.0 - x), |_| 0.15, dw.view());
    assert_eq!(batch.dim(), (m, n));

    for p in 0..m {
      let row = dw.row(p).to_vec();
      let single = step.simulate(0.4, |x| 0.6 * (1.0 - x), |_| 0.15, row.as_slice());
      for i in 0..n {
        let diff = (batch[[p, i]] - single[i]).abs();
        assert!(
          diff < 1e-12,
          "p={p} i={i} batch={} single={} diff={diff}",
          batch[[p, i]],
          single[i]
        );
      }
    }
  }
}
