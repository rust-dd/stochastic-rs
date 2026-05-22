use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray::Axis;
use ndarray::parallel::prelude::*;

use super::simd::RoughSimd;
use crate::rough::kernel::RlKernel;
use crate::traits::FloatExt;

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
