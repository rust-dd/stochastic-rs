//! The Markov lift of a Volterra equation: `N'` exponential factors, `O(nN')` in place of `O(n²)`.
//!
//! $$
//! \begin{aligned}
//! X_{n+1} &= X_0 + \Bigl(\textstyle\int_0^{\delta t} K(u)\,du\Bigr) f(t_n, X_n) +
//! \sum_{l=1}^{N'} w_l\, e^{-x_l \delta t}\,\bigl(H_l^{(n)} + J_l^{(n)}\bigr) +
//! K(\delta t)\, g(t_n, X_n)\,\delta W_n \\
//! H_l^{(n+1)} &= \tfrac{f(t_n,X_n)}{x_l}\bigl(1 - e^{-x_l \delta t}\bigr) +
//! e^{-x_l \delta t}\, H_l^{(n)} \\
//! J_l^{(n+1)} &= e^{-x_l \delta t}\bigl(g(t_n,X_n)\,\delta W_n + J_l^{(n)}\bigr)
//! \end{aligned}
//! $$
//!
//! For any [`VolterraKernel`], whose weights carry every normalising constant.
//!
//! Reference: Abi Jaber & El Euch (2018), arXiv:1801.10359
//! Reference: Bilokon & Wong (2026), J. Appl. Probab., doi:10.1017/jpr.2025.10071
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use ndarray::Axis;
use ndarray::parallel::prelude::*;

use crate::traits::FloatExt;
use crate::volterra::kernel::VolterraKernel;

/// Kernel-generic single-path and batch Markov-lift stepper for
/// $f,g$-driven Volterra SDEs.
#[derive(Debug, Clone)]
pub struct VolterraLift<T: FloatExt, K: VolterraKernel<T>> {
  /// Kernel approximation (nodes + trait-normalised weights).
  pub kernel: K,
  /// Time-step size $\delta t$.
  pub dt: T,
  /// $e^{-x_l \delta t}$ for each node.
  pub(crate) exp_neg_x_dt: Array1<T>,
  /// Pre-merged $w_l\, e^{-x_l \delta t}$, constant across steps, from [`VolterraKernel::weights`]
  /// alone: they carry every normalising constant, so no further factor is applied anywhere.
  pub(crate) we: Array1<T>,
  /// Pre-computed $(1 - e^{-x_l \delta t})/x_l$ for the $H_l$ update.
  pub(crate) one_minus_e_over_x: Array1<T>,
  /// Drift boundary weight $\int_0^{\delta t} K(u)\,du$ — see
  /// [`VolterraKernel::integral_from_zero`].
  pub(crate) drift_boundary: T,
  /// Diffusion boundary weight $K(\delta t)$ — see
  /// [`VolterraKernel::evaluate`].
  pub(crate) diffusion_boundary: T,
}

impl<T: FloatExt, K: VolterraKernel<T>> VolterraLift<T, K> {
  /// A stepper for `kernel` at step $\delta t$, built from [`VolterraKernel`]'s methods alone;
  /// panics unless `dt > 0`.
  #[must_use]
  pub fn new(kernel: K, dt: T) -> Self {
    assert!(
      dt > T::zero(),
      "dt must satisfy `dt > T::zero()`, got dt = {dt:?}"
    );

    let n_prime = kernel.degree();
    let nodes = kernel.nodes();
    let weights = kernel.weights();
    let mut exp_neg_x_dt = Array1::<T>::zeros(n_prime);
    let mut we = Array1::<T>::zeros(n_prime);
    let mut one_minus_e_over_x = Array1::<T>::zeros(n_prime);
    for l in 0..n_prime {
      let e = (-nodes[l] * dt).exp();
      exp_neg_x_dt[l] = e;
      we[l] = weights[l] * e;
      one_minus_e_over_x[l] = (T::one() - e) / nodes[l];
    }

    let drift_boundary = kernel.integral_from_zero(dt);
    let diffusion_boundary = kernel.evaluate(dt);

    Self {
      drift_boundary,
      diffusion_boundary,
      exp_neg_x_dt,
      we,
      one_minus_e_over_x,
      kernel,
      dt,
    }
  }

  /// Integrate one path from the Brownian increments `dw` (length $n{-}1$). `f` and `g` receive
  /// $(t_n, X_n)$, the current time and state, so coefficients may depend on time.
  pub fn simulate<F, G>(&self, x0: T, f: F, g: G, dw: &[T]) -> Array1<T>
  where
    F: Fn(T, T) -> T,
    G: Fn(T, T) -> T,
  {
    let n = dw.len() + 1;
    let n_prime = self.kernel.degree();

    let mut path = Array1::<T>::zeros(n);
    path[0] = x0;

    let mut h_state = vec![T::zero(); n_prime];
    let mut j_state = vec![T::zero(); n_prime];

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
      let t = T::from_usize_(step) * self.dt;
      let x_prev = path[step];
      let f_prev = f(t, x_prev);
      let g_prev = g(t, x_prev);
      let dw_n = dw[step];
      let g_dw = g_prev * dw_n;

      let history = T::history_sum_fused(we, &h_state, &j_state);

      path[step + 1] =
        x0 + self.drift_boundary * f_prev + history + self.diffusion_boundary * g_prev * dw_n;

      T::update_state_fused(&mut h_state, &mut j_state, exp_neg, omx, f_prev, g_dw);
    }

    path
  }

  /// The `(m, n)` paths the `(m, n − 1)` increments `dw` drive, stepped in tiles of 64 paths whose
  /// lift states stay in L1; `simulate_batch_par` runs the tiles in parallel.
  pub fn simulate_batch<F, G>(&self, x0: T, f: F, g: G, dw: ArrayView2<T>) -> Array2<T>
  where
    F: Fn(T, T) -> T,
    G: Fn(T, T) -> T,
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

  /// [`simulate_batch`](Self::simulate_batch) with the outer tile loop on rayon. `f`, `g` and
  /// `K` need only be `Sync`: every worker calls them through a shared reference.
  pub fn simulate_batch_par<F, G>(&self, x0: T, f: F, g: G, dw: ArrayView2<T>) -> Array2<T>
  where
    F: Fn(T, T) -> T + Sync,
    G: Fn(T, T) -> T + Sync,
    K: Sync,
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
    F: Fn(T, T) -> T,
    G: Fn(T, T) -> T,
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
    mut paths: ArrayViewMut2<T>,
  ) where
    F: Fn(T, T) -> T,
    G: Fn(T, T) -> T,
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

    let we = self.we.as_slice().expect("we contiguous");
    let exp_neg = self.exp_neg_x_dt.as_slice().expect("exp_neg contiguous");
    let omx = self.one_minus_e_over_x.as_slice().expect("omx contiguous");

    let k_drift = self.drift_boundary;
    let k_diff = self.diffusion_boundary;

    for step in 0..n_minus_1 {
      let dw_row = &dw_t[step * tile_size..(step + 1) * tile_size];
      let t = T::from_usize_(step) * self.dt;

      for local_p in 0..tile_size {
        let xp = current_x[local_p];
        f_prev[local_p] = f(t, xp);
        g_dw[local_p] = g(t, xp) * dw_row[local_p];
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
          x0 + k_drift * f_prev[local_p] + history[local_p] + k_diff * g_dw[local_p];
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

/// The batch tile: at 64 paths the `(N', tile)` states fit a 32 KB L1 for `N' ≲ 30`.
pub(crate) const BATCH_TILE: usize = 64;

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use ndarray::Array2;

  use super::VolterraLift;
  use crate::volterra::kernel::ExponentialKernel;
  use crate::volterra::kernel::SumOfExponentials;

  #[test]
  fn trivial_drift_zero_diffusion_stays_at_x0() {
    let kernel = ExponentialKernel::new(0.5_f64, 1.0_f64);
    let lift = VolterraLift::new(kernel, 0.02_f64);
    let dw = vec![0.0_f64; 40];
    let path = lift.simulate(0.37, |_t, _x| 0.0, |_t, _x| 0.0, &dw);
    for v in path.iter() {
      assert!((*v - 0.37).abs() < 1e-12);
    }
  }

  /// A pure exponential kernel with constant `f` and `g = 0` gives $f\,c\,(1 - e^{-\beta t})/\beta$
  /// with no fitting error, so the recursion must match to near machine precision.
  #[test]
  fn exponential_kernel_constant_drift_matches_closed_form() {
    let beta = 0.7_f64;
    let c = 1.3_f64;
    let f_const = 0.9_f64;
    let n = 401;
    let total_t = 2.0_f64;
    let dt = total_t / (n as f64 - 1.0);
    let kernel = ExponentialKernel::new(beta, c);
    let lift = VolterraLift::new(kernel, dt);
    let dw = vec![0.0_f64; n - 1];

    let path = lift.simulate(0.0, |_t, _x| f_const, |_t, _x| 0.0, &dw);

    for i in 0..n {
      let t = dt * i as f64;
      let truth = f_const * c * (1.0 - (-beta * t).exp()) / beta;
      let rel = (path[i] - truth).abs() / truth.abs().max(1e-12);
      assert!(
        rel < 1e-9,
        "i={i} t={t} got={} truth={truth} rel={rel}",
        path[i]
      );
    }
  }

  /// Proves `f`/`g` genuinely receive the current step's simulation time
  /// (`step * dt`), not a placeholder constant.
  #[test]
  fn coefficients_receive_current_step_time() {
    let kernel = ExponentialKernel::new(0.3_f64, 1.0_f64);
    let dt = 0.1_f64;
    let lift = VolterraLift::new(kernel, dt);
    let dw = vec![0.0_f64; 5];
    let seen = std::cell::RefCell::new(Vec::new());
    let _ = lift.simulate(
      0.0,
      |t, _x| {
        seen.borrow_mut().push(t);
        0.0
      },
      |_t, _x| 0.0,
      &dw,
    );
    let expected = (0..5).map(|s| s as f64 * dt).collect::<Vec<_>>();
    let got = seen.into_inner();
    assert_eq!(got.len(), expected.len());
    for (g, e) in got.iter().zip(expected.iter()) {
      assert!((g - e).abs() < 1e-12, "got={g} expected={e}");
    }
  }

  /// The batch simulator equals repeated single-path runs row by row, here through a kernel that
  /// is not `RlKernel`: the batch path is kernel-generic.
  #[test]
  fn batch_matches_single_path_row_by_row() {
    let nodes = Array1::from_vec(vec![0.4_f64, 1.1, 2.3]);
    let weights = Array1::from_vec(vec![0.6_f64, 0.3, 0.15]);
    let kernel = SumOfExponentials::new(nodes, weights);
    let n = 40;
    let m = 6;
    let dt = 1.0_f64 / (n as f64 - 1.0);
    let lift = VolterraLift::new(kernel, dt);

    let mut dw = Array2::<f64>::zeros((m, n - 1));
    for p in 0..m {
      for i in 0..n - 1 {
        dw[[p, i]] = ((p as f64 + 1.0) * 0.11 + (i as f64) * 0.031).sin() * 0.02;
      }
    }

    let f = |_t: f64, x: f64| 0.5 * (1.0 - x);
    let g = |_t: f64, _x: f64| 0.1;
    let batch = lift.simulate_batch(0.2, f, g, dw.view());
    for p in 0..m {
      let row = dw.row(p).to_vec();
      let single = lift.simulate(0.2, f, g, row.as_slice());
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

  /// `simulate_batch_par` reproduces `simulate_batch` exactly; `m = 70` spans two
  /// [`super::BATCH_TILE`] tiles (64), which exercises the tile boundary.
  #[test]
  fn batch_par_matches_batch_sequential_across_tile_boundary() {
    let nodes = Array1::from_vec(vec![0.5_f64, 1.7]);
    let weights = Array1::from_vec(vec![0.8_f64, 0.2]);
    let kernel = SumOfExponentials::new(nodes, weights);
    let n = 20;
    let m = 70;
    let dt = 1.0_f64 / (n as f64 - 1.0);
    let lift = VolterraLift::new(kernel, dt);

    let mut dw = Array2::<f64>::zeros((m, n - 1));
    for p in 0..m {
      for i in 0..n - 1 {
        dw[[p, i]] = ((p as f64 + 1.0) * 0.07 + (i as f64) * 0.013).cos() * 0.015;
      }
    }

    let f = |_t: f64, x: f64| 0.3 * (0.5 - x);
    let g = |_t: f64, _x: f64| 0.2;
    let seq = lift.simulate_batch(0.1, f, g, dw.view());
    let par = lift.simulate_batch_par(0.1, f, g, dw.view());
    assert_eq!(seq, par);
  }
}
