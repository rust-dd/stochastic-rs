//! Discretisation schemes driven by pre-generated fractional Brownian increments.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::Array3;
use ndarray::Axis;
use ndarray::s;

use super::Sde;
use crate::device::Backend;
use crate::traits::FloatExt;

impl<T: FloatExt, F, G, B: Backend> Sde<T, F, G, B>
where
  F: Fn(&Array1<T>, T) -> Array1<T>,
  G: Fn(&Array1<T>, T) -> Array2<T>,
{
  /// Euler–Maruyama with fractional Brownian noise.
  pub(super) fn solve_euler_fractional(
    &self,
    x0: &Array1<T>,
    t0: T,
    dt: T,
    incs: &Array3<T>,
  ) -> Array3<T> {
    let (n_paths, steps, dim) = (
      incs.len_of(Axis(0)),
      incs.len_of(Axis(1)),
      incs.len_of(Axis(2)),
    );
    let mut out = Array3::zeros((n_paths, steps + 1, dim));
    for p in 0..n_paths {
      let mut x = x0.clone();
      let mut time = t0;
      out.slice_mut(s![p, 0, ..]).assign(&x);
      for i_step in 1..=steps {
        let d_w = incs.slice(s![p, i_step - 1, ..]);
        let mu_val = (self.drift)(&x, time);
        let sigma_val = (self.diffusion)(&x, time);
        for i_dim in 0..dim {
          let mut incr = mu_val[i_dim] * dt;
          for j_dim in 0..dim {
            incr += sigma_val[[i_dim, j_dim]] * d_w[j_dim];
          }
          x[i_dim] += incr;
        }
        time += dt;
        out.slice_mut(s![p, i_step, ..]).assign(&x);
      }
    }
    out
  }

  /// Milstein scheme with fractional Brownian noise.
  pub(super) fn solve_milstein_fractional(
    &self,
    x0: &Array1<T>,
    t0: T,
    dt: T,
    incs: &Array3<T>,
  ) -> Array3<T> {
    let (n_paths, steps, dim) = (
      incs.len_of(Axis(0)),
      incs.len_of(Axis(1)),
      incs.len_of(Axis(2)),
    );
    let mut out = Array3::zeros((n_paths, steps + 1, dim));
    for p in 0..n_paths {
      let mut x = x0.clone();
      let mut time = t0;
      out.slice_mut(s![p, 0, ..]).assign(&x);
      for i_step in 1..=steps {
        let d_w = incs.slice(s![p, i_step - 1, ..]);
        let mu_val = (self.drift)(&x, time);
        let sigma_val = (self.diffusion)(&x, time);
        let correction = self.milstein_correction(&x, time, d_w, dt);
        for i_dim in 0..dim {
          let mut incr = mu_val[i_dim] * dt;
          for j_dim in 0..dim {
            incr += sigma_val[[i_dim, j_dim]] * d_w[j_dim];
          }
          incr += correction[i_dim];
          x[i_dim] += incr;
        }
        time += dt;
        out.slice_mut(s![p, i_step, ..]).assign(&x);
      }
    }
    out
  }

  /// Stochastic midpoint (RK2-style) method with fractional Brownian noise.
  pub(super) fn solve_srk2_fractional(
    &self,
    x0: &Array1<T>,
    t0: T,
    dt: T,
    incs: &Array3<T>,
  ) -> Array3<T> {
    let (n_paths, steps, dim) = (
      incs.len_of(Axis(0)),
      incs.len_of(Axis(1)),
      incs.len_of(Axis(2)),
    );
    let half = T::from_f64_fast(0.5);
    let half_dt = half * dt;
    let mut out = Array3::zeros((n_paths, steps + 1, dim));
    for p in 0..n_paths {
      let mut x = x0.clone();
      let mut time = t0;
      out.slice_mut(s![p, 0, ..]).assign(&x);
      for i_step in 1..=steps {
        let d_w = incs.slice(s![p, i_step - 1, ..]);
        let mu1 = (self.drift)(&x, time);
        let sig1 = (self.diffusion)(&x, time);
        let mut x_half = x.clone();
        for i_dim in 0..dim {
          let mut incr = mu1[i_dim] * half_dt;
          for j_dim in 0..dim {
            incr += sig1[[i_dim, j_dim]] * (half * d_w[j_dim]);
          }
          x_half[i_dim] += incr;
        }
        let mu2 = (self.drift)(&x_half, time + half_dt);
        let sig2 = (self.diffusion)(&x_half, time + half_dt);
        for i_dim in 0..dim {
          let mut incr = mu2[i_dim] * dt;
          for j_dim in 0..dim {
            incr += sig2[[i_dim, j_dim]] * d_w[j_dim];
          }
          x[i_dim] += incr;
        }
        time += dt;
        out.slice_mut(s![p, i_step, ..]).assign(&x);
      }
    }
    out
  }

  /// Classical RK4 structure applied to both drift and diffusion, with fractional Brownian noise.
  pub(super) fn solve_srk4_fractional(
    &self,
    x0: &Array1<T>,
    t0: T,
    dt: T,
    incs: &Array3<T>,
  ) -> Array3<T> {
    let (n_paths, steps, dim) = (
      incs.len_of(Axis(0)),
      incs.len_of(Axis(1)),
      incs.len_of(Axis(2)),
    );
    let half = T::from_f64_fast(0.5);
    let two = T::from_f64_fast(2.0);
    let six = T::from_f64_fast(6.0);
    let half_dt = half * dt;
    let mut out = Array3::zeros((n_paths, steps + 1, dim));
    for p in 0..n_paths {
      let mut x = x0.clone();
      let mut time = t0;
      out.slice_mut(s![p, 0, ..]).assign(&x);
      for i_step in 1..=steps {
        let d_w_full = incs.slice(s![p, i_step - 1, ..]);
        let k1_mu = (self.drift)(&x, time);
        let k1_sig = (self.diffusion)(&x, time);
        let mut x1 = x.clone();
        for i_dim in 0..dim {
          let mut incr = k1_mu[i_dim] * half_dt;
          for j_dim in 0..dim {
            incr += k1_sig[[i_dim, j_dim]] * (d_w_full[j_dim] * half);
          }
          x1[i_dim] += incr;
        }
        let k2_mu = (self.drift)(&x1, time + half_dt);
        let k2_sig = (self.diffusion)(&x1, time + half_dt);
        let mut x2 = x.clone();
        for i_dim in 0..dim {
          let mut incr = k2_mu[i_dim] * half_dt;
          for j_dim in 0..dim {
            incr += k2_sig[[i_dim, j_dim]] * (d_w_full[j_dim] * half);
          }
          x2[i_dim] += incr;
        }
        let k3_mu = (self.drift)(&x2, time + half_dt);
        let k3_sig = (self.diffusion)(&x2, time + half_dt);
        let mut x3 = x.clone();
        for i_dim in 0..dim {
          let mut incr = k3_mu[i_dim] * dt;
          for j_dim in 0..dim {
            incr += k3_sig[[i_dim, j_dim]] * d_w_full[j_dim];
          }
          x3[i_dim] += incr;
        }
        let k4_mu = (self.drift)(&x3, time + dt);
        let k4_sig = (self.diffusion)(&x3, time + dt);
        for i_dim in 0..dim {
          let drift_avg =
            (k1_mu[i_dim] + two * k2_mu[i_dim] + two * k3_mu[i_dim] + k4_mu[i_dim]) / six;
          let mut incr = drift_avg * dt;
          for j_dim in 0..dim {
            let diff_ij = (k1_sig[[i_dim, j_dim]]
              + two * k2_sig[[i_dim, j_dim]]
              + two * k3_sig[[i_dim, j_dim]]
              + k4_sig[[i_dim, j_dim]])
              / six;
            incr += diff_ij * d_w_full[j_dim];
          }
          x[i_dim] += incr;
        }
        time += dt;
        out.slice_mut(s![p, i_step, ..]).assign(&x);
      }
    }
    out
  }
}
