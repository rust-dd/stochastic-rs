//! Stochastic differential equation solver

use ndarray::s;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Array3;
use ndarray::ArrayView1;
use ndarray::Axis;
use rand::Rng;

use super::noise::fgn::FGN;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub enum NoiseModel {
  Gaussian,
  Fractional,
}

pub enum SdeMethod {
  Euler,
  Milstein,
  SRK2,
  SRK4,
}

pub struct Sde<F, G>
where
  F: Fn(&Array1<f64>, f64) -> Array1<f64>,
  G: Fn(&Array1<f64>, f64) -> Array2<f64>,
{
  pub drift: F,
  pub diffusion: G,
  pub noise: NoiseModel,
  pub hursts: Option<Array1<f64>>,
}

impl<F, G> Sde<F, G>
where
  F: Fn(&Array1<f64>, f64) -> Array1<f64>,
  G: Fn(&Array1<f64>, f64) -> Array2<f64>,
{
  pub fn new(drift: F, diffusion: G, noise: NoiseModel, hursts: Option<Array1<f64>>) -> Self {
    Self {
      drift,
      diffusion,
      noise,
      hursts,
    }
  }

  pub fn solve(
    &self,
    x0: &Array1<f64>,
    t0: f64,
    t1: f64,
    dt: f64,
    n_paths: usize,
    method: SdeMethod,
    rng: &mut impl Rng,
  ) -> Array3<f64> {
    match self.noise {
      NoiseModel::Gaussian => match method {
        SdeMethod::Euler => self.solve_euler_gauss(x0, t0, t1, dt, n_paths, rng),
        SdeMethod::Milstein => self.solve_milstein_gauss(x0, t0, t1, dt, n_paths, rng),
        SdeMethod::SRK2 => self.solve_srk2_gauss(x0, t0, t1, dt, n_paths, rng),
        SdeMethod::SRK4 => self.solve_srk4_gauss(x0, t0, t1, dt, n_paths, rng),
      },
      NoiseModel::Fractional => {
        let steps = ((t1 - t0) / dt).ceil() as usize;
        let dim = x0.len();
        let mut incs = Array3::zeros((n_paths, steps, dim));

        if let Some(h) = &self.hursts {
          let fgns: Vec<FGN<f64>> = (0..dim)
            .map(|d| FGN::new(h[d], steps, Some(t1 - t0)))
            .collect();

          for p in 0..n_paths {
            for d in 0..dim {
              let data = fgns[d].sample();
              incs.slice_mut(s![p, .., d]).assign(&data);
            }
          }
        }

        match method {
          SdeMethod::Euler => self.solve_euler_fractional(x0, t0, dt, &incs),
          SdeMethod::Milstein => self.solve_milstein_fractional(x0, t0, dt, &incs),
          SdeMethod::SRK2 => self.solve_srk2_fractional(x0, t0, dt, &incs),
          SdeMethod::SRK4 => self.solve_srk4_fractional(x0, t0, dt, &incs),
        }
      }
    }
  }

  fn fill_gauss_increment(&self, out: &mut [f64], sqrt_dt: f64, _rng: &mut impl Rng) {
    <f64 as FloatExt>::fill_standard_normal_slice(out);
    for x in out.iter_mut() {
      *x *= sqrt_dt;
    }
  }

  fn solve_euler_gauss(
    &self,
    x0: &Array1<f64>,
    t0: f64,
    t1: f64,
    dt: f64,
    n_paths: usize,
    rng: &mut impl Rng,
  ) -> Array3<f64> {
    let steps = ((t1 - t0) / dt).ceil() as usize;
    let dim = x0.len();
    let sqrt_dt = dt.sqrt();
    let mut out = Array3::zeros((n_paths, steps + 1, dim));
    for p in 0..n_paths {
      let mut x = x0.clone();
      let mut d_w = vec![0.0; dim];
      let mut time = t0;
      out.slice_mut(s![p, 0, ..]).assign(&x);
      for i in 1..=steps {
        self.fill_gauss_increment(&mut d_w, sqrt_dt, rng);
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
        out.slice_mut(s![p, i, ..]).assign(&x);
      }
    }
    out
  }

  fn milstein_correction(
    &self,
    x: &Array1<f64>,
    time: f64,
    d_w: ArrayView1<'_, f64>,
    dt: f64,
  ) -> Array1<f64> {
    let dim = x.len();
    let eps = 1e-7;
    let sigma_val = (self.diffusion)(x, time);
    let mut correction = Array1::zeros(dim);

    for j in 0..dim {
      let mut x_plus = x.clone();
      x_plus[j] += eps;
      let sigma_plus = (self.diffusion)(&x_plus, time);

      for i_dim in 0..dim {
        for l in 0..dim {
          let dsig = (sigma_plus[[i_dim, l]] - sigma_val[[i_dim, l]]) / eps;
          let lj_bil = sigma_val[[j, l]] * dsig;
          let i_jl = if j == l {
            0.5 * (d_w[j] * d_w[l] - dt)
          } else {
            0.5 * d_w[j] * d_w[l]
          };
          correction[i_dim] += lj_bil * i_jl;
        }
      }
    }

    correction
  }

  fn solve_milstein_gauss(
    &self,
    x0: &Array1<f64>,
    t0: f64,
    t1: f64,
    dt: f64,
    n_paths: usize,
    rng: &mut impl Rng,
  ) -> Array3<f64> {
    let steps = ((t1 - t0) / dt).ceil() as usize;
    let dim = x0.len();
    let sqrt_dt = dt.sqrt();
    let mut out = Array3::zeros((n_paths, steps + 1, dim));
    for p in 0..n_paths {
      let mut x = x0.clone();
      let mut d_w = vec![0.0; dim];
      let mut time = t0;
      out.slice_mut(s![p, 0, ..]).assign(&x);
      for i in 1..=steps {
        self.fill_gauss_increment(&mut d_w, sqrt_dt, rng);
        let mu_val = (self.drift)(&x, time);
        let sigma_val = (self.diffusion)(&x, time);
        let correction = self.milstein_correction(&x, time, ArrayView1::from(&d_w[..]), dt);
        for i_dim in 0..dim {
          let mut incr = mu_val[i_dim] * dt;
          for j_dim in 0..dim {
            incr += sigma_val[[i_dim, j_dim]] * d_w[j_dim];
          }
          incr += correction[i_dim];
          x[i_dim] += incr;
        }
        time += dt;
        out.slice_mut(s![p, i, ..]).assign(&x);
      }
    }
    out
  }

  fn solve_srk2_gauss(
    &self,
    x0: &Array1<f64>,
    t0: f64,
    t1: f64,
    dt: f64,
    n_paths: usize,
    rng: &mut impl Rng,
  ) -> Array3<f64> {
    let steps = ((t1 - t0) / dt).ceil() as usize;
    let dim = x0.len();
    let sqrt_dt = dt.sqrt();
    let mut out = Array3::zeros((n_paths, steps + 1, dim));
    for p in 0..n_paths {
      let mut x = x0.clone();
      let mut d_w = vec![0.0; dim];
      let mut time = t0;
      out.slice_mut(s![p, 0, ..]).assign(&x);
      for i in 1..=steps {
        self.fill_gauss_increment(&mut d_w, sqrt_dt, rng);
        let mu1 = (self.drift)(&x, time);
        let sig1 = (self.diffusion)(&x, time);
        let mut x_half = x.clone();
        for i_dim in 0..dim {
          let mut incr = mu1[i_dim] * (0.5 * dt);
          for j_dim in 0..dim {
            incr += sig1[[i_dim, j_dim]] * (0.5 * d_w[j_dim]);
          }
          x_half[i_dim] += incr;
        }
        let mu2 = (self.drift)(&x_half, time + 0.5 * dt);
        let sig2 = (self.diffusion)(&x_half, time + 0.5 * dt);
        for i_dim in 0..dim {
          let mut incr = mu2[i_dim] * dt;
          for j_dim in 0..dim {
            incr += sig2[[i_dim, j_dim]] * d_w[j_dim];
          }
          x[i_dim] += incr;
        }
        time += dt;
        out.slice_mut(s![p, i, ..]).assign(&x);
      }
    }
    out
  }

  fn solve_srk4_gauss(
    &self,
    x0: &Array1<f64>,
    t0: f64,
    t1: f64,
    dt: f64,
    n_paths: usize,
    rng: &mut impl Rng,
  ) -> Array3<f64> {
    let steps = ((t1 - t0) / dt).ceil() as usize;
    let dim = x0.len();
    let sqrt_dt = dt.sqrt();
    let mut out = Array3::zeros((n_paths, steps + 1, dim));
    for p in 0..n_paths {
      let mut x = x0.clone();
      let mut d_w_full = vec![0.0; dim];
      let mut time = t0;
      out.slice_mut(s![p, 0, ..]).assign(&x);
      for i in 1..=steps {
        self.fill_gauss_increment(&mut d_w_full, sqrt_dt, rng);
        let k1_mu = (self.drift)(&x, time);
        let k1_sig = (self.diffusion)(&x, time);
        let mut x1 = x.clone();
        for i_dim in 0..dim {
          let mut incr = k1_mu[i_dim] * (dt / 2.0);
          for j_dim in 0..dim {
            incr += k1_sig[[i_dim, j_dim]] * (d_w_full[j_dim] * 0.5);
          }
          x1[i_dim] += incr;
        }
        let k2_mu = (self.drift)(&x1, time + dt / 2.0);
        let k2_sig = (self.diffusion)(&x1, time + dt / 2.0);
        let mut x2 = x.clone();
        for i_dim in 0..dim {
          let mut incr = k2_mu[i_dim] * (dt / 2.0);
          for j_dim in 0..dim {
            incr += k2_sig[[i_dim, j_dim]] * (d_w_full[j_dim] * 0.5);
          }
          x2[i_dim] += incr;
        }
        let k3_mu = (self.drift)(&x2, time + dt / 2.0);
        let k3_sig = (self.diffusion)(&x2, time + dt / 2.0);
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
            (k1_mu[i_dim] + 2.0 * k2_mu[i_dim] + 2.0 * k3_mu[i_dim] + k4_mu[i_dim]) / 6.0;
          let mut incr = drift_avg * dt;
          for j_dim in 0..dim {
            let diff_ij = (k1_sig[[i_dim, j_dim]]
              + 2.0 * k2_sig[[i_dim, j_dim]]
              + 2.0 * k3_sig[[i_dim, j_dim]]
              + k4_sig[[i_dim, j_dim]])
              / 6.0;
            incr += diff_ij * d_w_full[j_dim];
          }
          x[i_dim] += incr;
        }
        time += dt;
        out.slice_mut(s![p, i, ..]).assign(&x);
      }
    }
    out
  }

  fn solve_euler_fractional(
    &self,
    x0: &Array1<f64>,
    t0: f64,
    dt: f64,
    incs: &Array3<f64>,
  ) -> Array3<f64> {
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
        let dW = incs.slice(s![p, i_step - 1, ..]);
        let mu_val = (self.drift)(&x, time);
        let sigma_val = (self.diffusion)(&x, time);
        for i_dim in 0..dim {
          let mut incr = mu_val[i_dim] * dt;
          for j_dim in 0..dim {
            incr += sigma_val[[i_dim, j_dim]] * dW[j_dim];
          }
          x[i_dim] += incr;
        }
        time += dt;
        out.slice_mut(s![p, i_step, ..]).assign(&x);
      }
    }
    out
  }

  fn solve_milstein_fractional(
    &self,
    x0: &Array1<f64>,
    t0: f64,
    dt: f64,
    incs: &Array3<f64>,
  ) -> Array3<f64> {
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

  fn solve_srk2_fractional(
    &self,
    x0: &Array1<f64>,
    t0: f64,
    dt: f64,
    incs: &Array3<f64>,
  ) -> Array3<f64> {
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
        let dW = incs.slice(s![p, i_step - 1, ..]);
        let mu1 = (self.drift)(&x, time);
        let sig1 = (self.diffusion)(&x, time);
        let mut x_half = x.clone();
        for i_dim in 0..dim {
          let mut incr = mu1[i_dim] * (0.5 * dt);
          for j_dim in 0..dim {
            incr += sig1[[i_dim, j_dim]] * (0.5 * dW[j_dim]);
          }
          x_half[i_dim] += incr;
        }
        let mu2 = (self.drift)(&x_half, time + 0.5 * dt);
        let sig2 = (self.diffusion)(&x_half, time + 0.5 * dt);
        for i_dim in 0..dim {
          let mut incr = mu2[i_dim] * dt;
          for j_dim in 0..dim {
            incr += sig2[[i_dim, j_dim]] * dW[j_dim];
          }
          x[i_dim] += incr;
        }
        time += dt;
        out.slice_mut(s![p, i_step, ..]).assign(&x);
      }
    }
    out
  }

  fn solve_srk4_fractional(
    &self,
    x0: &Array1<f64>,
    t0: f64,
    dt: f64,
    incs: &Array3<f64>,
  ) -> Array3<f64> {
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
        let dW_full = incs.slice(s![p, i_step - 1, ..]);
        let k1_mu = (self.drift)(&x, time);
        let k1_sig = (self.diffusion)(&x, time);
        let mut x1 = x.clone();
        for i_dim in 0..dim {
          let mut incr = k1_mu[i_dim] * (dt / 2.0);
          for j_dim in 0..dim {
            incr += k1_sig[[i_dim, j_dim]] * (dW_full[j_dim] * 0.5);
          }
          x1[i_dim] += incr;
        }
        let k2_mu = (self.drift)(&x1, time + dt / 2.0);
        let k2_sig = (self.diffusion)(&x1, time + dt / 2.0);
        let mut x2 = x.clone();
        for i_dim in 0..dim {
          let mut incr = k2_mu[i_dim] * (dt / 2.0);
          for j_dim in 0..dim {
            incr += k2_sig[[i_dim, j_dim]] * (dW_full[j_dim] * 0.5);
          }
          x2[i_dim] += incr;
        }
        let k3_mu = (self.drift)(&x2, time + dt / 2.0);
        let k3_sig = (self.diffusion)(&x2, time + dt / 2.0);
        let mut x3 = x.clone();
        for i_dim in 0..dim {
          let mut incr = k3_mu[i_dim] * dt;
          for j_dim in 0..dim {
            incr += k3_sig[[i_dim, j_dim]] * dW_full[j_dim];
          }
          x3[i_dim] += incr;
        }
        let k4_mu = (self.drift)(&x3, time + dt);
        let k4_sig = (self.diffusion)(&x3, time + dt);
        for i_dim in 0..dim {
          let drift_avg =
            (k1_mu[i_dim] + 2.0 * k2_mu[i_dim] + 2.0 * k3_mu[i_dim] + k4_mu[i_dim]) / 6.0;
          let mut incr = drift_avg * dt;
          for j_dim in 0..dim {
            let diff_ij = (k1_sig[[i_dim, j_dim]]
              + 2.0 * k2_sig[[i_dim, j_dim]]
              + 2.0 * k3_sig[[i_dim, j_dim]]
              + k4_sig[[i_dim, j_dim]])
              / 6.0;
            incr += diff_ij * dW_full[j_dim];
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
