//! Stochastic differential equation solver

use ndarray::{s, Array1, Array2, Array3, Axis};
use rand::Rng;

use super::{noise::fgn::FGN, Sampling};

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
          for p in 0..n_paths {
            for d in 0..dim {
              let fgn = FGN::new(h[d], steps, Some(t1 - t0), None);
              let data = fgn.sample();

              for i in 0..steps {
                incs[[p, i, d]] = data[i];
              }
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

  fn gauss_increment(&self, dim: usize, dt: f64, rng: &mut impl Rng) -> Array1<f64> {
    let sqrt_dt = dt.sqrt();
    let mut inc = Array1::zeros(dim);
    for i in 0..dim {
      inc[i] = rng.sample::<f64, _>(rand_distr::StandardNormal) * sqrt_dt;
    }
    inc
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
    let mut out = Array3::zeros((n_paths, steps + 1, dim));
    for p in 0..n_paths {
      let mut x = x0.clone();
      let mut time = t0;
      out.slice_mut(s![p, 0, ..]).assign(&x);
      for i in 1..=steps {
        let dW = self.gauss_increment(dim, dt, rng);
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
        out.slice_mut(s![p, i, ..]).assign(&x);
      }
    }
    out
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
    let mut out = Array3::zeros((n_paths, steps + 1, dim));
    for p in 0..n_paths {
      let mut x = x0.clone();
      let mut time = t0;
      out.slice_mut(s![p, 0, ..]).assign(&x);
      for i in 1..=steps {
        let dW = self.gauss_increment(dim, dt, rng);
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
    let mut out = Array3::zeros((n_paths, steps + 1, dim));
    for p in 0..n_paths {
      let mut x = x0.clone();
      let mut time = t0;
      out.slice_mut(s![p, 0, ..]).assign(&x);
      for i in 1..=steps {
        let dW = self.gauss_increment(dim, dt, rng);
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
    let mut out = Array3::zeros((n_paths, steps + 1, dim));
    for p in 0..n_paths {
      let mut x = x0.clone();
      let mut time = t0;
      out.slice_mut(s![p, 0, ..]).assign(&x);
      for i in 1..=steps {
        let dW_full = self.gauss_increment(dim, dt, rng);
        let mut partials = vec![Array1::<f64>::zeros(dim); 4];
        for idx in 0..4 {
          for d in 0..dim {
            partials[idx][d] = dW_full[d] * 0.25;
          }
        }
        let k1_mu = (self.drift)(&x, time);
        let k1_sig = (self.diffusion)(&x, time);
        let mut x1 = x.clone();
        for i_dim in 0..dim {
          let mut incr = k1_mu[i_dim] * (dt / 4.0);
          for j_dim in 0..dim {
            incr += k1_sig[[i_dim, j_dim]] * partials[0][j_dim];
          }
          x1[i_dim] += incr;
        }
        let k2_mu = (self.drift)(&x1, time + dt / 4.0);
        let k2_sig = (self.diffusion)(&x1, time + dt / 4.0);
        let mut x2 = x.clone();
        for i_dim in 0..dim {
          let mut incr = k2_mu[i_dim] * (dt / 4.0);
          for j_dim in 0..dim {
            incr += k2_sig[[i_dim, j_dim]] * partials[1][j_dim];
          }
          x2[i_dim] += incr;
        }
        let k3_mu = (self.drift)(&x2, time + dt / 4.0);
        let k3_sig = (self.diffusion)(&x2, time + dt / 4.0);
        let mut x3 = x.clone();
        for i_dim in 0..dim {
          let mut incr = k3_mu[i_dim] * (dt / 4.0);
          for j_dim in 0..dim {
            incr += k3_sig[[i_dim, j_dim]] * partials[2][j_dim];
          }
          x3[i_dim] += incr;
        }
        let k4_mu = (self.drift)(&x3, time + dt / 4.0);
        let k4_sig = (self.diffusion)(&x3, time + dt / 4.0);
        for i_dim in 0..dim {
          let drift_avg =
            (k1_mu[i_dim] + 2.0 * k2_mu[i_dim] + 2.0 * k3_mu[i_dim] + k4_mu[i_dim]) / 6.0;
          let mut diff_vec = vec![0.0; dim];
          for j_dim in 0..dim {
            diff_vec[j_dim] = (k1_sig[[i_dim, j_dim]]
              + 2.0 * k2_sig[[i_dim, j_dim]]
              + 2.0 * k3_sig[[i_dim, j_dim]]
              + k4_sig[[i_dim, j_dim]])
              / 6.0;
          }
          let mut incr = drift_avg * dt;
          for j_dim in 0..dim {
            incr += diff_vec[j_dim] * dW_full[j_dim];
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
        let mut partials = vec![Array1::<f64>::zeros(dim); 4];
        for idx in 0..4 {
          for d in 0..dim {
            partials[idx][d] = dW_full[d] * 0.25;
          }
        }
        let k1_mu = (self.drift)(&x, time);
        let k1_sig = (self.diffusion)(&x, time);
        let mut x1 = x.clone();
        for i_dim in 0..dim {
          let mut incr = k1_mu[i_dim] * (dt / 4.0);
          for j_dim in 0..dim {
            incr += k1_sig[[i_dim, j_dim]] * partials[0][j_dim];
          }
          x1[i_dim] += incr;
        }
        let k2_mu = (self.drift)(&x1, time + dt / 4.0);
        let k2_sig = (self.diffusion)(&x1, time + dt / 4.0);
        let mut x2 = x.clone();
        for i_dim in 0..dim {
          let mut incr = k2_mu[i_dim] * (dt / 4.0);
          for j_dim in 0..dim {
            incr += k2_sig[[i_dim, j_dim]] * partials[1][j_dim];
          }
          x2[i_dim] += incr;
        }
        let k3_mu = (self.drift)(&x2, time + dt / 4.0);
        let k3_sig = (self.diffusion)(&x2, time + dt / 4.0);
        let mut x3 = x.clone();
        for i_dim in 0..dim {
          let mut incr = k3_mu[i_dim] * (dt / 4.0);
          for j_dim in 0..dim {
            incr += k3_sig[[i_dim, j_dim]] * partials[2][j_dim];
          }
          x3[i_dim] += incr;
        }
        let k4_mu = (self.drift)(&x3, time + dt / 4.0);
        let k4_sig = (self.diffusion)(&x3, time + dt / 4.0);
        for i_dim in 0..dim {
          let drift_avg =
            (k1_mu[i_dim] + 2.0 * k2_mu[i_dim] + 2.0 * k3_mu[i_dim] + k4_mu[i_dim]) / 6.0;
          let mut diff_vec = vec![0.0; dim];
          for j_dim in 0..dim {
            diff_vec[j_dim] = (k1_sig[[i_dim, j_dim]]
              + 2.0 * k2_sig[[i_dim, j_dim]]
              + 2.0 * k3_sig[[i_dim, j_dim]]
              + k4_sig[[i_dim, j_dim]])
              / 6.0;
          }
          let mut incr = drift_avg * dt;
          for j_dim in 0..dim {
            incr += diff_vec[j_dim] * dW_full[j_dim];
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

#[cfg(test)]
mod tests {
  use super::*;
  use ndarray::{arr1, arr2};
  use rand::thread_rng;

  #[test]
  fn test_fgn_1d_euler() {
    let drift = |x: &Array1<f64>, _t: f64| arr1(&[-0.5 * x[0]]);
    let diffusion = |_x: &Array1<f64>, _t: f64| arr2(&[[0.1]]);
    let sde = Sde::new(
      drift,
      diffusion,
      NoiseModel::Fractional,
      Some(arr1(&[0.7, 0.8])),
    );
    let mut rng = thread_rng();
    let x0 = arr1(&[1.0]);
    let result = sde.solve(&x0, 0.0, 1.0, 0.01, 2, SdeMethod::Euler, &mut rng);
    assert_eq!(result.shape(), &[2, 101, 1]);
  }

  #[test]
  fn test_fgn_2d_srk2() {
    let drift = |x: &Array1<f64>, _t: f64| arr1(&[-0.5 * x[0], -0.2 * x[1]]);
    let diffusion = |_x: &Array1<f64>, _t: f64| arr2(&[[0.1, 0.0], [0.0, 0.2]]);
    let sde = Sde::new(
      drift,
      diffusion,
      NoiseModel::Fractional,
      Some(arr1(&[0.8, 0.75])),
    );
    let mut rng = thread_rng();
    let x0 = arr1(&[1.0, 1.0]);
    let result = sde.solve(&x0, 0.0, 1.0, 0.01, 2, SdeMethod::SRK2, &mut rng);
    assert_eq!(result.shape(), &[2, 101, 2]);
  }

  #[test]
  fn test_fgn_4d_srk2() {
    let drift =
      |x: &Array1<f64>, _t: f64| arr1(&[-0.5 * x[0], -0.2 * x[1], -0.3 * x[2], -0.4 * x[3]]);
    let diffusion = |_x: &Array1<f64>, _t: f64| {
      arr2(&[
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.0],
        [0.0, 0.0, 0.3, 0.0],
        [0.0, 0.0, 0.0, 0.4],
      ])
    };
    let sde = Sde::new(
      drift,
      diffusion,
      NoiseModel::Fractional,
      Some(arr1(&[0.8, 0.9, 0.75, 0.78])),
    );
    let mut rng = thread_rng();
    let x0 = arr1(&[1.0, 1.0, 1.0, 1.0]);
    let result = sde.solve(&x0, 0.0, 1.0, 0.01, 4, SdeMethod::SRK2, &mut rng);
    assert_eq!(result.shape(), &[4, 101, 4]);
  }
}
