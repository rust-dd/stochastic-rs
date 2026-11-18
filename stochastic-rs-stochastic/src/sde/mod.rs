//! $$
//! dX_t^i = a^i(X_t, t)\,dt + \sum_{j=1}^{d} b^{ij}(X_t, t)\,dW_t^j, \quad i = 1, \ldots, d
//! $$
//!
//! Solves a custom $d$-dimensional Itô SDE from drift and diffusion closures; a built-in model has
//! its own struct. Schemes: <https://stochastic.rust-dd.com/docs/processes#generic-sde-solver>.
//!
//! Reference: Maruyama (1955), Rend. Circ. Mat. Palermo 4, 48–90
//! Reference: Milstein (1974), Theory Probab. Appl. 19(3), 557–562
//!
//! ```rust
//! use ndarray::{array, Array1, Array2};
//! use stochastic_rs_core::simd_rng::Deterministic;
//! use stochastic_rs_stochastic::sde::{Sde, SdeMethod, NoiseModel};
//!
//! // Geometric Brownian Motion: dS = mu*S*dt + sigma*S*dW
//! let mu = 0.05_f64;
//! let sigma = 0.2_f64;
//!
//! let sde = Sde::new(
//!   // drift: a(x, t) = mu * x
//!   move |x: &Array1<f64>, _t: f64| array![mu * x[0]],
//!   // diffusion: b(x, t) = [[sigma * x]]
//!   move |x: &Array1<f64>, _t: f64| Array2::from_elem((1, 1), sigma * x[0]),
//!   NoiseModel::Gaussian,
//!   None,
//! );
//!
//! let x0 = array![100.0_f64];
//! let paths = sde.solve(&x0, 0.0, 1.0, 1000, 100, SdeMethod::Milstein, &Deterministic::new(42));
//! assert_eq!(paths.shape(), [100, 1001, 1]);
//! ```

mod fractional;
mod gaussian;
#[cfg(test)]
mod tests;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::Array3;
use ndarray::ArrayView1;
use ndarray::s;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use super::noise::fgn::Fgn;
use crate::device::Cpu;
use crate::device::FgnBackend;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Selects the driving noise process for the SDE.
#[derive(Debug)]
#[non_exhaustive]
pub enum NoiseModel {
  /// Standard Brownian motion with i.i.d. Gaussian increments.
  Gaussian,
  /// Fractional Brownian motion. Requires [`Sde::hursts`] to be set with one
  /// Hurst parameter $H \in (0, 1)$ per state dimension.
  Fractional,
}

/// Discretisation scheme of [`Sde::solve`]; every scheme takes one Brownian increment per step.
#[derive(Debug)]
#[non_exhaustive]
pub enum SdeMethod {
  /// Euler–Maruyama scheme. Strong order 0.5, weak order 1.0.
  Euler,
  /// Milstein with the commutative-noise approximation of the double Itô integrals (no Lévy area)
  /// and forward differences at $\varepsilon = 10^{-7}$. Strong order 1.0, weak order 1.0.
  Milstein,
  /// Stochastic midpoint method (RK2-style). Strong order 0.5.
  SRK2,
  /// Classical RK4 structure on drift and diffusion. Strong order 0.5, not 4: for higher-order
  /// stochastic schemes see Rößler (2010) or Burrage & Platen (1994).
  SRK4,
}

/// Solver for $d$-dimensional Itô SDEs: `F` is the drift `a(x, t) -> R^d` and `G` the diffusion
/// `b(x, t) -> R^{d×d}`, whose entry `[i, j]` maps noise dimension `j` to state dimension `i`.
pub struct Sde<T: FloatExt, F, G, B = Cpu>
where
  F: Fn(&Array1<T>, T) -> Array1<T>,
  G: Fn(&Array1<T>, T) -> Array2<T>,
{
  /// Drift coefficient $a(X_t, t)$.
  pub drift: F,
  /// Diffusion matrix $b(X_t, t)$. Shape `[d, d]` where `b[i][j]` maps noise $dW^j$ to state $X^i$.
  pub diffusion: G,
  /// Noise model selection (Gaussian or fractional).
  pub noise: NoiseModel,
  /// Hurst parameters for fractional noise, one per dimension.
  /// Required when `noise` is [`NoiseModel::Fractional`], ignored otherwise.
  pub hursts: Option<Array1<T>>,
  pub backend: B,
}

impl<T: FloatExt, F, G, B: std::fmt::Debug> std::fmt::Debug for Sde<T, F, G, B>
where
  F: Fn(&Array1<T>, T) -> Array1<T>,
  G: Fn(&Array1<T>, T) -> Array2<T>,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("Sde")
      .field("noise", &self.noise)
      .field("hursts", &self.hursts)
      .field("backend", &self.backend)
      .finish_non_exhaustive()
  }
}

impl<T: FloatExt, F, G> Sde<T, F, G, Cpu>
where
  F: Fn(&Array1<T>, T) -> Array1<T>,
  G: Fn(&Array1<T>, T) -> Array2<T>,
{
  /// `hursts` holds one Hurst parameter per dimension and is read under fractional noise only.
  pub fn new(drift: F, diffusion: G, noise: NoiseModel, hursts: Option<Array1<T>>) -> Self {
    Self {
      drift,
      diffusion,
      noise,
      hursts,
      backend: Cpu,
    }
  }
}

backend_switch!([T: FloatExt, F, G] Sde<T, F, G> { drift, diffusion, noise, hursts } via phantom
  where F: Fn(&Array1<T>, T) -> Array1<T>, G: Fn(&Array1<T>, T) -> Array2<T>);

impl<T: FloatExt, F, G, B: FgnBackend<T>> Sde<T, F, G, B>
where
  F: Fn(&Array1<T>, T) -> Array1<T>,
  G: Fn(&Array1<T>, T) -> Array2<T>,
{
  /// `[n_paths, steps + 1, dim]` paths from `x0` at `t0` to `t1`, `dt = (t1 − t0) / steps`, drawn
  /// from `seed`; panics unless `t0 < t1` with a finite span and `steps ≥ 1`.
  pub fn solve<S: SeedExt>(
    &self,
    x0: &Array1<T>,
    t0: T,
    t1: T,
    steps: usize,
    n_paths: usize,
    method: SdeMethod,
    seed: &S,
  ) -> Array3<T>
  where
    B: Default,
  {
    assert!(
      steps >= 1,
      "steps must satisfy `steps >= 1`, got steps = {steps:?}"
    );
    assert!(
      t1 > t0 && (t1 - t0).is_finite(),
      "t1 must satisfy `t1 > t0 && (t1 - t0).is_finite()`, got t0 = {t0:?}, t1 = {t1:?}"
    );
    let dt = (t1 - t0) / T::from_usize_(steps);
    match self.noise {
      NoiseModel::Gaussian => {
        let mut normal = SimdNormal::<T>::new(T::zero(), T::one()).seeded(seed);
        match method {
          SdeMethod::Euler => self.solve_euler_gauss(x0, t0, steps, dt, n_paths, &mut normal),
          SdeMethod::Milstein => self.solve_milstein_gauss(x0, t0, steps, dt, n_paths, &mut normal),
          SdeMethod::SRK2 => self.solve_srk2_gauss(x0, t0, steps, dt, n_paths, &mut normal),
          SdeMethod::SRK4 => self.solve_srk4_gauss(x0, t0, steps, dt, n_paths, &mut normal),
        }
      }
      NoiseModel::Fractional => {
        let dim = x0.len();
        let mut incs = Array3::zeros((n_paths, steps, dim));

        if let Some(h) = &self.hursts {
          let fgns = (0..dim)
            .map(|d| Fgn::new(h[d], steps, Some(t1 - t0), seed.derive()).on::<B>())
            .collect::<Vec<_>>();

          for p in 0..n_paths {
            for (d, fgn) in fgns.iter().enumerate().take(dim) {
              let data = fgn.sample();
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

  /// Fills `out` with i.i.d. $\mathcal{N}(0, \Delta t)$ Brownian increments, one per dimension.
  pub(super) fn fill_gauss_increment(
    &self,
    out: &mut [T],
    sqrt_dt: T,
    normal: &mut Seeded<SimdNormal<T>>,
  ) {
    normal.fill_slice(out);
    for x in out.iter_mut() {
      *x = *x * sqrt_dt;
    }
  }

  /// The Milstein correction of component $i$, with forward differences at
  /// $\varepsilon = 10^{-7}$ and the commutative-noise approximation of the double Itô integrals:
  /// $$
  /// C^i = \sum_{j_1=1}^{d} \sum_{j_2=1}^{d}
  ///   \biggl[\sum_{k=1}^{d} b^{k,j_1} \frac{\partial b^{i,j_2}}{\partial x_k}\biggr]
  ///   \cdot I_{(j_1, j_2)}
  /// $$
  pub(super) fn milstein_correction(
    &self,
    x: &Array1<T>,
    time: T,
    d_w: ArrayView1<'_, T>,
    dt: T,
  ) -> Array1<T> {
    let dim = x.len();
    let eps = T::from_f64_fast(1e-7);
    let half = T::from_f64_fast(0.5);
    let sigma_val = (self.diffusion)(x, time);
    let mut correction = Array1::zeros(dim);

    let mut dsigma_dx: Vec<Array2<T>> = Vec::with_capacity(dim);
    for k in 0..dim {
      let mut x_plus = x.clone();
      x_plus[k] += eps;
      let sigma_plus = (self.diffusion)(&x_plus, time);
      dsigma_dx.push((&sigma_plus - &sigma_val) / eps);
    }

    for j1 in 0..dim {
      for j2 in 0..dim {
        let i_j1j2 = if j1 == j2 {
          half * (d_w[j1] * d_w[j2] - dt)
        } else {
          half * d_w[j1] * d_w[j2]
        };

        for i_dim in 0..dim {
          let mut lj1_sigma_ij2 = T::zero();
          for k in 0..dim {
            lj1_sigma_ij2 += sigma_val[[k, j1]] * dsigma_dx[k][[i_dim, j2]];
          }
          correction[i_dim] += lj1_sigma_ij2 * i_j1j2;
        }
      }
    }

    correction
  }
}
