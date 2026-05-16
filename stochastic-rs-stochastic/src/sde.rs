//! # Generic SDE Solver
//!
//! ## When to use this vs. the concrete process structs
//!
//! The crate ships ~140 concrete process structs (`Bm`, `Gbm`, `Heston`,
//! `Vasicek`, …) implementing [`crate::traits::ProcessExt`]. Those are the
//! preferred entry points for built-in models — they have hand-optimised
//! samplers (FFT circulant embedding, Andersen QE, antithetic, GPU paths, etc.)
//! and integrate with the broader pipeline (`ModelPricer`, `MalliavinExt`,
//! visualization, Python bindings).
//!
//! This [`Sde`] type is the **research / custom-SDE** alternative: a
//! flexible generic solver that takes user-supplied drift and diffusion
//! closures and discretises them with Euler–Maruyama, Milstein, midpoint
//! RK2, or implicit Euler. Use it when:
//!
//! - Your SDE is not in the catalogue of built-in processes
//! - You want to compare integration schemes side-by-side
//! - You're implementing a one-off model and don't need a dedicated struct
//!
//! Outputs flow through `Array1<T>` / `Array2<T>` and so do not require a
//! separate `ProcessExt` impl, but they are also not auto-discoverable to
//! the rest of the pricing / Greeks / vol-surface stack.
//!
//! Numerical solver for $d$-dimensional Itô stochastic differential equations of the form
//!
//! $$
//! dX_t^i = a^i(X_t, t)\,dt + \sum_{j=1}^{d} b^{ij}(X_t, t)\,dW_t^j, \quad i = 1, \ldots, d
//! $$
//!
//! where:
//! - $X_t \in \mathbb{R}^d$ is the state vector,
//! - $a: \mathbb{R}^d \times \mathbb{R} \to \mathbb{R}^d$ is the drift coefficient,
//! - $b: \mathbb{R}^d \times \mathbb{R} \to \mathbb{R}^{d \times d}$ is the diffusion matrix,
//! - $W_t \in \mathbb{R}^d$ is a $d$-dimensional Wiener process (standard or fractional).
//!
//! The solver is generic over any precision type `T` implementing [`FloatExt`] (e.g. `f32`, `f64`).
//!
//! ## Numerical Methods
//!
//! Four discretization schemes are provided via [`SdeMethod`]:
//!
//! ### Euler–Maruyama (`SdeMethod::Euler`)
//!
//! First-order strong scheme (strong order 0.5, weak order 1.0). The simplest and most widely
//! used method. At each step:
//!
//! $$
//! X_{n+1}^i = X_n^i + a^i(X_n, t_n)\,\Delta t + \sum_{j=1}^{d} b^{ij}(X_n, t_n)\,\Delta W_n^j
//! $$
//!
//! where $\Delta W_n^j \sim \mathcal{N}(0, \Delta t)$ are independent Gaussian increments.
//!
//! **Reference:** Maruyama, G. (1955). *Continuous Markov processes and stochastic equations.*
//! Rendiconti del Circolo Matematico di Palermo, 4, 48–90.
//!
//! ### Milstein (`SdeMethod::Milstein`)
//!
//! Second-order strong scheme (strong order 1.0, weak order 1.0) that adds a correction term
//! from the Itô–Taylor expansion. The update formula is:
//!
//! $$
//! X_{n+1}^i = X_n^i + a^i \Delta t + \sum_j b^{ij} \Delta W^j
//!   + \sum_{j_1=1}^{d} \sum_{j_2=1}^{d} \mathcal{L}^{j_1} b^{i,j_2} \cdot I_{(j_1,j_2)}
//!
//! $$
//!
//! where the operator $\mathcal{L}^{j_1}$ is defined as:
//!
//! $$
//! \mathcal{L}^{j_1} b^{i,j_2} = \sum_{k=1}^{d} b^{k,j_1}(X_n, t_n) \frac{\partial b^{i,j_2}}{\partial x_k}(X_n, t_n)
//! $$
//!
//! and the double Itô integrals $I_{(j_1,j_2)}$ are approximated under the **commutative noise**
//! assumption as:
//!
//! $$
//! I_{(j,j)} = \tfrac{1}{2}\bigl((\Delta W^j)^2 - \Delta t\bigr), \quad
//! I_{(j_1,j_2)} \approx \tfrac{1}{2}\,\Delta W^{j_1}\,\Delta W^{j_2} \text{ for } j_1 \neq j_2
//! $$
//!
//! The partial derivatives $\partial b / \partial x_k$ are computed via forward finite differences
//! with step size $\varepsilon = 10^{-7}$.
//!
//! **Note:** The approximation $I_{(j_1,j_2)} \approx \frac{1}{2} \Delta W^{j_1} \Delta W^{j_2}$
//! for $j_1 \neq j_2$ is only valid for SDEs with **commutative noise** (i.e., where
//! $\mathcal{L}^{j_1} b^{i,j_2} = \mathcal{L}^{j_2} b^{i,j_1}$). For non-commutative noise,
//! the Lévy area must be simulated, which this implementation does not support.
//!
//! **Reference:** Milstein, G. N. (1974). *Approximate integration of stochastic differential
//! equations.* Theory of Probability & Its Applications, 19(3), 557–562.
//!
//! ### Midpoint RK2 (`SdeMethod::SRK2`)
//!
//! A stochastic midpoint method inspired by the deterministic RK2 (midpoint) scheme.
//! Evaluates drift and diffusion at an intermediate point:
//!
//! $$
//! \hat{X} = X_n + \tfrac{1}{2}\,a(X_n, t_n)\,\Delta t + \tfrac{1}{2}\,b(X_n, t_n)\,\Delta W_n
//! $$
//! $$
//! X_{n+1} = X_n + a(\hat{X},\, t_n + \tfrac{\Delta t}{2})\,\Delta t
//!   + b(\hat{X},\, t_n + \tfrac{\Delta t}{2})\,\Delta W_n
//!
//! $$
//!
//! This method has the same strong convergence order as Euler–Maruyama (0.5) but can offer
//! improved accuracy for drift-dominated problems and better weak convergence properties.
//!
//! ### RK4-style (`SdeMethod::SRK4`)
//!
//! A four-stage method applying the classical RK4 structure to both drift and diffusion.
//! The deterministic component uses the standard RK4 weights $(1, 2, 2, 1)/6$, while the
//! diffusion is evaluated at four points and averaged with the same weights:
//!
//! $$
//! \text{drift:}\quad \bar{a} = \tfrac{1}{6}\bigl(a_1 + 2a_2 + 2a_3 + a_4\bigr)
//! $$
//! $$
//! \text{diffusion:}\quad \bar{b}^{ij} = \tfrac{1}{6}\bigl(b_1^{ij} + 2b_2^{ij} + 2b_3^{ij} + b_4^{ij}\bigr)
//! $$
//! $$
//! X_{n+1} = X_n + \bar{a}\,\Delta t + \sum_j \bar{b}^{ij}\,\Delta W^j
//! $$
//!
//! where the four stages evaluate at $t_n$, $t_n + \Delta t/2$ (twice), and $t_n + \Delta t$
//! using intermediate states constructed analogously to the deterministic RK4 method.
//!
//! **Important:** This is a heuristic extension of deterministic RK4 to SDEs using a single
//! Brownian increment per step. It does **not** achieve strong order 4.0. Its strong convergence
//! order is 0.5 (same as Euler–Maruyama). It can provide improved accuracy for drift-dominated
//! or small-noise SDEs where the deterministic component benefits from higher-order treatment.
//! For genuinely higher-order stochastic methods, see Rößler (2010) or Burrage & Platen (1994).
//!
//! ## Noise Models
//!
//! Two noise models are supported via [`NoiseModel`]:
//!
//! - **Gaussian** (`NoiseModel::Gaussian`): Standard Brownian motion with i.i.d.
//!   $\mathcal{N}(0, \Delta t)$ increments.
//! - **Fractional** (`NoiseModel::Fractional`): Fractional Brownian motion with Hurst parameter
//!   $H \in (0, 1)$, generated via the [`Fgn`] module. Requires setting the `hursts` field with
//!   one Hurst parameter per dimension.
//!
//! ## Output
//!
//! All methods return an [`Array3<T>`] with shape `[n_paths, steps + 1, dim]`, where:
//! - `n_paths` is the number of independent sample paths,
//! - `steps + 1` includes the initial condition at index 0,
//! - `dim` is the state dimension.
//!
//! ## Example
//!
//! ```rust
//! use ndarray::{array, Array1, Array2};
//! use rand::rng;
//! use stochastic_rs::stochastic::sde::{Sde, SdeMethod, NoiseModel};
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
//! let paths = sde.solve(&x0, 0.0, 1.0, 0.001, 100, SdeMethod::Milstein, &mut rng());
//! // paths.shape() == [100, 1001, 1]
//! ```

use ndarray::Array1;
use ndarray::Array2;
use ndarray::Array3;
use ndarray::ArrayView1;
use ndarray::Axis;
use ndarray::s;
use rand::Rng;
use stochastic_rs_core::simd_rng::Unseeded;

use super::noise::fgn::Fgn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Selects the driving noise process for the SDE.
pub enum NoiseModel {
  /// Standard Brownian motion with i.i.d. Gaussian increments.
  Gaussian,
  /// Fractional Brownian motion. Requires [`Sde::hursts`] to be set with one
  /// Hurst parameter $H \in (0, 1)$ per state dimension.
  Fractional,
}

/// Numerical discretization method for the SDE solver.
///
/// See the [module-level documentation](self) for the mathematical formulas and
/// convergence properties of each method.
pub enum SdeMethod {
  /// Euler–Maruyama scheme. Strong order 0.5, weak order 1.0.
  Euler,
  /// Milstein scheme with commutative noise approximation. Strong order 1.0, weak order 1.0.
  Milstein,
  /// Stochastic midpoint method (RK2-style). Strong order 0.5.
  SRK2,
  /// Classical RK4 structure applied to both drift and diffusion. Strong order 0.5.
  SRK4,
}

/// Generic solver for $d$-dimensional Itô SDEs.
///
/// # Type Parameters
///
/// - `T`: Floating-point precision type (e.g. `f32`, `f64`).
/// - `F`: Drift function `a(x, t) -> R^d`. Takes the current state `x ∈ R^d` and time `t`,
///   returns the drift vector.
/// - `G`: Diffusion function `b(x, t) -> R^{d×d}`. Takes the current state and time,
///   returns the diffusion matrix where entry `[i, j]` maps noise dimension `j` to
///   state dimension `i`.
pub struct Sde<T: FloatExt, F, G>
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
}

impl<T: FloatExt, F, G> Sde<T, F, G>
where
  F: Fn(&Array1<T>, T) -> Array1<T>,
  G: Fn(&Array1<T>, T) -> Array2<T>,
{
  /// Creates a new SDE solver.
  ///
  /// # Arguments
  ///
  /// * `drift` - Drift coefficient function $a(x, t)$
  /// * `diffusion` - Diffusion matrix function $b(x, t)$
  /// * `noise` - Noise model (Gaussian or fractional)
  /// * `hursts` - Optional Hurst parameters for fractional noise (one per dimension)
  pub fn new(drift: F, diffusion: G, noise: NoiseModel, hursts: Option<Array1<T>>) -> Self {
    Self {
      drift,
      diffusion,
      noise,
      hursts,
    }
  }

  /// Solves the SDE and returns simulated paths.
  ///
  /// # Arguments
  ///
  /// * `x0` - Initial state $X_0 \in \mathbb{R}^d$
  /// * `t0` - Start time
  /// * `t1` - End time
  /// * `dt` - Time step size $\Delta t$
  /// * `n_paths` - Number of independent sample paths to simulate
  /// * `method` - Numerical discretization scheme to use
  /// * `rng` - Random number generator (used only for Gaussian noise)
  ///
  /// # Returns
  ///
  /// [`Array3<T>`] with shape `[n_paths, steps + 1, dim]` containing all simulated paths.
  /// Index `[p, 0, :]` contains the initial condition for path `p`.
  pub fn solve(
    &self,
    x0: &Array1<T>,
    t0: T,
    t1: T,
    dt: T,
    n_paths: usize,
    method: SdeMethod,
    rng: &mut impl Rng,
  ) -> Array3<T> {
    match self.noise {
      NoiseModel::Gaussian => match method {
        SdeMethod::Euler => self.solve_euler_gauss(x0, t0, t1, dt, n_paths, rng),
        SdeMethod::Milstein => self.solve_milstein_gauss(x0, t0, t1, dt, n_paths, rng),
        SdeMethod::SRK2 => self.solve_srk2_gauss(x0, t0, t1, dt, n_paths, rng),
        SdeMethod::SRK4 => self.solve_srk4_gauss(x0, t0, t1, dt, n_paths, rng),
      },
      NoiseModel::Fractional => {
        let steps = ((t1 - t0) / dt).ceil().to_usize().unwrap();
        let dim = x0.len();
        let mut incs = Array3::zeros((n_paths, steps, dim));

        if let Some(h) = &self.hursts {
          let fgns: Vec<Fgn<T>> = (0..dim)
            .map(|d| Fgn::new(h[d], steps, Some(t1 - t0), Unseeded))
            .collect();

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

  fn fill_gauss_increment(&self, out: &mut [T], sqrt_dt: T, _rng: &mut impl Rng) {
    T::fill_standard_normal_scaled_slice(out, sqrt_dt);
  }

  /// Euler–Maruyama with Gaussian noise.
  ///
  /// Update: $X_{n+1}^i = X_n^i + a^i \Delta t + \sum_j b^{ij} \Delta W^j$
  fn solve_euler_gauss(
    &self,
    x0: &Array1<T>,
    t0: T,
    t1: T,
    dt: T,
    n_paths: usize,
    rng: &mut impl Rng,
  ) -> Array3<T> {
    let steps = ((t1 - t0) / dt).ceil().to_usize().unwrap();
    let dim = x0.len();
    let sqrt_dt = dt.sqrt();
    let mut out = Array3::zeros((n_paths, steps + 1, dim));
    for p in 0..n_paths {
      let mut x = x0.clone();
      let mut d_w = vec![T::zero(); dim];
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

  /// Computes the Milstein correction term for the multi-dimensional case.
  ///
  /// For each state component $i$, computes:
  ///
  /// $$
  /// C^i = \sum_{j_1=1}^{d} \sum_{j_2=1}^{d}
  ///   \biggl[\sum_{k=1}^{d} b^{k,j_1} \frac{\partial b^{i,j_2}}{\partial x_k}\biggr]
  ///   \cdot I_{(j_1, j_2)}
  /// $$
  ///
  /// Partial derivatives are approximated via forward finite differences with
  /// step size $\varepsilon = 10^{-7}$. The double Itô integrals use the commutative
  /// noise approximation.
  fn milstein_correction(
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

    // Pre-compute partial derivatives: dsigma_dx[k][[i, j]] = db^{ij}/dx_k
    let mut dsigma_dx: Vec<Array2<T>> = Vec::with_capacity(dim);
    for k in 0..dim {
      let mut x_plus = x.clone();
      x_plus[k] += eps;
      let sigma_plus = (self.diffusion)(&x_plus, time);
      dsigma_dx.push((&sigma_plus - &sigma_val) / eps);
    }

    // Compute: sum_{j1,j2} [sum_k b_{k,j1} * db_{i,j2}/dx_k] * I_{(j1,j2)}
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

  /// Milstein scheme with Gaussian noise.
  fn solve_milstein_gauss(
    &self,
    x0: &Array1<T>,
    t0: T,
    t1: T,
    dt: T,
    n_paths: usize,
    rng: &mut impl Rng,
  ) -> Array3<T> {
    let steps = ((t1 - t0) / dt).ceil().to_usize().unwrap();
    let dim = x0.len();
    let sqrt_dt = dt.sqrt();
    let mut out = Array3::zeros((n_paths, steps + 1, dim));
    for p in 0..n_paths {
      let mut x = x0.clone();
      let mut d_w = vec![T::zero(); dim];
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

  /// Stochastic midpoint (RK2-style) method with Gaussian noise.
  ///
  /// 1. Predict: $\hat{X} = X_n + \frac{1}{2} a(X_n, t_n) \Delta t + \frac{1}{2} b(X_n, t_n) \Delta W$
  /// 2. Correct: $X_{n+1} = X_n + a(\hat{X}, t_n + \frac{\Delta t}{2}) \Delta t + b(\hat{X}, t_n + \frac{\Delta t}{2}) \Delta W$
  fn solve_srk2_gauss(
    &self,
    x0: &Array1<T>,
    t0: T,
    t1: T,
    dt: T,
    n_paths: usize,
    rng: &mut impl Rng,
  ) -> Array3<T> {
    let steps = ((t1 - t0) / dt).ceil().to_usize().unwrap();
    let dim = x0.len();
    let sqrt_dt = dt.sqrt();
    let half = T::from_f64_fast(0.5);
    let half_dt = half * dt;
    let mut out = Array3::zeros((n_paths, steps + 1, dim));
    for p in 0..n_paths {
      let mut x = x0.clone();
      let mut d_w = vec![T::zero(); dim];
      let mut time = t0;
      out.slice_mut(s![p, 0, ..]).assign(&x);
      for i in 1..=steps {
        self.fill_gauss_increment(&mut d_w, sqrt_dt, rng);
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
        out.slice_mut(s![p, i, ..]).assign(&x);
      }
    }
    out
  }

  /// Classical RK4 structure applied to both drift and diffusion, with Gaussian noise.
  ///
  /// Uses four stage evaluations with the standard $(1, 2, 2, 1)/6$ weighting for both
  /// drift and diffusion averages. All stages share the same Brownian increment $\Delta W$.
  fn solve_srk4_gauss(
    &self,
    x0: &Array1<T>,
    t0: T,
    t1: T,
    dt: T,
    n_paths: usize,
    rng: &mut impl Rng,
  ) -> Array3<T> {
    let steps = ((t1 - t0) / dt).ceil().to_usize().unwrap();
    let dim = x0.len();
    let sqrt_dt = dt.sqrt();
    let half = T::from_f64_fast(0.5);
    let two = T::from_f64_fast(2.0);
    let six = T::from_f64_fast(6.0);
    let half_dt = half * dt;
    let mut out = Array3::zeros((n_paths, steps + 1, dim));
    for p in 0..n_paths {
      let mut x = x0.clone();
      let mut d_w_full = vec![T::zero(); dim];
      let mut time = t0;
      out.slice_mut(s![p, 0, ..]).assign(&x);
      for i in 1..=steps {
        self.fill_gauss_increment(&mut d_w_full, sqrt_dt, rng);
        // Stage 1: evaluate at (x, t)
        let k1_mu = (self.drift)(&x, time);
        let k1_sig = (self.diffusion)(&x, time);
        // Stage 2: evaluate at (x + dt/2 * k1_mu + dW/2 * k1_sig, t + dt/2)
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
        // Stage 3: evaluate at (x + dt/2 * k2_mu + dW/2 * k2_sig, t + dt/2)
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
        // Stage 4: evaluate at (x + dt * k3_mu + dW * k3_sig, t + dt)
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
        // Final update: RK4 weighted average
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
        out.slice_mut(s![p, i, ..]).assign(&x);
      }
    }
    out
  }

  /// Euler–Maruyama with fractional Brownian noise.
  fn solve_euler_fractional(&self, x0: &Array1<T>, t0: T, dt: T, incs: &Array3<T>) -> Array3<T> {
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
  fn solve_milstein_fractional(&self, x0: &Array1<T>, t0: T, dt: T, incs: &Array3<T>) -> Array3<T> {
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
  fn solve_srk2_fractional(&self, x0: &Array1<T>, t0: T, dt: T, incs: &Array3<T>) -> Array3<T> {
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
  fn solve_srk4_fractional(&self, x0: &Array1<T>, t0: T, dt: T, incs: &Array3<T>) -> Array3<T> {
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

#[cfg(test)]
mod tests {
  use ndarray::Array2;
  use ndarray::array;
  use rand::SeedableRng;
  use rand::rngs::StdRng;

  use super::*;

  /// Geometric Brownian motion `dS = μS dt + σS dW` has analytical mean
  /// `E[S_T] = S_0 · exp(μ·T)`. Each method should hit it within MC tolerance.
  fn gbm_sde()
  -> Sde<f64, impl Fn(&Array1<f64>, f64) -> Array1<f64>, impl Fn(&Array1<f64>, f64) -> Array2<f64>>
  {
    let mu = 0.05;
    let sigma = 0.20;
    Sde::new(
      move |x: &Array1<f64>, _t: f64| array![mu * x[0]],
      move |x: &Array1<f64>, _t: f64| Array2::from_elem((1, 1), sigma * x[0]),
      NoiseModel::Gaussian,
      None,
    )
  }

  fn final_mean(paths: &ndarray::Array3<f64>) -> f64 {
    let n_paths = paths.shape()[0];
    let last = paths.shape()[1] - 1;
    let sum: f64 = (0..n_paths).map(|p| paths[[p, last, 0]]).sum();
    sum / n_paths as f64
  }

  #[test]
  fn euler_gbm_recovers_analytical_mean() {
    let sde = gbm_sde();
    let s0 = array![100.0_f64];
    let t = 1.0_f64;
    let dt = 1e-3_f64;
    let mut rng = StdRng::seed_from_u64(0x5DE_F00D);
    let paths = sde.solve(&s0, 0.0, t, dt, 4_000, SdeMethod::Euler, &mut rng);

    let analytic = 100.0_f64 * (0.05_f64 * t).exp();
    let m = final_mean(&paths);
    let rel_err = (m - analytic).abs() / analytic;
    assert!(rel_err < 0.02, "Euler relative error {rel_err}");
  }

  #[test]
  fn milstein_gbm_recovers_analytical_mean() {
    let sde = gbm_sde();
    let s0 = array![100.0_f64];
    let t = 1.0_f64;
    let dt = 1e-3_f64;
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let paths = sde.solve(&s0, 0.0, t, dt, 4_000, SdeMethod::Milstein, &mut rng);

    let analytic = 100.0_f64 * (0.05_f64 * t).exp();
    let m = final_mean(&paths);
    let rel_err = (m - analytic).abs() / analytic;
    assert!(rel_err < 0.02, "Milstein relative error {rel_err}");
  }

  #[test]
  fn srk2_and_srk4_produce_finite_paths() {
    let sde = gbm_sde();
    let s0 = array![100.0_f64];
    let mut rng = StdRng::seed_from_u64(0xBADCAFE);
    for method in [SdeMethod::SRK2, SdeMethod::SRK4] {
      let paths = sde.solve(&s0, 0.0, 0.5, 1e-3, 200, method, &mut rng);
      assert!(paths.iter().all(|v| v.is_finite()));
      // Initial condition preserved.
      for p in 0..paths.shape()[0] {
        assert!((paths[[p, 0, 0]] - 100.0).abs() < 1e-12);
      }
    }
  }

  /// Pure drift (σ ≡ 0) collapses the SDE to an ODE; Euler should track
  /// `dx/dt = -k·x` exactly to first order and bracket the analytical
  /// `e^{-k·T}` decay.
  #[test]
  fn pure_drift_decay_tracks_exponential() {
    let k = 1.0;
    let sde = Sde::new(
      move |x: &Array1<f64>, _t: f64| array![-k * x[0]],
      |_x: &Array1<f64>, _t: f64| Array2::<f64>::zeros((1, 1)),
      NoiseModel::Gaussian,
      None,
    );
    let s0 = array![1.0_f64];
    let t = 1.0_f64;
    let dt = 1e-3_f64;
    let mut rng = StdRng::seed_from_u64(0xDEC0DE);
    let paths = sde.solve(&s0, 0.0, t, dt, 1, SdeMethod::Euler, &mut rng);

    let last = paths.shape()[1] - 1;
    let computed = paths[[0, last, 0]];
    let analytic = (-k * t).exp();
    assert!(
      (computed - analytic).abs() < 5e-3,
      "Euler decay {computed} vs analytic {analytic}"
    );
  }

  /// Initial condition copied to row 0 across all paths and methods.
  #[test]
  fn initial_condition_preserved() {
    let sde = gbm_sde();
    let s0 = array![100.0_f64];
    let mut rng = StdRng::seed_from_u64(7);
    let paths = sde.solve(&s0, 0.0, 0.1, 1e-3, 5, SdeMethod::Euler, &mut rng);
    for p in 0..5 {
      assert_eq!(paths[[p, 0, 0]], 100.0);
    }
  }
}
