//! Itô's lemma by finite differences for a user-supplied diffusion and function, `f64` only.
//!
//! $$
//! df(t,X_t)=\left(f_t+a f_x+\tfrac12 b^2 f_{xx}\right)dt + b f_x dW_t
//! $$
//!
//! The closures take `f64`, as `f32` loses accuracy in the central differences; `Sde` is generic.
use std::fmt;

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

/// The SDE's `drift(t, x)` and `diffusion(t, x)`, and an optional `jump_term(t, x, dt)`, the jump
/// increment over a step of `dt`.
pub struct DiffusionProcessFn {
  pub drift: Box<dyn Fn(f64, f64) -> f64 + Send + Sync>,
  pub diffusion: Box<dyn Fn(f64, f64) -> f64 + Send + Sync>,
  pub jump_term: Option<Box<dyn Fn(f64, f64, f64) -> f64 + Send + Sync>>,
}

/// A 2D function \(\, f(t, x) \), used for Ito's lemma transformations.
/// * `eval(t, x) -> f64`: Evaluate the function at the point \((t, x)\).
pub struct Function2D {
  pub eval: Box<dyn Fn(f64, f64) -> f64 + Send + Sync>,
}

impl fmt::Debug for DiffusionProcessFn {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("DiffusionProcessFn").finish_non_exhaustive()
  }
}

impl fmt::Debug for Function2D {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Function2D").finish_non_exhaustive()
  }
}

/// A simple result structure holding the drift and diffusion terms from the Ito transformation.
#[derive(Debug)]
pub struct ItoResult {
  /// Combined `dt`-coefficient `∂f/∂t + μ ∂f/∂x + ½σ² ∂²f/∂x²` from Ito's
  /// lemma applied to `f(t, X_t)`.
  pub drift_term: f64,
  /// Combined `dW_t`-coefficient `σ ∂f/∂x` from Ito's lemma applied to
  /// `f(t, X_t)`.
  pub diffusion_term: f64,
}

/// Applies Itô's lemma to `function` along the SDE `process`, with partial derivatives by central
/// differences of step `h`; it also simulates the SDE's paths.
#[derive(Debug)]
pub struct ItoCalculator {
  /// The underlying SDE's drift/diffusion/jump functions being
  /// transformed by Ito's lemma.
  pub process: DiffusionProcessFn,
  /// The twice-differentiable transformation `f(t, x)` Ito's lemma is
  /// applied to.
  pub function: Function2D,
  /// Finite-difference step size for the central-difference derivative
  /// approximations of `function`.
  pub h: f64,
}

impl ItoCalculator {
  /// `h` is the central-difference step of every derivative; it is not checked.
  pub fn new(process: DiffusionProcessFn, function: Function2D, h: f64) -> Self {
    ItoCalculator {
      process,
      function,
      h,
    }
  }

  /// `∂f/∂x` by the central difference `(f(t, x + h) − f(t, x − h)) / 2h`.
  fn dfdx(&self, t: f64, x: f64) -> f64 {
    ((self.function.eval)(t, x + self.h) - (self.function.eval)(t, x - self.h)) / (2.0 * self.h)
  }

  /// `∂²f/∂x²` by the central difference `(f(t, x + h) − 2 f(t, x) + f(t, x − h)) / h²`.
  fn d2fdx2(&self, t: f64, x: f64) -> f64 {
    ((self.function.eval)(t, x + self.h) - 2.0 * (self.function.eval)(t, x)
      + (self.function.eval)(t, x - self.h))
      / self.h.powi(2)
  }

  /// `∂f/∂t` by the central difference `(f(t + h, x) − f(t − h, x)) / 2h`.
  fn dfdt(&self, t: f64, x: f64) -> f64 {
    ((self.function.eval)(t + self.h, x) - (self.function.eval)(t - self.h, x)) / (2.0 * self.h)
  }

  /// Itô's lemma for `f(t, x)`: returns the coefficient of `dt` and the coefficient of `dW_t`.
  ///
  /// $$
  /// df = \left(f_t + \mu f_x + \tfrac12 \sigma^2 f_{xx}\right)dt + \sigma f_x\,dW_t
  /// $$
  pub fn ito_transform(&self, t: f64, x: f64) -> ItoResult {
    let mu = (self.process.drift)(t, x);
    let sigma = (self.process.diffusion)(t, x);

    let dfdx = self.dfdx(t, x);
    let d2fdx2 = self.d2fdx2(t, x);
    let dfdt = self.dfdt(t, x);

    // Drift term: df/dt + mu * df/dx + (1/2) sigma^2 d²f/dx²
    let drift_term = dfdt + mu * dfdx + 0.5 * sigma.powi(2) * d2fdx2;
    // Diffusion term: sigma * df/dx
    let diffusion_term = sigma * dfdx;

    ItoResult {
      drift_term,
      diffusion_term,
    }
  }

  /// An Euler–Maruyama path of `dX = μ dt + σ dW`, plus the jump term, from `x0` at `t0` to `t1` as
  /// `(t, x)` pairs; the Brownian increments are drawn from `seed`.
  pub fn simulate<S: SeedExt>(
    &self,
    x0: f64,
    t0: f64,
    t1: f64,
    dt: f64,
    seed: &S,
  ) -> Array1<(f64, f64)> {
    let steps = ((t1 - t0) / dt).ceil().max(0.0) as usize;
    let sqrt_dt = dt.sqrt();
    let normals = SimdNormal::<f64>::new(0.0, 1.0)
      .seeded(seed)
      .sample_n(steps);

    let mut t = t0;
    let mut x = x0;
    let mut path = Vec::with_capacity(steps + 1);
    path.push((t, x));

    // Loop over the precomputed Gaussian increments.
    for &z in normals.iter() {
      let mu = (self.process.drift)(t, x);
      let sigma = (self.process.diffusion)(t, x);
      // Brownian increment ~ Normal(0, dt).
      let dw = sigma * sqrt_dt * z;

      // If a jump_term is defined, add its contribution over the interval dt.
      let jump = if let Some(jump_fn) = &self.process.jump_term {
        jump_fn(t, x, dt)
      } else {
        0.0
      };

      // Euler–Maruyama update:
      x += mu * dt + dw + jump;
      t += dt;
      path.push((t, x));
    }

    Array1::from(path)
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::*;

  fn ou_squared() -> ItoCalculator {
    ItoCalculator::new(
      DiffusionProcessFn {
        drift: Box::new(|_, x| -x),
        diffusion: Box::new(|_, _| 0.3),
        jump_term: None,
      },
      Function2D {
        eval: Box::new(|_, x| x * x),
      },
      1e-4,
    )
  }

  #[test]
  fn a_seeded_path_is_a_function_of_the_seed() {
    let ito = ou_squared();
    let path = |seed: &Deterministic| ito.simulate(1.0, 0.0, 1.0, 0.01, seed);
    assert_eq!(path(&Deterministic::new(42)), path(&Deterministic::new(42)));
    assert_ne!(path(&Deterministic::new(42)), path(&Deterministic::new(43)));
    let fresh = || ito.simulate(1.0, 0.0, 1.0, 0.01, &Unseeded);
    assert_ne!(fresh(), fresh());
  }

  #[test]
  fn the_calculator_crosses_threads() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<ItoCalculator>();
  }
}
