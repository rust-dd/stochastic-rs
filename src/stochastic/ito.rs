//! # Ito
//!
//! $$
//! df(t,X_t)=\left(f_t+a f_x+\tfrac12 b^2 f_{xx}\right)dt + b f_x dW_t
//! $$
//!
use ndarray::Array1;

use crate::traits::FloatExt;

/// A structure defining the drift and diffusion functions of the SDE.
/// Optionally, a jump term can be added (e.g., for jump-diffusion models).
///
/// * `drift(t, x) -> f64`: The drift function \(\mu(t, x)\).
/// * `diffusion(t, x) -> f64`: The diffusion function \(\sigma(t, x)\).
/// * `jump_term(t, x, dt) -> f64`: A user-provided jump function. If provided, it
///   should return the jump increment for the process within the time step `dt`.
pub struct DiffusionProcessFn {
  pub drift: Box<dyn Fn(f64, f64) -> f64>,
  pub diffusion: Box<dyn Fn(f64, f64) -> f64>,
  pub jump_term: Option<Box<dyn Fn(f64, f64, f64) -> f64>>,
}

/// A 2D function \(\, f(t, x) \), used for Ito's lemma transformations.
/// * `eval(t, x) -> f64`: Evaluate the function at the point \((t, x)\).
pub struct Function2D {
  pub eval: Box<dyn Fn(f64, f64) -> f64>,
}

/// A simple result structure holding the drift and diffusion terms from the Ito transformation.
pub struct ItoResult {
  /// Model parameter controlling process dynamics.
  pub drift_term: f64,
  /// Model parameter controlling process dynamics.
  pub diffusion_term: f64,
}

/// This calculator applies finite-difference approximations to the given `function` in order
/// to compute partial derivatives for Ito's lemma. It also simulates trajectories of the SDE
/// defined by `process`.
///
/// # Fields
/// * `process` - The stochastic process (drift, diffusion, jump).
/// * `function` - The 2D function \(f(t, x)\) for Ito's lemma.
/// * `h` - A small step used in the finite-difference approximation.
pub struct ItoCalculator {
  /// Model parameter controlling process dynamics.
  pub process: DiffusionProcessFn,
  /// Model parameter controlling process dynamics.
  pub function: Function2D,
  /// Model parameter controlling process dynamics.
  pub h: f64,
}

impl ItoCalculator {
  /// Creates a new instance of the ItoCalculator with the specified process, function, and step size `h`.
  ///
  /// * `process` - Contains the drift, diffusion, and optional jump term.
  /// * `function` - The function to be used in the Ito transformation.
  /// * `h` - The finite-difference step for numerical approximations of derivatives.
  pub fn new(process: DiffusionProcessFn, function: Function2D, h: f64) -> Self {
    ItoCalculator {
      process,
      function,
      h,
    }
  }

  /// Numerically approximates the partial derivative \(\frac{\partial f}{\partial x}\)
  /// using central differences:
  /// \[
  /// \frac{f(t, x + h) - f(t, x - h)}{2h}.
  /// \]
  fn dfdx(&self, t: f64, x: f64) -> f64 {
    ((self.function.eval)(t, x + self.h) - (self.function.eval)(t, x - self.h)) / (2.0 * self.h)
  }

  /// Numerically approximates the second partial derivative \(\frac{\partial^2 f}{\partial x^2}\)
  /// using central differences:
  /// \[
  /// \frac{f(t, x + h) - 2 f(t, x) + f(t, x - h)}{h^2}.
  /// \]
  fn d2fdx2(&self, t: f64, x: f64) -> f64 {
    ((self.function.eval)(t, x + self.h) - 2.0 * (self.function.eval)(t, x)
      + (self.function.eval)(t, x - self.h))
      / self.h.powi(2)
  }

  /// Numerically approximates the partial derivative \(\frac{\partial f}{\partial t}\)
  /// using central differences:
  /// \[
  /// \frac{f(t + h, x) - f(t - h, x)}{2h}.
  /// \]
  fn dfdt(&self, t: f64, x: f64) -> f64 {
    ((self.function.eval)(t + self.h, x) - (self.function.eval)(t - self.h, x)) / (2.0 * self.h)
  }

  /// Applies Ito's lemma to compute the drift and diffusion terms for the transformation
  /// of \(f(t, x)\), where:
  ///
  /// \[
  ///    df = \frac{\partial f}{\partial t} \, dt
  ///         + \mu(t, x) \frac{\partial f}{\partial x} \, dt
  ///         + \frac{1}{2} \sigma^2(t, x) \frac{\partial^2 f}{\partial x^2}\, dt
  ///         + \sigma(t, x) \frac{\partial f}{\partial x}\, dW_t.
  /// \]
  ///
  /// # Returns
  /// * `drift_term` - The combined coefficient in front of `dt`.
  /// * `diffusion_term` - The coefficient in front of `dW_t`.
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

  /// Simulates a path for the SDE given by:
  /// \[
  ///   dX_t = \mu(t, X_t) \, dt + \sigma(t, X_t) \, dW_t + \text{(optional jump)}.
  /// \]
  ///
  /// This uses a simple Euler–Maruyama scheme, optionally adding jump increments.
  ///
  /// # Arguments
  /// * `x0` - The initial value \(X_{t_0}\).
  /// * `t0` - The starting time.
  /// * `t1` - The final time.
  /// * `dt` - The time step for the numerical simulation.
  /// * `rng` - A random number generator that implements `rand::Rng`.
  ///
  /// # Returns
  /// An `ndarray::Array1` of `(time, value)` tuples representing the simulated path.
  pub fn simulate(
    &self,
    x0: f64,
    t0: f64,
    t1: f64,
    dt: f64,
    _rng: &mut impl rand::Rng,
  ) -> Array1<(f64, f64)> {
    let steps = ((t1 - t0) / dt).ceil().max(0.0) as usize;
    let sqrt_dt = dt.sqrt();
    let mut normals = vec![0.0; steps];
    <f64 as FloatExt>::fill_standard_normal_slice(&mut normals);

    let mut t = t0;
    let mut x = x0;
    let mut path = Vec::with_capacity(steps + 1);
    path.push((t, x));

    // Loop over the precomputed Gaussian increments.
    for z in normals {
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
