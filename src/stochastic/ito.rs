use ndarray::Array1;
use rand_distr::StandardNormal;

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
  pub drift_term: f64,
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
  pub process: DiffusionProcessFn,
  pub function: Function2D,
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
    rng: &mut impl rand::Rng,
  ) -> Array1<(f64, f64)> {
    let mut t = t0;
    let mut x = x0;
    let mut path = vec![(t, x)];

    // Loop until we reach or surpass the final time t1.
    while t < t1 {
      let mu = (self.process.drift)(t, x);
      let sigma = (self.process.diffusion)(t, x);
      // Brownian increment ~ Normal(0, dt).
      let dw = sigma * dt.sqrt() * rng.sample::<f64, _>(StandardNormal);

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
  use super::*;
  use plotly::common::{Line, LineShape, Mode};
  use plotly::{Plot, Scatter};
  use rand::thread_rng;
  use rand_distr::Distribution;

  /// This test demonstrates how to simulate and plot a single trajectory of the SDE.
  /// It uses the `DiffusionProcessFn`, `Function2D`, and `ItoCalculator` types.
  #[test]
  fn plot_simulated_trajectory() {
    // Define the SDE: drift, diffusion, and a jump term.
    let process = DiffusionProcessFn {
      drift: Box::new(|_t, x| 2.0 * (1.0 - x)), // Ornstein-Uhlenbeck style drift
      diffusion: Box::new(|_t, _x| 0.3),        // Constant diffusion
      jump_term: Some(Box::new(|_t, _x, dt| {
        // Demonstration: Poisson-like jump intensity times a small normal jump
        let lambda = 1.0;
        let expected_jumps = lambda * dt;
        let normal = rand_distr::Normal::new(0.0, 0.1).unwrap();
        let mut rng = rand::thread_rng();
        let num_jumps = expected_jumps.round() as usize;
        (0..num_jumps).map(|_| normal.sample(&mut rng)).sum::<f64>()
      })),
    };

    // Define a function f(t,x) = x*sin(x), just for demonstration
    let function = Function2D {
      eval: Box::new(|_t, x| x * x.sin()),
    };

    // Create the ItoCalculator with a small finite-difference step.
    let calc = ItoCalculator::new(process, function, 1e-5);
    let mut rng = thread_rng();

    // Simulate from t=0 to t=1 with steps of 0.01
    let data = calc.simulate(0.0, 0.0, 1.0, 0.01, &mut rng);
    // Extract the x-values (the states) for plotting
    let data = data.map(|(.., x)| x.to_owned());
    // x-axis indices for plotting
    let indices = (0..data.len()).collect();

    // Plot with Plotly
    let trace = Scatter::new(indices, data.to_vec())
      .mode(Mode::Lines)
      .line(Line::new().color("orange").shape(LineShape::Linear))
      .name("X(t)")
      .show_legend(true);

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.show();
  }
}
