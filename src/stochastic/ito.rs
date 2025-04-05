use ndarray::Array1;
use rand_distr::StandardNormal;

pub struct DiffusionProcessFn {
  pub drift: Box<dyn Fn(f64, f64) -> f64>,
  pub diffusion: Box<dyn Fn(f64, f64) -> f64>,
}

pub struct Function2D {
  pub eval: Box<dyn Fn(f64, f64) -> f64>,
}

pub struct ItoResult {
  pub drift_term: f64,
  pub diffusion_term: f64,
}

pub struct ItoCalculator {
  pub process: DiffusionProcessFn,
  pub function: Function2D,
  pub h: f64,
}

impl ItoCalculator {
  pub fn new(process: DiffusionProcessFn, function: Function2D, h: f64) -> Self {
    ItoCalculator {
      process,
      function,
      h,
    }
  }

  fn dfdx(&self, t: f64, x: f64) -> f64 {
    (self.function.eval)(t, x + self.h) - (self.function.eval)(t, x - self.h) / (2.0 * self.h)
  }

  fn d2fdx2(&self, t: f64, x: f64) -> f64 {
    ((self.function.eval)(t, x + self.h) - 2.0 * (self.function.eval)(t, x)
      + (self.function.eval)(t, x - self.h))
      / self.h.powi(2)
  }

  fn dfdt(&self, t: f64, x: f64) -> f64 {
    ((self.function.eval)(t + self.h, x) - (self.function.eval)(t - self.h, x)) / (2.0 * self.h)
  }

  pub fn ito_transform(&self, t: f64, x: f64) -> ItoResult {
    let mu = (self.process.drift)(t, x);
    let sigma = (self.process.diffusion)(t, x);

    let dfdx = self.dfdx(t, x);
    let d2fdx2 = self.d2fdx2(t, x);
    let dfdt = self.dfdt(t, x);

    let drift_term = mu * dfdt + 0.5 * sigma.powi(2) * d2fdx2;
    let diffusion_term = sigma * dfdx;

    ItoResult {
      drift_term,
      diffusion_term,
    }
  }

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

    while t < t1 {
      let mu = (self.process.drift)(t, x);
      let sigma = (self.process.diffusion)(t, x);
      let dw = sigma * dt.sqrt() * rng.sample::<f64, _>(StandardNormal);

      x += mu * dt + dw;
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

  #[test]
  fn plot_simulated_trajectory() {
    let process = DiffusionProcessFn {
      drift: Box::new(|_t, x| 2.0 * (1.0 - x)),
      diffusion: Box::new(|_t, _x| 0.3),
    };

    let function = Function2D {
      eval: Box::new(|_t, x| x * x.sin()),
    };

    let calc = ItoCalculator::new(process, function, 1e-5);
    let mut rng = thread_rng();

    let data = calc.simulate(0.0, 0.0, 1.0, 0.01, &mut rng);
    let data = data.map(|(.., x)| x.to_owned());
    let indices = (0..data.len()).collect();

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
