//! Maximum-likelihood fits of the skewed return laws: Johnson SU, Hansen's skew-t, variance gamma.
//!
//! $$
//! \hat\theta = \arg\max_\theta \sum_{i=1}^{n}\log f(x_i \mid \theta),\qquad
//! \widehat{\mathrm{Cov}}(\hat\theta) = \bigl(-\nabla^2\ell(\hat\theta)\bigr)^{-1}.
//! $$
//!
//! The search is in unconstrained coordinates; the Hessian is a central difference, as in `evt`.
//!
//! Reference: Johnson (1949), Biometrika 36(1/2), 149-176, DOI: 10.1093/biomet/36.1-2.149
//! Reference: Hansen (1994), International Economic Review 35(3), 705-730, DOI: 10.2307/2527081
//! Reference: Madan, Carr & Chang (1998), Eur. Fin. Rev. 2(1), 79-105, DOI: 10.1023/A:1009703431535

pub mod johnson_su;
pub mod skew_t;
#[cfg(test)]
mod tests;
pub mod variance_gamma;

use std::convert::Infallible;

use basin::CostFunction;
use basin::Executor;
use basin::Gradient;
use basin::LbfgsState;
use basin::Lbfgsb;

pub use johnson_su::JohnsonSuFit;
pub use johnson_su::fit_johnson_su;
use ndarray::ArrayView1;

pub use skew_t::SkewTFit;
pub use skew_t::fit_skew_t;
pub use variance_gamma::VarianceGammaFit;
pub use variance_gamma::fit_variance_gamma;

use crate::StatsError;
use crate::input::reject_constant;
use crate::input::validate;
use crate::optim::best_start;
use crate::optim::more_thuente;
use crate::optim::nelder_mead_vec;
use crate::stationarity::common::unit_scaled;
use crate::traits::FloatExt;

/// The sample times `unit`, the power of two nearest the inverse of its largest magnitude, with its
/// mean and standard deviation there.
pub(crate) struct Sample {
  pub(crate) x: Vec<f64>,
  pub(crate) mean: f64,
  pub(crate) sd: f64,
  pub(crate) unit: f64,
}

/// The [`Sample`] of `data`; `Err` below `minimum`, on a non-finite observation or a constant one.
pub(crate) fn sample<T: FloatExt>(
  data: ArrayView1<T>,
  minimum: usize,
) -> Result<Sample, StatsError> {
  let x = data
    .iter()
    .map(|v| v.to_f64().unwrap_or(f64::NAN))
    .collect::<Vec<f64>>();
  validate(&x, minimum)?;
  reject_constant(&x, "constant sample")?;
  let (x, unit) = unit_scaled(x);
  let n = x.len();
  let mean = x.iter().sum::<f64>() / n as f64;
  let sd = (x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
  Ok(Sample { x, mean, sd, unit })
}

/// The best of `starts`, a Nelder–Mead simplex with one restart, then an L-BFGS polish kept only if
/// it lowers the objective; returns the optimum, the iterations and whether the simplex converged.
pub(crate) fn minimise<F: Fn(&[f64]) -> f64 + Clone>(
  starts: &[Vec<f64>],
  objective: F,
) -> (Vec<f64>, usize, bool) {
  let start = best_start(starts, &objective).expect("a non-empty starting grid");
  let (theta, iters, converged) = nelder_mead_vec(start, 5_000, &objective);
  let (theta, iters_restart, converged_restart) = nelder_mead_vec(&theta, 5_000, &objective);
  let (theta, iters_polish) = Polish {
    objective: objective.clone(),
  }
  .run(theta);
  (
    theta,
    iters + iters_restart + iters_polish,
    converged || converged_restart,
  )
}

/// The objective as a Basin problem with numerical gradients.
#[derive(Clone)]
struct Polish<F> {
  objective: F,
}

impl<F: Fn(&[f64]) -> f64 + Clone> Polish<F> {
  fn run(&self, theta: Vec<f64>) -> (Vec<f64>, usize) {
    let start_cost = (self.objective)(&theta);
    let solver = Lbfgsb::with_line_search(more_thuente())
      .unbounded()
      .with_absolute_gradient_tolerance(f64::EPSILON.sqrt())
      .with_absolute_cost_change_tolerance(f64::EPSILON);
    let state = LbfgsState::new(theta.clone(), 10);
    let result = Executor::new(self.clone(), solver, state)
      .max_iter(200)
      .run()
      .expect("distribution-fitting objective is infallible");
    let iters = result.iter() as usize;
    let best = result.best_param();
    if (self.objective)(best) < start_cost {
      (best.clone(), iters)
    } else {
      (theta, iters)
    }
  }
}

impl<F: Fn(&[f64]) -> f64 + Clone> CostFunction for Polish<F> {
  type Param = Vec<f64>;
  type Output = f64;
  type Error = Infallible;

  fn cost(&self, theta: &Self::Param) -> Result<Self::Output, Self::Error> {
    Ok((self.objective)(theta))
  }
}

impl<F: Fn(&[f64]) -> f64 + Clone> Gradient for Polish<F> {
  type Gradient = Vec<f64>;

  fn gradient(&self, theta: &Self::Param) -> Result<Self::Gradient, Self::Error> {
    let mut grad = vec![0.0; theta.len()];
    for i in 0..theta.len() {
      let step = 1e-6 * (1.0 + theta[i].abs());
      let mut plus = theta.clone();
      let mut minus = theta.clone();
      plus[i] += step;
      minus[i] -= step;
      grad[i] = ((self.objective)(&plus) - (self.objective)(&minus)) / (2.0 * step);
    }
    Ok(grad)
  }
}

/// Negative log-likelihood from a per-observation log-density, `1e300`
/// when any term is not finite.
pub(crate) fn negative_log_likelihood<F: Fn(f64) -> f64>(x: &[f64], log_density: F) -> f64 {
  let mut total = 0.0;
  for &v in x {
    let l = log_density(v);
    if !l.is_finite() {
      return 1e300;
    }
    total -= l;
  }
  total
}
