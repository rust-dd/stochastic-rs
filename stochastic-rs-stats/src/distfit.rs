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
use crate::optim::nelder_mead_vec;
use crate::optim::search::by_differences;
use crate::optim::simplex::SimplexStop;
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
    .map(|v| v.to_f64().unwrap())
    .collect::<Vec<f64>>();
  validate(&x, minimum)?;
  reject_constant(&x, "constant sample")?;
  let (x, unit) = unit_scaled(x);
  let n = x.len();
  let mean = x.iter().sum::<f64>() / n as f64;
  let sd = (x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
  Ok(Sample { x, mean, sd, unit })
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

/// Johnson SU's and variance gamma's search: the best of `starts`, a simplex and its restart, the
/// L-BFGS polish; the optimum, the iterations, a simplex's verdict.
pub(crate) fn minimise(
  starts: &[Vec<f64>],
  objective: impl Fn(&[f64]) -> f64,
) -> (Vec<f64>, usize, bool) {
  const CAP: usize = 5_000;
  let start = best_start(starts, &objective).expect("a non-empty starting grid");
  let (theta, first, met) = nelder_mead_vec(start, CAP, &objective);
  let (theta, second, met_again) = nelder_mead_vec(&theta, CAP, &objective);
  let (theta, polish) = by_differences(&objective, theta);
  (theta, first + second + polish, met || met_again)
}

/// The message of that search, which a simplex ends by its spread or by its cap.
pub(crate) fn words(converged: bool) -> String {
  let stop = if converged {
    SimplexStop::Converged
  } else {
    SimplexStop::IterationCap
  };
  stop.message().into()
}
